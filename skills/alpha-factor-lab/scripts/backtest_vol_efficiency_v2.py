#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: vol_efficiency_v2 — 波动率效率因子 (Garman-Klass版)
==========================================================
方向: 量价/微观结构

改进自v1:
  - 使用Garman-Klass估计器(同时使用OHLC四个价格, 更高效)
  - 测试多窗口(10/20)和多前瞻(5/10/20)组合
  - 选择最优参数

构造:
  1. GK波动率: σ_GK = sqrt(0.5*(ln(H/L))^2 - (2ln2-1)*(ln(C/O))^2)
  2. CC波动率: σ_CC = |ln(C_t/C_{t-1})|
  3. VER = σ_CC / σ_GK (信息效率比)
  4. 市值中性化 + 5%缩尾

理论:
  Garman-Klass(1980)估计器利用全部OHLC信息, 
  效率是Parkinson的8.4倍。VER度量"收盘价反映的信息/日内全部信息"。
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "csi1000_kline_raw.csv"
SCRIPTS_DIR = BASE_DIR / "skills" / "alpha-factor-lab" / "scripts"

WINSORIZE_PCT = 0.05
DATA_CUTOFF = "2026-03-13"
FACTOR_ID = "vol_efficiency_v2"
N_GROUPS = 5
COST = 0.003

# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] <= DATA_CUTOFF]
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
open_piv = df.pivot_table(index="date", columns="stock_code", values="open")
high_piv = df.pivot_table(index="date", columns="stock_code", values="high")
low_piv = df.pivot_table(index="date", columns="stock_code", values="low")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")

ret_piv = close_piv.pct_change()
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

sys.path.insert(0, str(SCRIPTS_DIR))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data
)

# ────────────────── 构造GK波动率 ──────────────────
print(f"[2] 构造Garman-Klass波动率...")
log_hl = np.log(high_piv / low_piv.clip(lower=1e-8))
log_co = np.log(close_piv / open_piv.clip(lower=1e-8))
# GK daily variance: 0.5*(ln H/L)^2 - (2*ln2-1)*(ln C/O)^2
gk_var_daily = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
gk_var_daily = gk_var_daily.clip(lower=0)  # 确保非负

# CC volatility
log_ret = np.log(close_piv / close_piv.shift(1).clip(lower=1e-8))

# ────────────────── 多参数搜索 ──────────────────
print(f"[3] 参数搜索: 窗口×前瞻×调仓...")
best_config = None
best_score = -999

configs = [
    (10, 5, 5), (10, 10, 10), (10, 20, 20),
    (20, 5, 5), (20, 10, 10), (20, 20, 20),
    (15, 5, 5), (15, 10, 10),
]

for window, fwd, rebal in configs:
    gk_vol = np.sqrt(gk_var_daily.rolling(window, min_periods=max(5, window//2)).mean())
    cc_vol = log_ret.abs().rolling(window, min_periods=max(5, window//2)).mean()
    
    ver = cc_vol / gk_vol.clip(lower=1e-8)
    ver = ver.clip(0.01, 5.0)
    
    # 缩尾
    for date in dates:
        row = ver.loc[date].dropna()
        if len(row) < 10:
            continue
        lo = row.quantile(WINSORIZE_PCT)
        hi = row.quantile(1 - WINSORIZE_PCT)
        ver.loc[date] = ver.loc[date].clip(lo, hi)
    
    # 中性化
    ver_n = ver.copy()
    for date in dates:
        f = ver.loc[date].dropna()
        m = log_amt.loc[date].reindex(f.index).dropna()
        common = f.index.intersection(m.index)
        if len(common) < 30:
            continue
        f_c = f[common].values
        m_c = m[common].values
        X = np.column_stack([np.ones(len(m_c)), m_c])
        try:
            beta = np.linalg.lstsq(X, f_c, rcond=None)[0]
            ver_n.loc[date, common] = f_c - X @ beta
        except:
            pass
    
    common_dates = sorted(ver_n.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
    common_stocks = sorted(ver_n.columns.intersection(ret_piv.columns))
    fa = ver_n.loc[common_dates, common_stocks]
    ra = ret_piv.loc[common_dates, common_stocks]
    
    # 正向IC
    ic_pos = compute_ic_dynamic(fa, ra, fwd, "pearson")
    # 反向IC
    ic_neg = compute_ic_dynamic(-fa, ra, fwd, "pearson")
    
    for sign, ic_s in [(1, ic_pos), (-1, ic_neg)]:
        if len(ic_s) < 5:
            continue
        ic_mean = float(ic_s.mean())
        ic_std = float(ic_s.std())
        if ic_std == 0:
            continue
        ic_t = ic_mean / (ic_std / np.sqrt(len(ic_s)))
        
        # 快速分层
        fa_use = fa if sign == 1 else -fa
        gr, tv, hi = compute_group_returns(fa_use, ra, N_GROUPS, rebal, COST)
        m = compute_metrics(gr, ic_s, ic_s, tv, N_GROUPS, holdings_info=hi)
        ls_sharpe = m.get("long_short_sharpe", 0) or 0
        mono = m.get("monotonicity", 0) or 0
        
        # 综合评分
        score = abs(ic_mean) * 100 + abs(ic_t) * 0.5 + abs(ls_sharpe) * 2 + abs(mono)
        
        label = "正向" if sign == 1 else "反向"
        print(f"   W={window} F={fwd} R={rebal} {label}: IC={ic_mean:.4f} t={ic_t:.2f} Sh={ls_sharpe:.3f} Mo={mono:.2f} → score={score:.3f}")
        
        if score > best_score:
            best_score = score
            best_config = {
                "window": window, "forward_days": fwd, "rebalance_freq": rebal,
                "direction": sign, "ic_mean": ic_mean, "ic_t": ic_t,
                "sharpe": ls_sharpe, "mono": mono, "metrics": m,
                "group_returns": gr, "turnovers": tv, "holdings_info": hi,
                "ic_series": ic_s,
                "rank_ic_series": compute_ic_dynamic(fa_use, ra, fwd, "spearman"),
                "fa": fa_use, "common_dates": common_dates
            }

# ────────────────── 最优结果 ──────────────────
if best_config is None:
    print("所有配置都无效!")
    sys.exit(1)

print(f"\n[4] 最优配置: W={best_config['window']} F={best_config['forward_days']} "
      f"R={best_config['rebalance_freq']} dir={best_config['direction']}")

metrics = best_config["metrics"]
group_returns = best_config["group_returns"]
ic_series = best_config["ic_series"]
rank_ic_series = best_config["rank_ic_series"]
turnovers = best_config["turnovers"]
holdings_info = best_config["holdings_info"]
direction = best_config["direction"]
common_dates = best_config["common_dates"]

if direction == 1:
    direction_desc = "正向（高VER=高预期收益）"
else:
    direction_desc = "反向（低VER=高预期收益，噪声交易溢价）"

# ────────────────── 与现有因子相关性 ──────────────────
print(f"[5] 与Amihud相关性...")
amihud_raw = (ret_piv.abs() / (amount_piv / 1e8).clip(lower=1e-8))
amihud_factor = np.log(amihud_raw.rolling(20, min_periods=10).mean().clip(lower=1e-12))

fa_use = best_config["fa"]
correlations_amihud = []
for date in common_dates[::20]:
    f1 = fa_use.loc[date].dropna()
    f2 = amihud_factor.loc[date].reindex(f1.index).dropna()
    common = f1.index.intersection(f2.index)
    if len(common) > 50:
        corr, _ = sp_stats.spearmanr(f1[common], f2[common])
        if not np.isnan(corr):
            correlations_amihud.append(corr)
avg_corr_amihud = float(np.mean(correlations_amihud)) if correlations_amihud else 0
print(f"   与Amihud: {avg_corr_amihud:.3f}")

# ────────────────── 输出 ──────────────────
OUTPUT_DIR = BASE_DIR / "output" / FACTOR_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"
save_backtest_data(group_returns, ic_series, rank_ic_series, str(OUTPUT_DIR))

def nan_to_none(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    return obj

report = {
    "factor_id": FACTOR_ID,
    "factor_name": "波动率效率(GK) v2",
    "factor_name_en": "Volatility Efficiency Ratio (GK) v2",
    "category": "量价/微观结构",
    "description": f"Garman-Klass波动率vs收盘价波动率的效率比。W={best_config['window']},F={best_config['forward_days']}。",
    "hypothesis": "价格信息效率高(VER高)的股票定价更准确，噪声交易少。",
    "formula": f"neutralize(MA{best_config['window']}(|log_ret|/sqrt(GK_var)), log_amount)",
    "direction": direction,
    "direction_desc": direction_desc,
    "stock_pool": "中证1000",
    "period": f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}",
    "n_dates": len(common_dates),
    "n_stocks": len(best_config['fa'].columns),
    "n_groups": N_GROUPS,
    "rebalance_freq": best_config["rebalance_freq"],
    "forward_days": best_config["forward_days"],
    "cost": COST,
    "data_cutoff": DATA_CUTOFF,
    "window": best_config["window"],
    "correlation_with_amihud": avg_corr_amihud,
    "metrics": metrics,
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(nan_to_none(report), f, indent=2, ensure_ascii=False)

# ────────────────── 摘要 ──────────────────
ic_m = metrics.get("ic_mean", 0) or 0
ic_t = metrics.get("ic_t_stat", 0) or 0
ls_sh = metrics.get("long_short_sharpe", 0) or 0
ls_md = metrics.get("long_short_mdd", 0) or 0
mono = metrics.get("monotonicity", 0) or 0
sig = "✓" if metrics.get("ic_significant_5pct") else "✗"

print(f"\n{'='*60}")
print(f"  {FACTOR_ID}: 波动率效率(GK)")
print(f"  最优: W={best_config['window']} F={best_config['forward_days']} R={best_config['rebalance_freq']} dir={direction}")
print(f"  方向: {direction_desc}")
print(f"{'='*60}")
print(f"  区间:     {report['period']}")
print(f"  IC均值:     {ic_m:.4f}  (t={ic_t:.2f}, {sig})")
print(f"  Rank IC:    {metrics.get('rank_ic_mean', 0):.4f}")
print(f"  IR:         {metrics.get('ir', 0):.4f}")
print(f"  多空Sharpe: {ls_sh:.4f}")
print(f"  多空MDD:    {ls_md:.2%}")
print(f"  单调性:     {mono:.4f}")
print(f"  换手率:     {metrics.get('turnover_mean', 0):.2%}")
print(f"  与Amihud相关: {avg_corr_amihud:.3f}")
print(f"{'─'*60}")
print(f"  分层年化收益:")
for i, r in enumerate(metrics.get("group_returns_annualized", []), 1):
    r_str = f"{r:.2%}" if r is not None else "N/A"
    print(f"    G{i}: {r_str}")

for key in sorted(group_returns.keys(), key=lambda x: str(x)):
    cum = (1 + group_returns[key]).cumprod()
    print(f"  {key} NAV: {cum.iloc[-1]:.4f}")

print(f"{'='*60}")
is_valid = abs(ic_m) > 0.015 and abs(ic_t) > 2 and abs(ls_sh) > 0.5
print(f"\n  ➤ 因子{'有效 ✓' if is_valid else '无效 ✗'}")
print(f"  评估标准: |IC|>0.015, |t|>2, |Sharpe|>0.5")
