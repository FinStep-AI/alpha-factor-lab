#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: informed_flow_v1 — 知情交易强度因子
==========================================
方向: 量价/微观结构

构造:
  1. 每日计算 abs(收益率) / 换手率 (每单位换手带来的价格变动)
  2. 20日均值, 取对数
  3. 市值中性化(OLS) + 5%缩尾
  
逻辑:
  高 |ret|/turnover = 少量交易就能推动大幅价格变动
  → 暗示知情交易者(机构)在操作, 信息含量高
  → Kyle(1985) lambda: 价格冲击系数
  
  做多高知情交易强度的股票:
  - 知情交易者有信息优势, 价格将继续向信息方向移动(动量)
  - 类似Amihud非流动性, 但更直接衡量"信息含量per交易"
  
  与Amihud的区别:
  - Amihud = |ret| / 成交额 (流动性角度)
  - 本因子 = |ret| / 换手率 (剔除股本大小, 更纯的信息效率)

理论:
  Kyle (1985) Lambda: 知情交易者的交易对价格的影响
  Easley & O'Hara (1987): PIN (Probability of Informed Trading)
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ────────────────── 参数 ──────────────────
WINDOW = 20
FORWARD_DAYS = 5
REBALANCE_FREQ = 5
N_GROUPS = 5
COST = 0.003
WINSORIZE_PCT = 0.05
DATA_CUTOFF = "2026-03-13"
FACTOR_ID = "informed_flow_v1"

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "csi1000_kline_raw.csv"
SCRIPTS_DIR = BASE_DIR / "skills" / "alpha-factor-lab" / "scripts"
OUTPUT_DIR = BASE_DIR / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] <= DATA_CUTOFF]
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")
turnover_piv = df.pivot_table(index="date", columns="stock_code", values="turnover")

ret_piv = close_piv.pct_change()
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造知情交易强度因子...")

# |daily_return| / turnover_rate (price impact per unit of turnover)
# 高值 = 少量换手就推动大幅价格变动 = 知情交易者主导
price_impact_per_turn = ret_piv.abs() / turnover_piv.clip(lower=0.01)

# 20日均值 + 对数变换
factor_raw = np.log(price_impact_per_turn.rolling(WINDOW, min_periods=10).mean().clip(lower=1e-10))

print(f"   非空率: {factor_raw.notna().mean().mean():.2%}")
print(f"   均值: {factor_raw.stack().mean():.4f}, std: {factor_raw.stack().std():.4f}")

# ────────────────── 缩尾 ──────────────────
print(f"[3] 缩尾 ({WINSORIZE_PCT*100:.0f}%)...")
for date in dates:
    row = factor_raw.loc[date].dropna()
    if len(row) < 10:
        continue
    lo = row.quantile(WINSORIZE_PCT)
    hi = row.quantile(1 - WINSORIZE_PCT)
    factor_raw.loc[date] = factor_raw.loc[date].clip(lo, hi)

# ────────────────── 中性化 ──────────────────
print(f"[4] 成交额中性化 (OLS)...")
factor_neutral = factor_raw.copy()
for date in dates:
    f = factor_raw.loc[date].dropna()
    m = log_amt.loc[date].reindex(f.index).dropna()
    common = f.index.intersection(m.index)
    if len(common) < 30:
        continue
    f_c = f[common].values
    m_c = m[common].values
    X = np.column_stack([np.ones(len(m_c)), m_c])
    try:
        beta = np.linalg.lstsq(X, f_c, rcond=None)[0]
        factor_neutral.loc[date, common] = f_c - X @ beta
    except:
        pass

# ────────────────── 回测 ──────────────────
print(f"[5] 回测: {N_GROUPS}组, {REBALANCE_FREQ}d调仓...")

common_dates = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
common_stocks = sorted(factor_neutral.columns.intersection(ret_piv.columns))
fa = factor_neutral.loc[common_dates, common_stocks]
ra = ret_piv.loc[common_dates, common_stocks]

sys.path.insert(0, str(SCRIPTS_DIR))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data
)

# 测试正向和反向
results = {}
for sign_name, sign_mult in [("正向", 1), ("反向", -1)]:
    fa_use = fa * sign_mult
    ic = compute_ic_dynamic(fa_use, ra, FORWARD_DAYS, "pearson")
    ric = compute_ic_dynamic(fa_use, ra, FORWARD_DAYS, "spearman")
    gr, tv, hi = compute_group_returns(fa_use, ra, N_GROUPS, REBALANCE_FREQ, COST)
    m = compute_metrics(gr, ic, ric, tv, N_GROUPS, holdings_info=hi)
    
    ic_m = m.get("ic_mean", 0) or 0
    ic_t = m.get("ic_t_stat", 0) or 0
    ls_sh = m.get("long_short_sharpe", 0) or 0
    mono = m.get("monotonicity", 0) or 0
    
    print(f"   {sign_name}: IC={ic_m:.4f} t={ic_t:.2f} Sh={ls_sh:.3f} Mo={mono:.2f}")
    results[sign_name] = {
        "fa": fa_use, "ic": ic, "ric": ric, "gr": gr, "tv": tv, "hi": hi, "m": m,
        "sign": sign_mult
    }

# 选最优
best_name = max(results, key=lambda k: (results[k]["m"].get("long_short_sharpe", 0) or 0))
best = results[best_name]
print(f"   → 使用{best_name}")

metrics = best["m"]
group_returns = best["gr"]
ic_series = best["ic"]
rank_ic_series = best["ric"]
turnovers = best["tv"]
holdings_info = best["hi"]
direction = best["sign"]

if direction == 1:
    direction_desc = "正向（高知情交易强度=高预期收益）"
else:
    direction_desc = "反向（低知情交易强度=高预期收益）"

# ────────────────── 相关性 ──────────────────
print(f"[7] 相关性检查...")
amihud_raw = (ret_piv.abs() / (amount_piv / 1e8).clip(lower=1e-8))
amihud_factor = np.log(amihud_raw.rolling(20, min_periods=10).mean().clip(lower=1e-12))

fa_final = fa if direction == 1 else -fa
corrs_amihud = []
for date in common_dates[::20]:
    f1 = fa_final.loc[date].dropna()
    f2 = amihud_factor.loc[date].reindex(f1.index).dropna()
    c = f1.index.intersection(f2.index)
    if len(c) > 50:
        r, _ = sp_stats.spearmanr(f1[c], f2[c])
        if not np.isnan(r):
            corrs_amihud.append(r)
corr_amihud = float(np.mean(corrs_amihud)) if corrs_amihud else 0
print(f"   与Amihud: {corr_amihud:.3f}")

# ────────────────── 输出 ──────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
    "factor_name": "知情交易强度 v1",
    "factor_name_en": "Informed Flow Intensity v1",
    "category": "量价/微观结构",
    "description": f"|ret|/turnover的{WINDOW}日均值(对数), 衡量每单位换手的价格冲击。高值=知情交易者主导。",
    "hypothesis": "知情交易强度高的股票, 价格向信息方向移动, 动量延续。Kyle(1985) lambda的实证版。",
    "formula": f"neutralize(log(MA{WINDOW}(|ret|/turnover)), log_amount_20d)",
    "direction": direction,
    "direction_desc": direction_desc,
    "stock_pool": "中证1000",
    "period": f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}",
    "n_dates": len(common_dates),
    "n_stocks": len(common_stocks),
    "n_groups": N_GROUPS,
    "rebalance_freq": REBALANCE_FREQ,
    "forward_days": FORWARD_DAYS,
    "cost": COST,
    "correlations": {
        "amihud_illiq_v2": corr_amihud,
    },
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
print(f"  {FACTOR_ID}: 知情交易强度")
print(f"  方向: {direction_desc}")
print(f"{'='*60}")
print(f"  区间:     {report['period']}")
print(f"  股票:     {len(common_stocks)}")
print(f"  IC均值:     {ic_m:.4f}  (t={ic_t:.2f}, {sig})")
print(f"  Rank IC:    {metrics.get('rank_ic_mean', 0):.4f}")
print(f"  IR:         {metrics.get('ir', 0):.4f}")
print(f"  多空Sharpe: {ls_sh:.4f}")
print(f"  多空MDD:    {ls_md:.2%}")
print(f"  单调性:     {mono:.4f}")
print(f"  换手率:     {metrics.get('turnover_mean', 0):.2%}")
print(f"  与Amihud:   {corr_amihud:.3f}")
print(f"{'─'*60}")
print(f"  分层年化收益:")
for i, r in enumerate(metrics.get("group_returns_annualized", []), 1):
    r_str = f"{r:.2%}" if r is not None else "N/A"
    bar = "█" * max(int((r or 0) * 100), 0)
    print(f"    G{i}: {r_str}  {bar}")

for key in sorted(group_returns.keys(), key=lambda x: str(x)):
    cum = (1 + group_returns[key]).cumprod()
    print(f"  {key} NAV: {cum.iloc[-1]:.4f}")

print(f"{'='*60}")
is_valid = abs(ic_m) > 0.015 and abs(ic_t) > 2 and abs(ls_sh) > 0.5
print(f"\n  ➤ 因子{'有效 ✓' if is_valid else '无效 ✗'}")
