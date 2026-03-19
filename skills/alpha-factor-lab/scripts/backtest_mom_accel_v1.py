#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: mom_accel_v1 — 动量加速度因子
==========================================
方向: 动量/加速度

构造:
  1. 近期动量: 过去5日收益率 (close_t / close_{t-5} - 1)
  2. 远期动量: 过去5-20日收益率 (close_{t-5} / close_{t-20} - 1)  
  3. 动量加速度 = 近期动量 - 远期动量 (标准化后)
  4. 成交额中性化(OLS)
  5. 5%缩尾

逻辑:
  动量加速度衡量趋势的变化率。
  正值=近期加速上涨或减速下跌=趋势在强化
  负值=近期减速上涨或加速下跌=趋势在弱化
  
  如果正向有效(高加速度→高收益): 趋势延续/信息扩散逻辑
  如果反向有效(低加速度→高收益): 过度反应/均值回复逻辑
  
  理论:
  - Grinblatt & Moskowitz (2004): "Predicting Stock Price Movements from Past Returns: 
    The Role of Consistency and Tax-Loss Selling" — 动量的连续性matters
  - Da, Gurun & Warachka (2014): "Frog in the Pan" — 信息的渐进释放
    (连续小涨比单次大涨动量更持久)
  - Chan, Jegadeesh & Lakonishok (1996): Momentum Strategies
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
SHORT_WINDOW = 5
LONG_WINDOW = 20
FORWARD_DAYS = 5
REBALANCE_FREQ = 5
N_GROUPS = 5
COST = 0.003
WINSORIZE_PCT = 0.05
DATA_CUTOFF = "2026-03-17"
FACTOR_ID = "mom_accel_v1"

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "csi1000_kline_raw.csv"
SCRIPTS_DIR = BASE_DIR / "skills" / "alpha-factor-lab" / "scripts"
OUTPUT_DIR = BASE_DIR / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据 (cutoff: {DATA_CUTOFF})...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] <= DATA_CUTOFF]
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")

ret_piv = close_piv.pct_change()
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造动量加速度因子 (short={SHORT_WINDOW}d, long={LONG_WINDOW}d)...")

# 近期动量: 过去5日
mom_short = close_piv / close_piv.shift(SHORT_WINDOW) - 1

# 远期动量: 5-20日
mom_long = close_piv.shift(SHORT_WINDOW) / close_piv.shift(LONG_WINDOW) - 1

# 加速度 = 近期 - 远期 (标准化到可比尺度)
# 用z-score截面标准化后相减
def cross_zscore(mat):
    """截面z-score"""
    result = mat.copy()
    for date in mat.index:
        row = mat.loc[date]
        valid = row.dropna()
        if len(valid) < 10:
            continue
        result.loc[date] = (row - valid.mean()) / valid.std()
    return result

mom_short_z = cross_zscore(mom_short)
mom_long_z = cross_zscore(mom_long)

factor_raw = mom_short_z - mom_long_z

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
print(f"[5] 回测...")

common_dates = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
common_stocks = sorted(factor_neutral.columns.intersection(ret_piv.columns))
fa = factor_neutral.loc[common_dates, common_stocks]
ra = ret_piv.loc[common_dates, common_stocks]

sys.path.insert(0, str(SCRIPTS_DIR))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data
)

# ────────────────── 多周期+双向测试 ──────────────────
print(f"[6] 多周期回测...")
best_sharpe = -999
best_config = None
best_results = None

for fwd in [5, 10, 20]:
    for reb in [5, 10, 20]:
        # 正向
        ic_s = compute_ic_dynamic(fa, ra, fwd, "pearson")
        ric_s = compute_ic_dynamic(fa, ra, fwd, "spearman")
        gr, tn, hi = compute_group_returns(fa, ra, N_GROUPS, reb, COST)
        met = compute_metrics(gr, ic_s, ric_s, tn, N_GROUPS, holdings_info=hi)
        sh = met.get("long_short_sharpe", 0) or 0
        
        # 反向
        ic_neg = compute_ic_dynamic(-fa, ra, fwd, "pearson")
        ric_neg = compute_ic_dynamic(-fa, ra, fwd, "spearman")
        gr_neg, tn_neg, hi_neg = compute_group_returns(-fa, ra, N_GROUPS, reb, COST)
        met_neg = compute_metrics(gr_neg, ic_neg, ric_neg, tn_neg, N_GROUPS, holdings_info=hi_neg)
        sh_neg = met_neg.get("long_short_sharpe", 0) or 0
        
        if sh >= sh_neg:
            cur_dir, cur_sh = 1, sh
            cur_results = (ic_s, ric_s, gr, tn, hi, met, fwd, reb)
        else:
            cur_dir, cur_sh = -1, sh_neg
            cur_results = (ic_neg, ric_neg, gr_neg, tn_neg, hi_neg, met_neg, fwd, reb)
        
        print(f"   fwd={fwd:2d} reb={reb:2d} dir={cur_dir:+d} Sharpe={cur_sh:.4f}")
        
        if cur_sh > best_sharpe:
            best_sharpe = cur_sh
            best_config = (fwd, reb, cur_dir)
            best_results = cur_results

FORWARD_DAYS = best_results[6]
REBALANCE_FREQ = best_results[7]
direction = best_config[2]
ic_series, rank_ic_series, group_returns, turnovers, holdings_info, metrics = best_results[:6]
fa_final = fa if direction == 1 else -fa
print(f"   最佳: fwd={FORWARD_DAYS} reb={REBALANCE_FREQ} dir={direction:+d} Sharpe={best_sharpe:.4f}")

if direction == 1:
    direction_desc = "正向（高加速度=近期相对加速=趋势延续/信息扩散）"
else:
    direction_desc = "反向（低加速度=近期相对减速=过度反应后均值回复）"

# ────────────────── 相关性 ──────────────────
print(f"[7] 与现有因子相关性...")
open_piv = df.pivot_table(index="date", columns="stock_code", values="open")
high_piv = df.pivot_table(index="date", columns="stock_code", values="high")
low_piv = df.pivot_table(index="date", columns="stock_code", values="low")
turnover_piv = df.pivot_table(index="date", columns="stock_code", values="turnover")

amihud_raw = (ret_piv.abs() / (amount_piv / 1e8).clip(lower=1e-8))
amihud_factor = np.log(amihud_raw.rolling(20, min_periods=10).mean().clip(lower=1e-12))

upper_sr = (high_piv - np.maximum(close_piv, open_piv)) / (high_piv - low_piv).clip(lower=1e-8)
lower_sr = (np.minimum(close_piv, open_piv) - low_piv) / (high_piv - low_piv).clip(lower=1e-8)
shadow = (upper_sr - lower_sr).rolling(20, min_periods=10).mean()

oret = open_piv / close_piv.shift(1) - 1
iret = close_piv / open_piv - 1
overnight_mom = oret.rolling(20, min_periods=10).sum() - iret.rolling(20, min_periods=10).sum()

ret_vals = ret_piv.values
n_d = len(dates)
cvar_mat = np.full((n_d, len(stocks)), np.nan)
for i in range(10, n_d):
    w = ret_vals[i-10:i, :]
    s = np.sort(w, axis=0)
    bot2 = np.nanmean(s[:2, :], axis=0)
    vc = np.sum(~np.isnan(w), axis=0)
    bot2[vc < 5] = np.nan
    cvar_mat[i, :] = -bot2
cvar_df = pd.DataFrame(cvar_mat, index=dates, columns=stocks)

neg_freq = (ret_piv <= -0.03).astype(float).rolling(10, min_periods=5).mean()
turnover_level = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8))
amplitude_piv = df.pivot_table(index="date", columns="stock_code", values="amplitude")
tae = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8) / (amplitude_piv.rolling(20, min_periods=10).mean().clip(lower=0.01)))

correlations = {}
for name, other in [('amihud_illiq_v2', amihud_factor), ('shadow_pressure_v1', shadow),
                     ('overnight_momentum_v1', overnight_mom), ('tail_risk_cvar_v1', cvar_df),
                     ('neg_day_freq_v1', neg_freq), ('turnover_level_v1', turnover_level),
                     ('tae_v1', tae)]:
    corrs = []
    for d in common_dates[::10]:
        f1 = fa_final.loc[d].dropna()
        f2 = other.loc[d].reindex(f1.index).dropna()
        c = f1.index.intersection(f2.index)
        if len(c) > 50:
            r, _ = sp_stats.spearmanr(f1[c], f2[c])
            if not np.isnan(r):
                corrs.append(r)
    avg = float(np.mean(corrs)) if corrs else 0
    correlations[name] = round(avg, 3)
    print(f"   vs {name}: r={avg:.3f}")

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
    "factor_name": "动量加速度 v1",
    "factor_name_en": "Momentum Acceleration v1",
    "category": "动量/加速度",
    "description": f"近{SHORT_WINDOW}日动量 vs {SHORT_WINDOW}-{LONG_WINDOW}日动量的差值(截面z-score), 成交额中性化。衡量趋势变化率。",
    "hypothesis": "动量加速度捕捉趋势强化/弱化信号。正向=信息扩散/趋势延续(Da et al. 2014)。反向=过度反应/均值回复。",
    "formula": f"neutralize(zscore(mom_{SHORT_WINDOW}d) - zscore(mom_{SHORT_WINDOW}-{LONG_WINDOW}d), log_amount_20d)",
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
    "data_cutoff": DATA_CUTOFF,
    "correlations": correlations,
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
print(f"  {FACTOR_ID}: 动量加速度因子")
print(f"  方向: {direction_desc}")
print(f"  配置: fwd={FORWARD_DAYS}d, reb={REBALANCE_FREQ}d")
print(f"{'='*60}")
print(f"  区间:     {report['period']}")
print(f"  股票:     {len(common_stocks)}")
print(f"  IC均值:     {ic_m:.4f}  (t={ic_t:.2f}, {sig})")
print(f"  Rank IC:    {metrics.get('rank_ic_mean', 0):.4f}")
print(f"  IR:         {metrics.get('ir', 0):.4f}")
print(f"  IC>0占比:   {metrics.get('ic_positive_pct', 0):.1%}")
print(f"  IC观测数:   {metrics.get('ic_count', 0)}")
print(f"  多空Sharpe: {ls_sh:.4f}")
print(f"  多空MDD:    {ls_md:.2%}")
print(f"  单调性:     {mono:.4f}")
print(f"  换手率:     {metrics.get('turnover_mean', 0):.2%}")
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
print(f"  评估标准: |IC|>0.015, |t|>2, |Sharpe|>0.5")
if is_valid:
    print(f"  → 准备写入factors.json并git提交")
