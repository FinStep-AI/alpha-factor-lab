#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: ret_ts_skew_v1 — 收益率时序偏度因子
=======================================
方向: 反转/尾部风险（负侧偏度做多）

构造:
  1. 过去20日每日收益率序列 (ret_{t-19}...ret_t)
  2. 计算时序偏度: E[((x-μ)/σ)^3] = skewness
     - 正偏: 正向极端收益>负向 → 聪明钱过度乐观 → 后续反转向下
     - 负偏: 负向极端收益>正向 → 已充分抛售 → 后续反弹
  3. 负号: 做多负偏(已充分抛售→反弹)
  4. 成交额中性化(OLS) + 5%缩尾 + z-score

理论依据:
  - Bali, Engle & Murray (2016): Empirical Asset Pricing
  - Boyer, Mitton & Vorkink (2010): Expected Idiosyncratic Skewness
  - 时序负偏 → 短期反转效应已充分释放
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
DATA_CUTOFF = "2026-04-30"
FACTOR_ID = "ret_ts_skew_v1"

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "csi1000_kline_raw.csv"
SCRIPTS_DIR = BASE_DIR / "skills" / "alpha-factor-lab" / "scripts"
OUTPUT_DIR = BASE_DIR / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据 (cutoff: {DATA_CUTOFF})...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

# 剔除当前数据中的异常尾部 (仍有少量未来日期> cutoff)
df = df[df["date"] <= DATA_CUTOFF].copy()

# 预计算amplitude列 (如果不存在)
if "amplitude" not in df.columns:
    df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1).clip(lower=1e-8)

df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
open_piv  = df.pivot_table(index="date", columns="stock_code", values="open")
amount_piv= df.pivot_table(index="date", columns="stock_code", values="amount")
turnover_piv= df.pivot_table(index="date", columns="stock_code", values="turnover")
high_piv  = df.pivot_table(index="date", columns="stock_code", values="high")
low_piv   = df.pivot_table(index="date", columns="stock_code", values="low")

ret_piv = close_piv.pct_change()
log_amt   = np.log(amount_piv.rolling(20).mean().clip(lower=1))
log_amt_20d = log_amt.copy()
dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ────────────────── 时序偏度计算 ──────────────────
print(f"[2] 构造收益率时序偏度因子 (window={WINDOW}d)...")

# pandas rolling skewness
def rolling_skew_20d(series: pd.Series) -> pd.Series:
    """20日滚动时序偏度"""
    return series.rolling(WINDOW, min_periods=max(WINDOW-5, 10)).skew()

# 按股票分组计算
skew_matrix = pd.DataFrame(index=dates, columns=stocks, dtype=float)
for stock in stocks:
    s = ret_piv[stock]
    skew_s = rolling_skew_20d(s)
    skew_matrix[stock] = skew_s

factor_raw = skew_matrix.astype(float)
print(f"   非空率: {factor_raw.notna().mean().mean():.2%}")
tmp = factor_raw.stack().dropna()
print(f"   均值: {tmp.mean():.4f}, std: {tmp.std():.4f}")

# ────────────────── 中性化(OLS) ──────────────────
print(f"[3] 成交额中性化 (OLS)...")

# log_amount 取均值字段
amount_mean = amount_piv.rolling(20).mean()
log_amt_current = np.log(amount_mean.clip(lower=1))

factor_neutral = factor_raw.copy()
for date in dates:
    f = factor_raw.loc[date].dropna()
    m = log_amt_current.loc[date].reindex(f.index).dropna()
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

# ────────────────── 缩尾 ──────────────────
print(f"[4] 缩尾 ({WINSORIZE_PCT*100:.0f}%)...")
for date in dates:
    row = factor_neutral.loc[date].dropna()
    if len(row) < 10:
        continue
    lo = row.quantile(WINSORIZE_PCT)
    hi = row.quantile(1 - WINSORIZE_PCT)
    factor_neutral.loc[date] = factor_neutral.loc[date].clip(lo, hi)

# ────────────────── z-score ──────────────────
print(f"[5] Z-score标准化...")
factor_z = pd.DataFrame(index=dates, columns=stocks, dtype=float)
for date in dates:
    row = factor_neutral.loc[date].dropna()
    if len(row) < 10:
        continue
    mu, sigma = row.mean(), row.std()
    if sigma > 1e-12:
        factor_z.loc[date, row.index] = (row - mu) / sigma

# ────────────────── 方向确认 ──────────────────
print(f"[5b] 方向确认: 正向(高因子→高收益) vs 反向(低因子→高收益)...")

sys.path.insert(0, str(SCRIPTS_DIR))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data
)

common_dates = sorted(factor_z.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
common_stocks = sorted(factor_z.columns.intersection(ret_piv.columns))
fa_z = factor_z.loc[common_dates, common_stocks]
ra  = ret_piv.loc[common_dates, common_stocks]

ic_pos = compute_ic_dynamic(fa_z,          ra, FORWARD_DAYS, "pearson")
rank_ic_pos = compute_ic_dynamic(fa_z,      ra, FORWARD_DAYS, "spearman")
gr_pos, tn_pos, hi_pos = compute_group_returns(fa_z,          ra, N_GROUPS, REBALANCE_FREQ, COST)
m_pos = compute_metrics(gr_pos, ic_pos, rank_ic_pos, tn_pos, N_GROUPS, holdings_info=hi_pos)

ic_neg = compute_ic_dynamic(-fa_z,         ra, FORWARD_DAYS, "pearson")
rank_ic_neg = compute_ic_dynamic(-fa_z,     ra, FORWARD_DAYS, "spearman")
gr_neg, tn_neg, hi_neg = compute_group_returns(-fa_z,         ra, N_GROUPS, REBALANCE_FREQ, COST)
m_neg = compute_metrics(gr_neg, ic_neg, rank_ic_neg, tn_neg, N_GROUPS, holdings_info=hi_neg)

pos_sh = m_pos.get("long_short_sharpe", 0) or 0
neg_sh = m_neg.get("long_short_sharpe", 0) or 0
pos_ic = m_pos.get("ic_mean", 0) or 0
neg_ic = m_neg.get("ic_mean", 0) or 0

print(f"   正向(高时序正偏→高收益)  IC={pos_ic:.4f}  Sharpe={pos_sh:.4f}")
print(f"   反向(高时序负偏→高收益)  IC={neg_ic:.4f}  Sharpe={neg_sh:.4f}")

if neg_sh > pos_sh:
    direction = -1
    direction_desc = "反向（负时序偏度=高预期收益；已充分抛售→反弹）"
    ic_series      = ic_neg
    rank_ic_series = rank_ic_neg
    group_returns  = gr_neg
    turnovers      = tn_neg
    metrics        = m_neg
    fa_final       = -fa_z
    print(f"   → 使用反向 ✓ (负偏做多)")
else:
    direction = 1
    direction_desc = "正向（正时序偏度=高预期收益；趋势延续）"
    ic_series      = ic_pos
    rank_ic_series = rank_ic_pos
    group_returns  = gr_pos
    turnovers      = tn_pos
    metrics        = m_pos
    fa_final       = fa_z
    print(f"   → 使用正向 ✓ (正偏做多)")

# ────────────────── 与现有因子相关性 ──────────────────
print(f"[6] 与现有因子相关性...")

# Amihud
amihud_raw = (ret_piv.abs() / (amount_piv / 1e8).clip(lower=1e-8))
amihud_factor = np.log(amihud_raw.rolling(20, min_periods=10).mean().clip(lower=1e-12))

# Shadow pressure
upper_sr = (high_piv - np.maximum(close_piv, open_piv)) / (high_piv - low_piv).clip(lower=1e-8)
lower_sr = (np.minimum(close_piv, open_piv) - low_piv) / (high_piv - low_piv).clip(lower=1e-8)
shadow = (upper_sr - lower_sr).rolling(20, min_periods=10).mean()

# Overnight momentum
oret = open_piv / close_piv.shift(1) - 1
iret = close_piv / open_piv - 1
overnight_mom = oret.rolling(20, min_periods=10).sum() - iret.rolling(20, min_periods=10).sum()

# CVaR
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

# Neg day freq
neg_freq = (ret_piv <= -0.03).astype(float).rolling(10, min_periods=5).mean()

# Turnover level
turnover_level = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8))

# amp_level
amplitude_piv = df.pivot_table(index="date", columns="stock_code", values="amplitude")
amp_level = np.log(amplitude_piv.rolling(60, min_periods=30).mean().clip(lower=0.01))

correlations = {}
for name, other in [
    ('amihud_illiq_v2', amihud_factor),
    ('shadow_pressure_v1', shadow),
    ('overnight_momentum_v1', overnight_mom),
    ('tail_risk_cvar_v1', cvar_df),
    ('neg_day_freq_v1', neg_freq),
    ('turnover_level_v1', turnover_level),
    ('amp_level_v2', amp_level),
    # Additional in new factors
    ('vol_ret_align_v1', None),
    ('vwap_dev_v1', None),
    ('ma_disp_v1', None),
    ('tae_v1', None),
    ('price_mom_5d_v1', None),
]:
    if other is None:
        correlations[name] = None
        continue
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
    correlations[name] = round(avg, 3) if not np.isnan(avg) else None
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
    "factor_name": "时序收益率偏度 v1",
    "factor_name_en": "Return Time-Series Skewness v1",
    "category": "反转/尾部风险",
    "description": "过去20日每日收益率时序分布的偏度(时序 skewness)，负号取反。低时序负偏(高频已充分抛售)做多；高时序正偏(极端正收益多)做空。",
    "hypothesis": "时序负偏股票(频繁超跌)已具备充分抛售后短期反弹|均值回复;时序正偏股票(有极端正收益)面临盈利回吐压力。",
    "formula": f"neutralize(-skew(daily_ret, {WINDOW}d), log_amount_20d)",
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
gr_ann = metrics.get("group_returns_annualized", [None]*N_GROUPS)

print(f"\n{'='*60}")
print(f"  {FACTOR_ID}: 时序收益率偏度因子")
print(f"  方向: {direction_desc}")
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
for i, r in enumerate(gr_ann, 1):
    r_str = f"{r:.2%}" if r is not None else "N/A"
    bar = "█" * max(int((r or 0) * 100), 0)
    print(f"    G{i}: {r_str}  {bar}")
print(f"{'='*60}")

is_valid = abs(ic_m) > 0.015 and abs(ic_t) > 2 and abs(ls_sh) > 0.5
print(f"\n  ➤ 因子{'有效 ✓' if is_valid else '无效 ✗'}")
print(f"  评估标准: |IC|>0.015, |t|>2, |Sharpe|>0.5")
if is_valid:
    print(f"  → 准备写入factors.json并git提交")

sys.exit(0 if is_valid else 1)
