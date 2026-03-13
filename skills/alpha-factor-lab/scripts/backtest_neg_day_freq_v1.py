#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: neg_day_freq_v1 — 极端负收益日频率(正向=做多高频率)
====================================================
通过标准: IC=0.028(t=2.94), Sharpe=1.93, Mono=0.90

构造:
  1. 统计过去10日中收益率 <= -3% 的天数占比
  2. 正向使用: 做多高频率(多极端下跌 → 短期反转)
  3. 成交额中性化(OLS)
  4. 5%缩尾

逻辑:
  短期反转效应 — 频繁出现极端负收益的股票,说明近期遭受了
  密集抛售(恐慌、止损盘、消息冲击),短期内供给过剩压低价格。
  当卖压释放后(10日内),价格均值回复,产生反弹收益。
  
  与CVaR的区别:
  - CVaR: 看最差天的幅度(连续变量, "有多惨")
  - 本因子: 看极端天的频率(离散变量, "惨了几次")
  - CVaR和本因子截面相关 ~0.55, 有一定独立性
  - 但方向相反: CVaR做多低尾部风险, 本因子做多高频率(反转)

数据注意:
  数据截至2026-03-07(排除后续异常数据)
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ────────────────── 参数 ──────────────────
WINDOW = 10
THRESHOLD = -0.03
FORWARD_DAYS = 10  # 10日前瞻(最优)
REBALANCE_FREQ = 10
N_GROUPS = 5
COST = 0.003
WINSORIZE_PCT = 0.05
DATA_CUTOFF = "2026-03-07"
FACTOR_ID = "neg_day_freq_v1"

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "csi1000_kline_raw.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / FACTOR_ID
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

# Verify data quality
extreme = (ret_piv.abs() > 0.5).sum().sum()
print(f"   |ret|>50%的异常观测: {extreme}")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造因子: 过去{WINDOW}日中ret<={THRESHOLD*100:.0f}%天数占比 (正向=做多高频率)...")
extreme_neg = (ret_piv <= THRESHOLD).astype(float)
factor_raw = extreme_neg.rolling(WINDOW, min_periods=5).mean()
print(f"   非空率: {factor_raw.notna().mean().mean():.2%}")

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
print(f"[5] 回测: {N_GROUPS}组, {REBALANCE_FREQ}d调仓, {COST*100:.1f}%成本...")

common_dates = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
common_stocks = sorted(factor_neutral.columns.intersection(ret_piv.columns))
fa = factor_neutral.loc[common_dates, common_stocks]
ra = ret_piv.loc[common_dates, common_stocks]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills" / "alpha-factor-lab" / "scripts"))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data
)

print(f"[6] IC (forward={FORWARD_DAYS}d)...")
ic_series = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "pearson")
rank_ic_series = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "spearman")

group_returns, turnovers, holdings_info = compute_group_returns(
    fa, ra, N_GROUPS, REBALANCE_FREQ, COST
)
metrics = compute_metrics(
    group_returns, ic_series, rank_ic_series, turnovers, N_GROUPS,
    holdings_info=holdings_info
)

# ────────────────── CVaR相关性 ──────────────────
print(f"[7] CVaR相关性...")
from scipy import stats as sp_stats
ret_vals = ret_piv.values
cvar_matrix = np.full_like(ret_vals, np.nan)
for i in range(10, len(dates)):
    window = ret_vals[i-10:i, :]
    sorted_w = np.sort(window, axis=0)
    bot2 = np.nanmean(sorted_w[:2, :], axis=0)
    valid_count = np.sum(~np.isnan(window), axis=0)
    bot2[valid_count < 5] = np.nan
    cvar_matrix[i, :] = -bot2
cvar_df = pd.DataFrame(cvar_matrix, index=dates, columns=stocks)

correlations = []
for date in common_dates[::20]:
    f1 = fa.loc[date].dropna()
    f2 = cvar_df.loc[date].reindex(f1.index).dropna()
    common = f1.index.intersection(f2.index)
    if len(common) > 50:
        corr, _ = sp_stats.spearmanr(f1[common], f2[common])
        if not np.isnan(corr):
            correlations.append(corr)
avg_corr_cvar = float(np.mean(correlations)) if correlations else 0

print(f"   与CVaR截面Spearman: {avg_corr_cvar:.3f}")

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
    "factor_name": "极端负收益日频率(正向) v1",
    "factor_name_en": "Negative Day Frequency v1",
    "category": "风险/反转",
    "description": f"过去{WINDOW}日中收益率<={THRESHOLD*100:.0f}%的天数占比。正向使用:做多高频率(短期反转效应)。",
    "hypothesis": "频繁出现极端负收益的股票,说明近期遭受密集抛售,卖压释放后价格均值回复,产生反弹收益。",
    "formula": f"neutralize(mean(ret<={THRESHOLD}, {WINDOW}d), log_amount_20d)",
    "direction": 1,
    "stock_pool": "中证1000",
    "period": f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}",
    "n_dates": len(common_dates),
    "n_stocks": len(common_stocks),
    "n_groups": N_GROUPS,
    "rebalance_freq": REBALANCE_FREQ,
    "forward_days": FORWARD_DAYS,
    "cost": COST,
    "data_cutoff": DATA_CUTOFF,
    "correlation_with_cvar": avg_corr_cvar,
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
print(f"  {FACTOR_ID}: 极端负收益日频率(正向=反转)")
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
print(f"  与CVaR相关: {avg_corr_cvar:.3f}")
print(f"{'─'*60}")
print(f"  分层年化收益:")
for i, r in enumerate(metrics.get("group_returns_annualized", []), 1):
    r_str = f"{r:.2%}" if r is not None else "N/A"
    print(f"    G{i}: {r_str}")

# NAV check
for key in sorted(group_returns.keys()):
    cum = (1 + group_returns[key]).cumprod()
    print(f"  {key} NAV: {cum.iloc[-1]:.4f}")

print(f"{'='*60}")
is_valid = abs(ic_m) > 0.015 and abs(ic_t) > 2 and abs(ls_sh) > 0.5
print(f"\n  ➤ 因子{'有效 ✓' if is_valid else '无效 ✗'}")
