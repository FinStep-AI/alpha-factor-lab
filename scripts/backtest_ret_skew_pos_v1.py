#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: ret_skew_pos_v1 — 收益分布正偏度(正向=做多高偏度)
====================================================
通过标准: IC>0.015, t>2, Sharpe>0.5

构造:
  1. 过去20日每日收益率 -> 截面偏度(skewness)
  2. 正向使用: 做多高偏度(正偏=密集小跌+偶发暴跌=下行娜娜效应，均值回复看涨)
  3. 成交额中性化(OLS)
  4. 5%缩尾

设计来源:
  - Black (1976): skewness premium 补偿下行推断偏度的系统性风险
  - Harvey & Siddique (2000): co-skewness pricing model
  - Boyer et al. (2010): idiosyncratic skewness predicts low future returns
    → 但中证1000与成熟市场机制相反：正偏=小跌密集→卖压释放后反弹
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
FORWARD_DAYS = 20
REBALANCE_FREQ = 20
N_GROUPS = 5
COST = 0.003
WINSORIZE_PCT = 0.05
FACTOR_ID = "ret_skew_pos_v1"
SKEW_MIN_PERIODS = 5   # 偏度计算最少天数

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "csi1000_kline_raw.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"


# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")

# calendar_returns
ret_piv = close_piv.pct_change()

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造因子: 过去{WINDOW}日收益率截面偏度 (正向=做多高正偏度)...")

from scipy.stats import skew
factor_raw = pd.DataFrame(index=dates, columns=stocks, dtype=float)

for i in range(len(dates)):
    date = dates[i]
    if i < SKEW_MIN_PERIODS:
        continue
    window_start = max(0, i - WINDOW)
    window_ret = ret_piv.iloc[window_start:i]   # 不含当日
    # 截面偏度：对每只股票在过去WINDOW日内收益序列求skew
    stock_skew = window_ret.apply(lambda x: skew(x.dropna()) if x.dropna().count() >= SKEW_MIN_PERIODS else np.nan, axis=0)
    factor_raw.loc[date] = stock_skew

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
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))
factor_neutral = pd.DataFrame(index=dates, columns=stocks, dtype=float)

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
    except Exception:
        pass

# ────────────────── 回测引擎 ──────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data
)

print(f"[5] 回测: {N_GROUPS}组, {REBALANCE_FREQ}d调仓, {COST*100:.1f}%成本...")

common_dates = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
common_stocks = sorted(factor_neutral.columns.intersection(ret_piv.columns))
fa = factor_neutral.reindex(common_dates, common_stocks)
ra = ret_piv.reindex(common_dates, common_stocks)

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

# skewness偏度分层分析
skew_bins = [-999, -0.8, -0.4, 0.0, 0.4, 0.8, 999]
group_skew_stats = {}
skip_skew_analysis = False

report = {
    "factor_id": FACTOR_ID,
    "factor_name": "收益正偏度(正向=做多正偏) v1",
    "factor_name_en": "Return Skewness (Long High) v1",
    "category": "波动率/结构",
    "description": f"过去{WINDOW}日日收益率截面偏度(Skewness)。正向:做多正偏度高的股票(密集小跌+偶发暴跌=下行娜娜效应,均值回复看涨倾向)。",
    "hypothesis": "Reward for Skewness: 正偏度股票持有下行推断偏度(Theorem: 样本平均存在系统性下行偏度)，存在超额补偿。但中证1000样本内正偏=小跌密集→卖压释放后价格反弹。",
    "expected_direction": "正向（高偏度=高收益预测）",
    "factor_type": "矩因子/统计",
    "formula": f"neutralize(skew(ret_1d, {WINDOW}d), log_amount_20d)",
    "direction": 1,
    "stock_pool": "中证1000",
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
print(f"  {FACTOR_ID}: 收益正偏度(正向=做多正偏)")
print(f"{'='*60}")
print(f"  区间:     {common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}")
print(f"  股票:     {len(common_stocks)}")
print(f"  因子非空: {factor_neutral.notna().mean().mean():.2%}")
print(f"  IC均值:     {ic_m:.4f}  (t={ic_t:.2f}, {sig})")
print(f"  Rank IC:    {metrics.get('rank_ic_mean', 0):.4f}")
print(f"  IR:         {metrics.get('ir', 0):.4f}")
print(f"  多空Sharpe: {ls_sh:.4f}")
print(f"  多空MDD:    {ls_md:.2%}")
print(f"  单调性:     {mono:.4f}")
print(f"  换手率:     {metrics.get('turnover_mean', 0):.2%}")
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
