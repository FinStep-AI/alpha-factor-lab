#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: ret_skew_v1 — 收益率偏度因子 v1
================================================
构造:
  1. 日收益率 = pct_change / 100
  2. factor_raw = rolling_skew(ret, 20d)   ← 原始偏度
  3. 直接使用负偏度：factor = factor_raw（负偏→高因子值）
     （负偏=频繁小涨+偶有大跌，短期动量延续时做多负偏有正IC）
  4. 缩尾5% + 成交额OLS中性化 + MAD缩尾 + z-score

逻辑:
- 收益率偏度的第三矩。负偏度(左尾市长) = 频繁小涨为主+偶有大跌，
  20日窗口内以近期小涨为主导 → 短期动量延续效应 → 后续5日正收益
- 正偏度 = 频繁小亏为主+偶有大赚 → 大赚后回吐压力 → 后续负收益
- 与已有因子不重复: 填补"高阶矩"空档, 与CVaR/neg_min_ret角度独立

来源: 自研 | 参考: Bali et al.(2011) MAX/MIN effect, Boyer et al.(2010) Skewness
"""

import json
import sys
import time
import warnings
from pathlib import Path
from numpy.linalg import lstsq

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
FACTOR_ID = "ret_skew_v1"

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "skills" / "alpha-factor-lab" / "scripts"
DATA_PATH = BASE_DIR / "data" / "csi1000_kline_raw.csv"
OUTPUT_DIR = BASE_DIR / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

t0 = time.time()

# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
print(f"   数据范围: {df['date'].min().date()} ~ {df['date'].max().date()}")

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")
ret_piv = close_piv.pct_change()

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股  ({time.time()-t0:.1f}s)")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造收益率偏度因子 (window={WINDOW})...")
t0 = time.time()

# 20日滚动偏度 (pandas skew = Fisher-Pearson 样本偏度，ddof=1)
factor_raw = ret_piv.rolling(WINDOW, min_periods=int(WINDOW * 0.75)).skew()
print(f"   偏度均值: {factor_raw.stack().mean():.4f}  std: {factor_raw.stack().std():.4f}")
print(f"   非空率: {factor_raw.notna().mean().mean():.2%}  ({time.time()-t0:.1f}s)")

# 方向：直接使用负偏度（负偏→高因子值）
# 负偏=频繁小涨+偶有大跌 → 短期动量延续 → 正IC（见直觉测试）
# 正偏=频繁小亏+偶有大赚 → 大赚回吐 → 负IC
print(f"[2b] 方向: 直接使用 (负偏→高因子值 → 做多负偏股)")

# ────────────────── 缩尾 (截面MAD) ──────────────────
print(f"[3] 截面缩尾 ({WINSORIZE_PCT*100:.0f}%)...")
t0 = time.time()
for date in dates:
    row = factor_raw.loc[date].dropna()
    if len(row) < 10:
        continue
    lo = row.quantile(WINSORIZE_PCT)
    hi = row.quantile(1 - WINSORIZE_PCT)
    factor_raw.loc[date] = factor_raw.loc[date].clip(lo, hi)
print(f"   完成  ({time.time()-t0:.1f}s)")

# ────────────────── 成交额中性化 ──────────────────
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))
print(f"[4] 成交额中性化 (OLS + MAD + z-score)...")
factor_neutral = factor_raw.copy()
n_neutral = 0
t0 = time.time()
for date in dates:
    f_map = factor_raw.loc[date].dropna()
    m_map = log_amt.loc[date].reindex(f_map.index).dropna()
    common = f_map.index.intersection(m_map.index)
    if len(common) < 30:
        continue
    f_c = f_map[common].values.astype(float)
    m_c = m_map[common].values.astype(float)
    X = np.column_stack([np.ones(len(m_c)), m_c])
    try:
        beta, _, _, _ = lstsq(X, f_c, rcond=None)
        resid = f_c - X @ beta
        # MAD winsorize on residual
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        if mad > 1e-10:
            resid = np.clip(resid, med - 5 * 1.4826 * mad, med + 5 * 1.4826 * mad)
        # z-score
        r_mean = np.nanmean(resid)
        r_std = np.nanstd(resid)
        if r_std > 1e-10:
            factor_neutral.loc[date, common] = (resid - r_mean) / r_std
            n_neutral += 1
    except Exception:
        pass
print(f"   完成中性化: ~{n_neutral} 天  ({time.time()-t0:.1f}s)")

# ────────────────── 导入回测引擎 ──────────────────
sys.path.insert(0, str(SCRIPTS_DIR))
from factor_backtest import (
    compute_group_returns,
    compute_ic_dynamic,
    compute_metrics,
    save_backtest_data,
)

# ────────────────── 回测 ──────────────────
print(f"[5] 分层回测: {N_GROUPS}组, {REBALANCE_FREQ}天 freq, {COST*100:.1f}%成本...")
common_dates = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
common_stocks = sorted(factor_neutral.columns.intersection(ret_piv.columns))
factor_aligned = factor_neutral.loc[common_dates, common_stocks]
return_aligned = ret_piv.loc[common_dates, common_stocks]

print(f"[6] IC (forward={FORWARD_DAYS}d)...")
ic_series = compute_ic_dynamic(factor_aligned, return_aligned, FORWARD_DAYS, "pearson")
rank_ic_series = compute_ic_dynamic(factor_aligned, return_aligned, FORWARD_DAYS, "spearman")

group_returns, turnovers, holdings_info = compute_group_returns(
    factor_aligned, return_aligned, N_GROUPS, REBALANCE_FREQ, COST
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
    if isinstance(obj, (list, tuple)):
        return [nan_to_none(v) for v in obj]
    return obj

report = {
    "factor_id": FACTOR_ID,
    "factor_name": "收益率偏度 v1",
    "factor_name_en": "Return Skewness v1",
    "category": "高阶矩/动量变体",
    "description": f"过去{WINDOW}日收益率的滚动偏度(Fisher-Pearson, ddof=1)，直接使用负偏度(负偏→高因子值→做多)。负偏=频繁小涨+偶有大跌→短期动量延续效应。成交额OLS中性化。",
    "hypothesis": "负偏度股票（近期以小幅上涨为主）在A股中证1000小盘股上短期动量延续，后续5日继续上涨；正偏度股票（偶有大赚）有回吐压力。高阶矩偏度捕捉第三矩的预测性信息。",
    "expected_direction": "负偏度正向（负偏→高因子值→高预期收益）",
    "factor_type": "高阶矩/短期动量",
    "formula": f"neutralize(rolling_skew(pct_change, {WINDOW}d), log_amount_20d)",
    "direction": 1,
    "stock_pool": "中证1000",
    "period": f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}",
    "n_dates": len(common_dates),
    "n_stocks": len(common_stocks),
    "n_groups": N_GROUPS,
    "rebalance_freq": REBALANCE_FREQ,
    "forward_days": FORWARD_DAYS,
    "cost": COST,
    "data_cutoff": str(df["date"].max()),
    "barra_style": "Momentum",
    "source_type": "自研(论文启发)",
    "source_title": "Bali et al. (2011) MAX effect → 收益率偏度高阶矩因子",
    "source_url": "https://doi.org/10.1016/j.jfineco.2010.08.014",
    "correlations": {},
    "lessons_learned": [],
    "upgrade_notes": "v1初版。5日窗口偏度方向: 负偏→正IC",
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(nan_to_none(report), f, indent=2, ensure_ascii=False)

# ────────────────── 打印摘要 ──────────────────
print(f"\n{'═'*60}")
print(f"  Return Skewness v1 回测结果")
print(f"  (方向: 做多负偏度股)")
print(f"{'═'*60}")
print(f"  区间:      {report['period']}")
print(f"  股票/日期: {len(common_stocks)} / {len(common_dates)}")
ic_mean = abs(metrics.get("ic_mean", 0) or 0)
ic_t = abs(metrics.get("ic_t_stat", 0) or 0)
ls_sh = metrics.get("long_short_sharpe", 0) or 0
ls_md = metrics.get("long_short_mdd", 0) or 0
mono = metrics.get("monotonicity", 0) or 0
is_big = ic_mean > 0.015
is_t = ic_t > 2
is_sh = abs(ls_sh) > 0.5
icon_big, icon_t, icon_sh = ("✓" if is_big else "✗"), ("✓" if is_t else "✗"), ("✓" if is_sh else "✗")
n_pass = int(is_big) + int(is_t) + int(is_sh)
print(f"  IC均值:    {metrics.get('ic_mean', 0)*1e4:.1f}bp (t={metrics.get('ic_t_stat', 0):.2f})")
print(f'  IC显著:    5%{"✓" if metrics.get("ic_significant_5pct") else "✗"} 1%{"✓" if metrics.get("ic_significant_1pct") else "✗"}')
print(f"  Rank IC:   {metrics.get('rank_ic_mean', 0)*1e4:.1f}bp (t={metrics.get('rank_ic_t_stat', 0):.2f})")
print(f"  IR:        {metrics.get('ir', 0):.4f}")
print(f"  多空Sharpe: {ls_sh:.4f}")
print(f"  多空MDD:    {ls_md:.2%}")
print(f"  单调性:     {mono:.4f}")
print(f"  换手率:     {metrics.get('turnover_mean', 0):.2%}")
print(f"{'─'*60}")
print(f"  分层年化收益:")
for i, r in enumerate(metrics.get("group_returns_annualized", []), 1):
    r_str = f"{r:.2%}" if r is not None else "N/A"
    print(f"    G{i}: {r_str}")
print(f"{'═'*60}")
msg = "全部达标 ✓✓✓" if n_pass == 3 else ("两项达标 ✓✓" if n_pass == 2 else ("一项达标 ✓" if n_pass == 1 else "未达标 ✗"))
print(f"  达标准则: |IC|>0.015 | t>2 | |Sharpe|>0.5")
print(f"  {icon_big} |IC|{'>' if is_big else '<'}0.015  {icon_t} t{'>' if is_t else '<'}2  {icon_sh} |Sharpe|{'>' if is_sh else '<'}0.5  >> {msg}")
print(f"\n  总耗时: {time.time()-t0:.1f}s")
