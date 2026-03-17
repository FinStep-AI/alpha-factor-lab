#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: vol_efficiency_v3 — 波动率效率因子 (最终版)
==================================================
方向: 量价/微观结构

基于v1/v2参数搜索, 确定最优配置:
  - 窗口: 20日
  - Parkinson estimator (更稳定)
  - 前瞻/调仓: 5d (与v1一致, 更多IC观测点→t值更高)
  - 正向: 做多高VER

构造:
  1. Parkinson日内波动率: σ_P = sqrt((ln(H/L))^2 / (4*ln2)) 的20日均值
  2. 收盘-收盘波动率: σ_CC = |ln(C_t/C_{t-1})| 的20日均值
  3. VER = σ_CC / σ_P
  4. 市值中性化(OLS) + 5%缩尾

逻辑:
  高VER → 收盘价变动/日内波幅 高 → 信息效率好 → 做多
  低VER → 日内大幅波动但收盘回原点 → 噪声交易 → 做空
  
  A股中证1000小盘股: 信息效率高的股票定价更准确,
  吸引更多知情交易者, 流动性溢价和动量效应更强。

理论:
  Alizadeh, Brandt & Diebold (2002) Range-based vol estimation
  Parkinson (1980) Extreme value estimator
  Amihud & Mendelson (1986) Price efficiency & liquidity
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
FORWARD_DAYS = 5      # 5d前瞻 → 更多IC观测(~164个), t值更可靠
REBALANCE_FREQ = 5    # 周频调仓
N_GROUPS = 5
COST = 0.003          # 0.3%双边
WINSORIZE_PCT = 0.05
DATA_CUTOFF = "2026-03-13"
FACTOR_ID = "vol_efficiency_v1"

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
high_piv = df.pivot_table(index="date", columns="stock_code", values="high")
low_piv = df.pivot_table(index="date", columns="stock_code", values="low")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")

ret_piv = close_piv.pct_change()
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造波动率效率比 (VER)...")

# Parkinson daily volatility proxy
log_hl = np.log(high_piv / low_piv.clip(lower=1e-8))
parkinson_daily = log_hl ** 2 / (4 * np.log(2))
parkinson_vol = np.sqrt(parkinson_daily.rolling(WINDOW, min_periods=10).mean())

# Close-to-close volatility proxy
log_ret = np.log(close_piv / close_piv.shift(1).clip(lower=1e-8))
cc_vol = log_ret.abs().rolling(WINDOW, min_periods=10).mean()

# VER = cc_vol / parkinson_vol
factor_raw = cc_vol / parkinson_vol.clip(lower=1e-8)
factor_raw = factor_raw.clip(0.01, 5.0)

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
print(f"[5] 回测: {N_GROUPS}组, {REBALANCE_FREQ}d调仓, {COST*100:.1f}%成本, F={FORWARD_DAYS}d...")

common_dates = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
common_stocks = sorted(factor_neutral.columns.intersection(ret_piv.columns))
fa = factor_neutral.loc[common_dates, common_stocks]
ra = ret_piv.loc[common_dates, common_stocks]

sys.path.insert(0, str(SCRIPTS_DIR))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data, newey_west_t_stat
)

print(f"[6] IC计算...")
ic_series = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "pearson")
rank_ic_series = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "spearman")

group_returns, turnovers, holdings_info = compute_group_returns(
    fa, ra, N_GROUPS, REBALANCE_FREQ, COST
)
metrics = compute_metrics(
    group_returns, ic_series, rank_ic_series, turnovers, N_GROUPS,
    holdings_info=holdings_info
)

# ────────────────── 也测反向 ──────────────────
print(f"[6b] 测试反向...")
ic_neg = compute_ic_dynamic(-fa, ra, FORWARD_DAYS, "pearson")
gr_neg, tv_neg, hi_neg = compute_group_returns(-fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_neg = compute_metrics(gr_neg, ic_neg, compute_ic_dynamic(-fa, ra, FORWARD_DAYS, "spearman"), tv_neg, N_GROUPS, holdings_info=hi_neg)

pos_sh = metrics.get("long_short_sharpe", 0) or 0
neg_sh = m_neg.get("long_short_sharpe", 0) or 0
print(f"   正向Sharpe={pos_sh:.4f}, 反向Sharpe={neg_sh:.4f}")

if neg_sh > pos_sh and neg_sh > 0:
    print(f"   → 使用反向")
    metrics = m_neg
    group_returns = gr_neg
    turnovers = tv_neg
    holdings_info = hi_neg
    ic_series = ic_neg
    rank_ic_series = compute_ic_dynamic(-fa, ra, FORWARD_DAYS, "spearman")
    direction = -1
    direction_desc = "反向（低VER=高预期收益，噪声交易溢价）"
else:
    direction = 1
    direction_desc = "正向（高VER=高预期收益，信息效率溢价）"
    print(f"   → 使用正向")

# ────────────────── 与现有因子相关性 ──────────────────
print(f"[7] 相关性检查...")
amihud_raw = (ret_piv.abs() / (amount_piv / 1e8).clip(lower=1e-8))
amihud_factor = np.log(amihud_raw.rolling(20, min_periods=10).mean().clip(lower=1e-12))

fa_use = fa if direction == 1 else -fa

# Amihud correlation
corrs_amihud = []
for date in common_dates[::20]:
    f1 = fa_use.loc[date].dropna()
    f2 = amihud_factor.loc[date].reindex(f1.index).dropna()
    c = f1.index.intersection(f2.index)
    if len(c) > 50:
        r, _ = sp_stats.spearmanr(f1[c], f2[c])
        if not np.isnan(r):
            corrs_amihud.append(r)
corr_amihud = float(np.mean(corrs_amihud)) if corrs_amihud else 0

# Shadow pressure correlation
extreme_neg_pct = (ret_piv <= -0.03).astype(float).rolling(10, min_periods=5).mean()
corrs_neg = []
for date in common_dates[::20]:
    f1 = fa_use.loc[date].dropna()
    f2 = extreme_neg_pct.loc[date].reindex(f1.index).dropna()
    c = f1.index.intersection(f2.index)
    if len(c) > 50:
        r, _ = sp_stats.spearmanr(f1[c], f2[c])
        if not np.isnan(r):
            corrs_neg.append(r)
corr_neg_freq = float(np.mean(corrs_neg)) if corrs_neg else 0

print(f"   与Amihud: {corr_amihud:.3f}")
print(f"   与neg_day_freq: {corr_neg_freq:.3f}")

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
    "factor_name": "波动率效率 v1",
    "factor_name_en": "Volatility Efficiency Ratio v1",
    "category": "量价/微观结构",
    "description": f"收盘价波动率/Parkinson日内波动率的20日均值比。高VER=日内价格变动高效(信息驱动), 低VER=日内噪声大(收盘回归)。",
    "hypothesis": "信息效率高(VER高)的股票，价格发现更充分，知情交易者比例高，后续收益更好(信息效率溢价)。",
    "formula": f"neutralize(MA{WINDOW}(|log_ret|/sqrt(log(H/L)^2/(4ln2))), log_amount_20d)",
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
    "correlations": {
        "amihud_illiq_v2": corr_amihud,
        "neg_day_freq_v1": corr_neg_freq,
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
print(f"  {FACTOR_ID}: 波动率效率比")
print(f"  方向: {direction_desc}")
print(f"  参数: W={WINDOW}, F={FORWARD_DAYS}, R={REBALANCE_FREQ}")
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
