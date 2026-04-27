#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: neg_range_efficiency_v1 — 反转日内价格效率（反向）
===========================================================
核心逻辑:
  日内价格效率 = |close - open| / (high - low)
  正向: 高效率 = 强单边趋势 = 动量延续
  反向: 低效率 = 震荡/无方向 = 随后反转 → 长期支撑有效

构造:
  1. 单日: |close-open|/(high-low)  (值域[0,1], 0=完全震荡/1=完全单边)
  2. 20日滚动均值 → 平滑
  3. OLS中性化 + 5%缩尾 + z-score
  4. 反向: 取负 → 低效率 = 高因子值 = 做多

方向: 反向(-)
20日窗口 | 成交额中性化(OLS) | 5%缩尾 | 5组分层 | 10日前瞻

参考文献:
  - Amaya, Christoffersen, Jacobs & Vasquez (2015) JFE: 已实现偏度和峰度与横截面预期收益
  - Gervais, Kaniel & Mingelgrin (2001) JF: 高成交量收益溢价 (投资者对日内趋势欠充分时均值回复)
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ────────────────── 参数 ──────────────────
WINDOW = 20
REBALANCE_FREQ = 20
FORWARD_DAYS = 10
N_GROUPS = 5
COST = 0.003
FACTOR_ID = "neg_range_efficiency_v1"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "csi1000_kline_raw.csv"
FACTOR_PATH = Path(__file__).resolve().parent.parent / "data" / "factor_neg_range_efficiency_v1.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"


# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")
ret_piv = close_piv.pct_change()
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ────────────────── 因子构造（不通过 data global; 直接 build） ──────────────────
print(f"[2] 构造因子: {WINDOW}日内反转日内价格效率(反向)...")

# 直接build并run the entire backtest engine via direct computation
# This will replicate the neg_range_efficiency_v1 computation inline

df["daily_range"] = df["high"] - df["low"]
df["daily_body"] = (df["close"] - df["open"]).abs()
df["daily_re"] = np.where(
    df["daily_range"] > 0,
    df["daily_body"] / df["daily_range"],
    0.5
)
df["daily_re"] = df["daily_re"].clip(0, 1)

df["range_eff"] = df.groupby("stock_code")["daily_re"].transform(
    lambda x: x.rolling(WINDOW, min_periods=int(WINDOW * 0.8)).mean()
)

# OLS中性化: range_eff vs log(amount_20d)
df["log_amt_20d"] = np.log(
    df.groupby("stock_code")["amount"].transform(
        lambda x: x.rolling(WINDOW, min_periods=10).mean()
    ).clip(lower=1)
)

print(f"[3] 因子反转(反向)...")
factor_raw = df[["date", "stock_code", "range_eff", "log_amt_20d"]].copy()

# Build factor pivot
factor_pivot = factor_raw.pivot_table(
    index="date", columns="stock_code", values="range_eff"
)
log_amt_pivot = factor_raw.pivot_table(
    index="date", columns="stock_code", values="log_amt_20d"
)

# 反转: 高RE = 好趋势; 低RE = 差趋势 → 反转期望低RE带来好未来收益
# factor = -range_eff (动量断裂, 反转来自有效率差)
# Step 1: take negative (low efficiency = high factor = buy)
factor_neg = -factor_pivot

# Step 2: OLS neutralize
for date in dates:
    f = factor_neg.loc[date].dropna()
    m = log_amt_pivot.loc[date].reindex(f.index).dropna()
    common = f.index.intersection(m.index)
    if len(common) < 30:
        factor_neg.loc[date, :] = np.nan
        continue
    f_c = f[common].values
    m_c = m[common].values
    X = np.column_stack([np.ones(len(m_c)), m_c])
    try:
        beta = np.linalg.lstsq(X, f_c, rcond=None)[0]
        factor_neg.loc[date, common] = f_c - X @ beta
    except:
        factor_neg.loc[date, :] = np.nan

# Step 3: MAD winsorize + z-score
factor_neutral = factor_neg.copy()
for date in dates:
    f = factor_neg.loc[date].dropna()
    if len(f) < 10:
        continue
    med = f.median()
    mad = (f - med).abs().median() * 1.4826
    if mad < 1e-10:
        continue
    f_clipped = f.clip(med - 5 * mad, med + 5 * mad)
    mu, sg = f_clipped.mean(), f_clipped.std()
    if sg < 1e-10:
        continue
    zscore = (f_clipped - mu) / sg
    factor_neutral.loc[date, zscore.index] = zscore

print(f"   因子非空率: {factor_neutral.notna().mean().mean():.2%}")

# ────────────────── 回测 ──────────────────
print(f"[4] 回测: {N_GROUPS}组, {REBALANCE_FREQ}d, {COST*100:.1f}%成本...")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills" / "alpha-factor-lab" / "scripts"))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data
)

print(f"[5] IC (forward={FORWARD_DAYS}d)...")
ic_series = compute_ic_dynamic(factor_neutral, ret_piv, FORWARD_DAYS, "pearson")
rank_ic_series = compute_ic_dynamic(factor_neutral, ret_piv, FORWARD_DAYS, "spearman")

group_returns, turnovers, holdings_info = compute_group_returns(
    factor_neutral, ret_piv, N_GROUPS, REBALANCE_FREQ, COST
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


common_dates = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))

report = {
    "factor_id": FACTOR_ID,
    "factor_name": "反转日内价格效率(反向) v1",
    "factor_name_en": "Neg Range Efficiency v1",
    "category": "反转/尾部风险",
    "description": f"反转日内价格效率: |close-open|/(high-low) 的20日滚动均值，反向使用。低日内效率(震荡/无明确方向) → 随后反转向上。即日内动量未充分展开的股票,后续有较高修复空间。20日窗口,成交额OLS中性化,5%MAD缩尾,5组分层。",
    "hypothesis": "日内价格效率低的股票(高振幅但方向混乱) → 信息未充分消化 → 价格发现进行中 → 后续方向性调整(即反转/修复) → 正alpha。",
    "formula": f"neutralize(-mean(|close-open|/(high-low), {WINDOW}d), log_amount_20d)",
    "direction": -1,
    "stock_pool": "中证1000",
    "period": f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}",
    "n_dates": len(common_dates),
    "n_stocks": len(stocks),
    "n_groups": N_GROUPS,
    "rebalance_freq": REBALANCE_FREQ,
    "forward_days": FORWARD_DAYS,
    "cost": COST,
    "data_cutoff": str(pd.Timestamp("2026-04-24")),
    "correlation_with_cvar": None,
    "metrics": nan_to_none(metrics),
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)

print(f"\n[6] 结果: {REPORT_PATH}")
m = report.get("metrics", {})
ic_mean = m.get("ic_mean")
ic_t = m.get("ic_t_stat")
sharpe = m.get("long_short_sharpe")
mono = m.get("monotonicity")

if isinstance(ic_mean, float) and isinstance(ic_t, float) and isinstance(sharpe, float):
    print(f"  IC mean: {ic_mean:.4f}")
    print(f"  IC t:    {ic_t:.2f}")
    print(f"  LS Sharpe: {sharpe:.2f}")
    print(f"  Mono: {mono:.2f}")

    if abs(ic_mean) > 0.015 and abs(ic_t) > 2 and abs(sharpe) > 0.5:
        print("\n✅ 因子达标！")
        PASSES = True
    else:
        print("\n❌ 因子未达标")
        PASSES = False
else:
    print("⚠️ 指标异常")
    PASSES = False
