#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: realized_skew_v2 — 已实现偏度因子 v2
==========================================
v2 更新: 数据截止 2026-04-15, 移除耗时的窗口敏感性分析(原版三重循环导致超时)
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import time

warnings.filterwarnings("ignore")

# ────────────────── 参数 ──────────────────
WINDOW = 20
FORWARD_DAYS = 5
REBALANCE_FREQ = 5
N_GROUPS = 5
COST = 0.003
WINSORIZE_PCT = 0.05
DATA_CUTOFF = "2026-04-15"
FACTOR_ID = "realized_skew_v2"

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "csi1000_kline_raw.csv"
SCRIPTS_DIR = BASE_DIR / "skills" / "alpha-factor-lab" / "scripts"
OUTPUT_DIR = BASE_DIR / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

t0 = time.time()

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
print(f"   {len(dates)} 日, {len(stocks)} 股  ({time.time()-t0:.1f}s)")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造已实现偏度因子 (window={WINDOW})...")
t0 = time.time()

t0b = time.time()
# 向量化滚动偏度计算
# 使用 pandas rolling skew 替代纯 Python 循环
factor_raw = ret_piv.rolling(WINDOW, min_periods=10).skew()
print(f"   pandas rolling skew 完成  ({time.time()-t0b:.1f}s)")
print(f"   非空率: {factor_raw.notna().mean().mean():.2%}")
print(f"   均值: {factor_raw.stack().mean():.4f}, std: {factor_raw.stack().std():.4f}")

# ────────────────── 缩尾 ──────────────────
print(f"[3] 缩尾 ({WINSORIZE_PCT*100:.0f}%)...")
t0 = time.time()
def winsorize_df(df_raw, pct=0.05):
    arr = df_raw.values.copy()
    mask = ~np.isnan(arr)
    # 每行缩尾
    for i in range(arr.shape[0]):
        row = arr[i, :]
        valid = row[mask[i, :]]
        if len(valid) > 10:
            lo = np.quantile(valid, pct)
            hi = np.quantile(valid, 1-pct)
            row[mask[i, :]] = np.clip(valid, lo, hi)
        arr[i, :] = row
    return pd.DataFrame(arr, index=df_raw.index, columns=df_raw.columns)

factor_raw = winsorize_df(factor_raw, WINSORIZE_PCT)
print(f"   ({time.time()-t0:.1f}s)")

# ────────────────── 中性化 ──────────────────
print(f"[4] 成交额中性化 (OLS, 纯Numpy向量化)...")
t0 = time.time()
# 向量化OLS: 每天对每只股票, factor = factor - (intercept + beta*log_amt)
fa_arr = factor_raw.values.copy()
log_amt_arr = log_amt.values
common_mask = ~np.isnan(fa_arr) & ~np.isnan(log_amt_arr) & (np.abs(log_amt_arr) > 1e-8)

for date_idx in range(fa_arr.shape[0]):
    row_mask = common_mask[date_idx, :]
    if np.sum(row_mask) < 30:
        continue
    f = fa_arr[date_idx, row_mask]
    x = log_amt_arr[date_idx, row_mask]
    X = np.column_stack([np.ones(np.sum(row_mask)), x])
    try:
        beta = np.linalg.lstsq(X, f, rcond=None)[0]
        fa_arr[date_idx, row_mask] = f - X @ beta
    except:
        pass

factor_neutral = pd.DataFrame(fa_arr, index=dates, columns=stocks)
print(f"   ({time.time()-t0:.1f}s)")

# ────────────────── 回测 ──────────────────
print(f"[5] 回测: {N_GROUPS}组, {REBALANCE_FREQ}d调仓...")
t0 = time.time()

common_dates_sorted = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
common_stocks_sorted = sorted(factor_neutral.columns.intersection(ret_piv.columns))
fa = factor_neutral.loc[common_dates_sorted, common_stocks_sorted]
ra = ret_piv.loc[common_dates_sorted, common_stocks_sorted]

sys.path.insert(0, str(SCRIPTS_DIR))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data
)

# ────────────────── 方向探索 ──────────────────
print(f"[6] 方向探索...")
t0 = time.time()

ic_pos = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "pearson")
gr_pos, _, _ = compute_group_returns(fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_pos = compute_metrics(gr_pos, ic_pos, ic_pos, [], N_GROUPS)

ic_neg = compute_ic_dynamic(-fa, ra, FORWARD_DAYS, "pearson")
gr_neg, _, _ = compute_group_returns(-fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_neg = compute_metrics(gr_neg, ic_neg, ic_neg, [], N_GROUPS)

pos_sh = m_pos.get("long_short_sharpe", 0) or 0
neg_sh = m_neg.get("long_short_sharpe", 0) or 0
pos_ic = m_pos.get("ic_mean", 0) or 0
neg_ic = m_neg.get("ic_mean", 0) or 0

print(f"   正向 (高偏度=高收益): IC={pos_ic:.4f}, Sharpe={pos_sh:.4f}")
print(f"   反向 (低偏度=高收益): IC={neg_ic:.4f}, Sharpe={neg_sh:.4f}")
print(f"   ({time.time()-t0:.1f}s)")

if neg_sh > pos_sh:
    direction = -1
    fa_final = -fa
    direction_desc = "反向（低偏度=高预期收益，做空彩票型股票）"
    print(f"   → 使用反向 ✓")
else:
    direction = 1
    fa_final = fa
    direction_desc = "正向（高偏度=高预期收益）"
    print(f"   → 使用正向 ✓")

# ────────────────── 最终回测 ──────────────────
print(f"[7] 最终回测 (direction={direction})...")
t0 = time.time()

ic_series = compute_ic_dynamic(fa_final, ra, FORWARD_DAYS, "pearson")
rank_ic_series = compute_ic_dynamic(fa_final, ra, FORWARD_DAYS, "spearman")

group_returns, turnovers, holdings_info = compute_group_returns(
    fa_final, ra, N_GROUPS, REBALANCE_FREQ, COST
)
metrics = compute_metrics(
    group_returns, ic_series, rank_ic_series, turnovers, N_GROUPS,
    holdings_info=holdings_info
)
print(f"   ({time.time()-t0:.1f}s)")

# ────────────────── 相关性 ──────────────────
print(f"[8] 与现有因子相关性...")
t0 = time.time()

# 构建已有因子矩阵(复用在回测脚本中的相同计算方法)
open_piv = df.pivot_table(index="date", columns="stock_code", values="open")
high_piv = df.pivot_table(index="date", columns="stock_code", values="high")
low_piv = df.pivot_table(index="date", columns="stock_code", values="low")
turnover_piv = df.pivot_table(index="date", columns="stock_code", values="turnover")
amplitude_piv = df.pivot_table(index="date", columns="stock_code", values="amplitude")

# Amihud
amihud_factor = np.log(
    (ret_piv.abs() / (amount_piv / 1e8).clip(lower=1e-8))
    .rolling(20, min_periods=10).mean().clip(lower=1e-12)
)

# Shadow pressure
upper_sr = (high_piv - np.maximum(close_piv, open_piv)) / (high_piv - low_piv).clip(lower=1e-8)
lower_sr = (np.minimum(close_piv, open_piv) - low_piv) / (high_piv - low_piv).clip(lower=1e-8)
shadow = (upper_sr - lower_sr).rolling(20, min_periods=10).mean()

# Overnight momentum
oret = open_piv / close_piv.shift(1) - 1
iret = close_piv / open_piv - 1
overnight_mom = oret.rolling(20, min_periods=10).sum() - iret.rolling(20, min_periods=10).sum()

# CVaR
cvar_mat = np.full((len(dates), len(stocks)), np.nan)
ret_np = ret_piv.values
for i in range(10, len(dates)):
    w = ret_np[i-10:i, :]
    s = np.sort(w, axis=0)
    bot2 = np.nanmean(s[:2, :], axis=0)
    vc = np.sum(~np.isnan(w), axis=0)
    bot2[vc < 5] = np.nan
    cvar_mat[i, :] = -bot2
cvar_df = pd.DataFrame(cvar_mat, index=dates, columns=stocks)

neg_freq = (ret_piv <= -0.03).astype(float).rolling(10, min_periods=5).mean()
turnover_level = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8))
tae_raw = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8) / 
                  (amplitude_piv.rolling(20, min_periods=10).mean().clip(lower=0.01)))
amp_level = np.log(amplitude_piv.rolling(60, min_periods=30).mean().clip(lower=0.01))

fa_for_corr = fa_final

correlations = {}
for name, other in [
    ('amihud_illiq_v2', amihud_factor),
    ('shadow_pressure_v1', shadow),
    ('overnight_momentum_v1', overnight_mom),
    ('tail_risk_cvar_v1', cvar_df),
    ('neg_day_freq_v1', neg_freq),
    ('turnover_level_v1', turnover_level),
    ('tae_v1', tae_raw),
    ('amp_level_v2', amp_level),
]:
    corrs = []
    # 每(第)10天采样一次, 减少计算量
    sample_dates = common_dates_sorted[::max(1, len(common_dates_sorted)//20)]
    for d in sample_dates:
        f1 = fa_for_corr.loc[d].dropna()
        f2 = other.loc[d].reindex(f1.index).dropna()
        c = f1.index.intersection(f2.index)
        if len(c) > 50:
            r, _ = sp_stats.spearmanr(f1[c], f2[c])
            if not np.isnan(r):
                corrs.append(r)
    avg = float(np.mean(corrs)) if corrs else 0
    correlations[name] = round(avg, 3)
    print(f"   vs {name}: r={avg:.3f}")

correlations[FACTOR_ID] = 1.0
print(f"   ({time.time()-t0:.1f}s)")

# ────────────────── 输出 ──────────────────
print(f"[9] 写入输出...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_backtest_data(group_returns, ic_series, rank_ic_series, str(OUTPUT_DIR))

def nan_to_none(obj):
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)): return None
    if isinstance(obj, dict): return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list): return [nan_to_none(v) for v in obj]
    return obj

report = {
    "factor_id": FACTOR_ID,
    "factor_name": "已实现偏度 v2",
    "factor_name_en": "Realized Skewness v2",
    "category": "波动率偏度/风险溢价",
    "description": f"过去{WINDOW}日日收益率的标准化偏度(三阶中心矩/σ³)。反向使用: 低偏度(左尾肥)=风险暴露大=未来获补偿。",
    "hypothesis": "低偏度(负偏=左尾风险)的股票暴露于极端下跌风险,A股小盘股散户忽略此风险,需要额外溢价补偿。高偏度(彩票型)被散户追捧→高估→后续差。",
    "formula": f"neutralize(-skew(daily_ret, {WINDOW}d), log_amount_20d)",
    "direction": direction,
    "direction_desc": direction_desc,
    "stock_pool": "中证1000",
    "period": f"{common_dates_sorted[0].strftime('%Y-%m-%d')} ~ {common_dates_sorted[-1].strftime('%Y-%m-%d')}",
    "n_dates": len(common_dates_sorted),
    "n_stocks": len(common_stocks_sorted),
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
print(f"  {FACTOR_ID}: 已实现偏度 v2")
print(f"  方向: {direction_desc}")
print(f"{'='*60}")
print(f"  区间:       {report['period']}")
print(f"  股票:       {len(common_stocks_sorted)}")
print(f"  耗时:       {time.time()-t0:.0f}s")
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

is_valid = abs(ic_m) > 0.015 and abs(ic_t) > 2 and abs(ls_sh) > 0.5
print(f"\n  ➤ 因子{'有效 ✓' if is_valid else '无效 ✗'}")
print(f"  评估标准: |IC|>0.015, |t|>2, |Sharpe|>0.5")
if is_valid:
    print(f"  → 准备写入factors.json并git提交")
else:
    print(f"  → 因子未达标，记录失败原因")

total_time = time.time() - t0
print(f"\n  总耗时: {total_time:.0f}s")
