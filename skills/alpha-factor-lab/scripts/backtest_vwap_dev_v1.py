#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: vwap_dev_v1 — VWAP偏离因子 v1
==========================================
方向: 量价/微观结构/动量代理

构造:
  1. 计算每日 VWAP = amount / volume
  2. VWAP离差 = (close - VWAP) / ATR(20) — 用ATR标准化幅度
  3. 20日窗口累积 = sum(VWAP偏离, 20d)，衡量持续的主动买卖方向
  4. 成交额OLS中性化 + 5%缩尾

逻辑:
  VWAP偏离反映了当日成交量加权的平均成交价格与收盘价的差:
  - 正偏离(close>VWAP): 尾盘买入推动价格高于全天均价→主动买入占优
  - 负偏离(close<VWAP): 尾盘卖出压低价格→主动卖出占优
  
  持续的正偏离 = day-after-day主动性买入 = 信息积累方向 → 动量延续
  持续的负偏离 = 持续性卖出压力 → 股价承压
  
  与pvr(量价相关性)的区别:
  - vol_ret_align: 只看量价方向一致性(有无趋势)
  - VWAP偏离: 量化主动买入vs卖出的方向性强度, magnitude-based
  
  与TAE的区别:
  - TAE: 纯波动率层面(高换手=参与度高)
  - VWAP偏离: 方向层面(谁在主导交易)

理论依据:
  - Brenner, Pasquariello & Subrahmanyam (2006) "On the Variability of VWAP Prices"
  - Kissell, Miao & Wang (2005) "Dynamic VWAP for Optimal Trading"
  - Chordia & Subrahmanyam (2004) Order Imbalance
"""

import json
import sys
import warnings
from pathlib import Path
import time

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
DATA_CUTOFF = "2026-04-15"
FACTOR_ID = "vwap_dev_v1"

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "csi1000_kline_raw.csv"
SCRIPTS_DIR = BASE_DIR / "skills" / "alpha-factor-lab" / "scripts"
OUTPUT_DIR = BASE_DIR / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

t0 = time.time()

# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] <= DATA_CUTOFF]
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")
volume_piv = df.pivot_table(index="date", columns="stock_code", values="volume")

ret_piv = close_piv.pct_change()
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = list(close_piv.index)
stocks = list(close_piv.columns)
print(f"   {len(dates)} 日, {len(stocks)} 股  ({time.time()-t0:.1f}s)")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造VWAP偏离因子...")
t0 = time.time()

# VWAP = amount / volume
vwap_piv = (amount_piv / volume_piv.clip(lower=1)).replace([np.inf, -np.inf], np.nan)

# VWAP偏离 = (close - VWAP) / ATR(20)
# ATR-like: average of (high-low) over 20 days as volatility scale
amp_piv = df.pivot_table(index="date", columns="stock_code", values="amplitude") / 100
atr_20 = amp_piv.rolling(20, min_periods=10).mean().clip(lower=0.001)
vwap_dev = (close_piv - vwap_piv) / atr_20

# 20日窗口累计VWAP偏离 = 持续性主动买入/卖出信号
factor_raw = vwap_dev.rolling(WINDOW, min_periods=10).mean()

print(f"   非空率: {factor_raw.notna().mean().mean():.2%}")
print(f"   均值: {factor_raw.stack().mean():.4f}, std: {factor_raw.stack().std():.4f}  ({time.time()-t0:.1f}s)")

# ────────────────── 缩尾 ──────────────────
print(f"[3] 缩尾 ({WINSORIZE_PCT*100:.0f}%)...")
t0 = time.time()
def winsorize_df(df_raw, pct=0.05):
    arr = df_raw.values.copy()
    mask = ~np.isnan(arr)
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
print(f"[4] 成交额中性化 (OLS, 向量化)...")
t0 = time.time()
fa_arr = factor_raw.values.copy()
log_amt_arr = log_amt.values
common_mask = ~np.isnan(fa_arr) & ~np.isnan(log_amt_arr)

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
print(f"[5] 回测引擎...")
t0 = time.time()

common_dates_sort = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
common_stocks_sort = sorted(factor_neutral.columns.intersection(ret_piv.columns))
fa = factor_neutral.loc[common_dates_sort, common_stocks_sort]
ra = ret_piv.loc[common_dates_sort, common_stocks_sort]

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

print(f"   正向 (高VWAP偏离→高收益): IC={pos_ic:.4f}, Sharpe={pos_sh:.4f}")
print(f"   反向 (低VWAP偏离→高收益): IC={neg_ic:.4f}, Sharpe={neg_sh:.4f}")
print(f"   ({time.time()-t0:.1f}s)")

if neg_sh > pos_sh and abs(neg_ic) > abs(pos_ic):
    direction = -1
    fa_final = -fa
    direction_desc = "反向（低VWAP偏离=高预期收益，尾盘尾卖压大→反弹）"
    print(f"   → 使用反向 ✓")
elif pos_sh > neg_sh:
    direction = 1
    fa_final = fa
    direction_desc = "正向（高VWAP偏离=高预期收益，尾盘尾主动买入→动量延续）"
    print(f"   → 使用正向 ✓")
else:
    direction = 1 if pos_ic > neg_ic else -1
    fa_final = fa if direction == 1 else -fa
    direction_desc = ("正向（高VWAP偏离=高预期收益）" if direction == 1 
                     else "反向（低VWAP偏离=高预期收益）")
    print(f"   → 使用{'正向' if direction==1 else '反向'} ✓ (IC优先)")

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

high_piv = df.pivot_table(index="date", columns="stock_code", values="high")
low_piv = df.pivot_table(index="date", columns="stock_code", values="low")
open_piv = df.pivot_table(index="date", columns="stock_code", values="open")
turnover_piv = df.pivot_table(index="date", columns="stock_code", values="turnover")

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
ret_np = ret_piv.values
n_d = len(dates)
cvar_mat = np.full((n_d, len(stocks)), np.nan)
for i in range(10, n_d):
    w = ret_np[i-10:i, :]
    s = np.sort(w, axis=0)
    bot2 = np.nanmean(s[:2, :], axis=0)
    vc = np.sum(~np.isnan(w), axis=0)
    bot2[vc < 5] = np.nan
    cvar_mat[i, :] = -bot2
cvar_df = pd.DataFrame(cvar_mat, index=dates, columns=stocks)

neg_freq = (ret_piv <= -0.03).astype(float).rolling(10, min_periods=5).mean()
turnover_level = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8))

amp_piv2 = df.pivot_table(index="date", columns="stock_code", values="amplitude") / 100
tae_raw = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8) /
                  (amp_piv2.rolling(20, min_periods=10).mean().clip(lower=0.01) + 0.01))

ma5 = close_piv.rolling(5).mean() / close_piv
ma10 = close_piv.rolling(10).mean() / close_piv
ma20 = close_piv.rolling(20).mean() / close_piv
ma40 = close_piv.rolling(40).mean() / close_piv
ma60 = close_piv.rolling(60).mean() / close_piv
ma120 = close_piv.rolling(120).mean() / close_piv
ma_stack = np.stack([ma5.values, ma10.values, ma20.values, ma40.values, ma60.values, ma120.values], axis=0)
ma_disp_vals = np.nanstd(ma_stack, axis=0)
ma_disp_factor = pd.DataFrame(ma_disp_vals, index=close_piv.index, columns=close_piv.columns)

fa_for_corr = fa_final

correlations = {}
for name, other in [
    ('amihud_illiq_v2', amihud_factor), ('shadow_pressure_v1', shadow),
    ('overnight_momentum_v1', overnight_mom), ('tail_risk_cvar_v1', cvar_df),
    ('neg_day_freq_v1', neg_freq), ('turnover_level_v1', turnover_level),
    ('tae_v1', tae_raw), ('ma_disp_v1', ma_disp_factor),
]:
    corrs = []
    sample_dates = common_dates_sort[::max(1, len(common_dates_sort)//20)]
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
    "factor_name": "VWAP偏离 v1",
    "factor_name_en": "VWAP Deviation v1",
    "category": "量价/微观结构",
    "description": f"20日(close-VWAP)/ATR均值。高偏离=持续主动买入/卖出=方向性信息流。mAGNITUDE-based量价同步信号。",
    "hypothesis": "持续正偏=尾盘/收盘主动买入推动=信息积累方向→动量延续。持续负偏=尾盘抛售→股价承压。ATR标准化消除个股波动率差异。",
    "formula": "neutralize(MA20((close-VWAP)/ATR20), log_amount_20d)",
    "direction": direction,
    "direction_desc": direction_desc,
    "stock_pool": "中证1000",
    "period": f"{common_dates_sort[0].strftime('%Y-%m-%d')} ~ {common_dates_sort[-1].strftime('%Y-%m-%d')}",
    "n_dates": len(common_dates_sort),
    "n_stocks": len(common_stocks_sort),
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
print(f"  {FACTOR_ID}: VWAP偏离因子")
print(f"  方向: {direction_desc}")
print(f"{'='*60}")
print(f"  区间:       {report['period']}")
print(f"  股票:       {len(common_stocks_sort)}")
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
    print(f"  → 因子未达标")
    if abs(ic_m) <= 0.005:
        print(f"  → 失败: IC极低, 信号几乎不区分截面")
    elif mono < 0.5:
        print(f"  → 失败: 分组单调性差, 信号非线性/不稳定")
    elif ls_sh < 0.5:
        print(f"  → 失败: 多空Sharpe不足, 无法覆盖成本")

total_time = time.time() - t0
print(f"\n  总耗时: {total_time:.0f}s")
