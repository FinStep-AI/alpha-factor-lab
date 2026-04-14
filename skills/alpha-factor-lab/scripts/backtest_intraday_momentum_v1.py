#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: intraday_momentum_v1 — 日内动量一致性因子
==========================================
方向: 量价/微观结构

构造:
  1. 计算每日intraday_return = (close - open) / open
  2. 20日窗口内，intraday_return方向一致性 = sign(sum(intraday_ret)) × abs(mean(intraday_ret))
  3. 组合信号: 20日累计intraday_return方向 + 5日方向加速
  4. 成交额中性化(OLS) + 5%缩尾 + Z-score

逻辑:
  日内趋势(open→close)的方向反映知情交易者在交易时段的方向判断。
  连续多日日内上涨(open→close向上)暗示持续的信息流或知情买入;
  日内动量衰减则暗示信息已被消化或散户噪音主导。

  与overnight_momentum_v1的区别:
  - overnight_momentum = 隔夜vs日内收益差(集合竞价+隔夜信息效应)
  - intraday_momentum = 日内趋势一致性(盘中知情交易持续性)
  - 两者角度不同: 一个看跨天信息释放节奏, 一个看日内趋势强度

理论依据:
  - Chordia & Subrahmanyam (2004): 日内动量与订单不平衡
  - Heston, Sadka & Ronnie (2010): 日内动量季节性
  - A股特色: T+1 + 集合竞价 + 涨跌停 → 日内动量效应可能更强
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
WINDOW = 20              # 主窗口
ACC_WINDOW = 5           # 加速窗口
FORWARD_DAYS = 5         # 前瞻天数
REBALANCE_FREQ = 5       # 调仓频率
N_GROUPS = 5
COST = 0.003
WINSORIZE_PCT = 0.05
DATA_CUTOFF = "2026-03-27"
FACTOR_ID = "intraday_momentum_v1"

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
open_piv = df.pivot_table(index="date", columns="stock_code", values="open")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")
turnover_piv = df.pivot_table(index="date", columns="stock_code", values="turnover")

ret_piv = close_piv.pct_change()
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))
dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造日内动量因子...")

# 日内收益率 = (close - open) / open
intraday_ret = (close_piv - open_piv) / open_piv.clip(lower=0.01)

# 信号1: 20日累计intraday方向
intraday_sum_20d = intraday_ret.rolling(WINDOW, min_periods=15).sum()

# 信号2: 5日方向加速 (近期vs远期)
intraday_sum_5d = intraday_ret.rolling(ACC_WINDOW, min_periods=3).sum()
intraday_sum_15d_prev = intraday_ret.rolling(WINDOW, min_periods=15).sum() - intraday_sum_5d
acceleration = intraday_sum_5d - (intraday_sum_15d_prev / max(WINDOW - ACC_WINDOW, 1))

# 复合因子: 20日方向 + 0.5 × 加速 (加速给予近期更高权重)
factor_raw = intraday_sum_20d + 0.5 * acceleration

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
print(f"[5] 回测: {N_GROUPS}组, {REBALANCE_FREQ}d调仓, {COST*100:.1f}%成本...")

common_dates = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
common_stocks = sorted(factor_neutral.columns.intersection(ret_piv.columns))
fa = factor_neutral.loc[common_dates, common_stocks]
ra = ret_piv.loc[common_dates, common_stocks]

sys.path.insert(0, str(SCRIPTS_DIR))
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

# ────────────────── 方向确认 ──────────────────
print(f"[6b] 方向确认...")
ic_neg = compute_ic_dynamic(-fa, ra, FORWARD_DAYS, "pearson")
gr_neg, _, _ = compute_group_returns(-fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_neg = compute_metrics(gr_neg, ic_neg, ic_neg, [], N_GROUPS)
pos_sh = metrics.get("long_short_sharpe", 0) or 0
neg_sh = m_neg.get("long_short_sharpe", 0) or 0
print(f"   正向Sharpe={pos_sh:.4f}, 反向Sharpe={neg_sh:.4f}")

if neg_sh > pos_sh:
    direction = -1
    direction_desc = "反向（低日内动量=高预期收益，日内动量衰减效应）"
    group_returns = gr_neg
    metrics = m_neg
    print(f"   → 使用反向 ✓")
else:
    direction = 1
    direction_desc = "正向（高日内动量=高预期收益，日内趋势延续效应）"
    print(f"   → 使用正向 ✓")

# ────────────────── 相关性 ──────────────────
print(f"[7] 与现有因子相关性...")
open_piv = df.pivot_table(index="date", columns="stock_code", values="open")
high_piv = df.pivot_table(index="date", columns="stock_code", values="high")
low_piv = df.pivot_table(index="date", columns="stock_code", values="low")

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

# Turnover level
turnover_level = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8))

# amp_level
amplitude_piv = df.pivot_table(index="date", columns="stock_code", values="amplitude")
amp_level = np.log(amplitude_piv.rolling(60, min_periods=30).mean().clip(lower=0.01))

# 日内动量因子(用于相关性)
fa_for_corr = fa if direction == 1 else -fa

correlations = {}
for name, other in [
    ('amihud_illiq_v2', amihud_factor),
    ('shadow_pressure_v1', shadow),
    ('overnight_momentum_v1', overnight_mom),
    ('tail_risk_cvar_v1', cvar_df),
    ('turnover_level_v1', turnover_level),
    ('amp_level_v2', amp_level),
]:
    corrs = []
    for d in common_dates[::10]:
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
    "factor_name": "日内动量一致性 v1",
    "factor_name_en": "Intraday Momentum Consistency v1",
    "category": "量价/微观结构",
    "description": f"20日累计日内收益率(close-open)/open + 5日方向加速。正向:高日内动量=持续知情买入。反向:低日内动量(或负值)=日内持续抛售。",
    "hypothesis": "日内趋势方向反映知情交易者在交易时段的方向判断。连续多日日内上涨暗示持续信息流入或知情买入。与隔夜动量(overnight=跨天信息释放节奏)角度不同——本因子测盘中趋势持续性。",
    "formula": f"neutralize(sum(close/open-1, 20d) + 0.5×(sum(5d) - sum(15d)/3), log_amount_20d)",
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
print(f"  {FACTOR_ID}: 日内动量一致性因子")
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

# Return exit code for shell
sys.exit(0 if is_valid else 1)
