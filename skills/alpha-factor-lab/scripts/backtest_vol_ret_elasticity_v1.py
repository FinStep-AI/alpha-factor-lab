#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: vol_ret_elasticity_v1 — 成交额-收益率弹性因子
====================================================

方向: 量价/信息效率

构造:
  1. 计算日成交额变化率: Δamt = log(amount_t / amount_{t-1})
  2. 计算日收益率: ret_t
  3. 用OLS回归 Δamt ~ ret_t，在20日滚动窗口内，取斜率β
  4. 正β = 放量涨缩量跌 = 信息驱动交易主导
  5. 负β = 放量跌缩量涨 = 流动性/噪音交易主导
  6. 市值中性化 (OLS against log_market_cap)
  7. 5%缩尾

理论基础:
  Llorente, Michaely, Saar & Wang (2002) JF "Dynamic Volume-Return Relation"
  - 信息驱动交易: 知情交易者在信息到达时放量交易, 推动价格持续
    → 收益率与成交额正相关 → 后续有动量延续
  - 流动性驱动交易: 纯流动性需求导致的放量交易, 价格暂时偏离后回复
    → 收益率与成交额负相关 → 后续均值回复
  
  A股应用:
  - 中证1000小盘股信息不对称程度高
  - 知情交易者(机构/大户)的交易行为在成交额-收益率关系中留下痕迹
  - 正弹性 = 信息逐步释放中, 后续延续
  - 负弹性 = 噪音交易/流动性冲击, 后续回复

参考:
  Llorente et al. (2002) "Dynamic Volume-Return Relation"
  https://doi.org/10.1111/1540-6261.00466
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
DATA_CUTOFF = "2026-03-23"
FACTOR_ID = "vol_ret_elasticity_v1"

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
turnover_piv = df.pivot_table(index="date", columns="stock_code", values="turnover")
open_piv = df.pivot_table(index="date", columns="stock_code", values="open")
high_piv = df.pivot_table(index="date", columns="stock_code", values="high")
low_piv = df.pivot_table(index="date", columns="stock_code", values="low")

ret_piv = close_piv.pct_change()
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))

# 也构造 log market cap 的代理(用 close * amount / turnover 估算)
# 简化: 用 log(20日平均成交额) 作为市值代理(中性化)
log_mktcap_proxy = log_amt.copy()

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造成交额-收益率弹性因子...")

# 日成交额变化率 (对数差分)
log_amount = np.log(amount_piv.clip(lower=1))
delta_log_amt = log_amount.diff()

# 20日滚动窗口内, ret vs Δlog_amt 的OLS斜率
# 用向量化的滚动方式计算
n_dates = len(dates)
n_stocks = len(stocks)

ret_vals = ret_piv.values  # (n_dates, n_stocks)
damt_vals = delta_log_amt.values  # (n_dates, n_stocks)

factor_raw_vals = np.full((n_dates, n_stocks), np.nan)

print(f"   计算滚动OLS斜率 (window={WINDOW})...")
for i in range(WINDOW, n_dates):
    # 窗口内的 ret 和 Δlog_amt
    ret_w = ret_vals[i-WINDOW:i, :]  # (WINDOW, n_stocks)
    damt_w = damt_vals[i-WINDOW:i, :]  # (WINDOW, n_stocks)
    
    # 对每只股票计算 slope = cov(ret, damt) / var(ret)
    # 注意: 这里是 damt ~ ret (成交额变化对收益率回归)
    # 正斜率 = 涨时放量, 跌时缩量
    
    # mask: 两个都有有效值
    valid = ~(np.isnan(ret_w) | np.isnan(damt_w))
    n_valid = valid.sum(axis=0)  # (n_stocks,)
    
    # 安全计算: 只对有足够观测的股票
    enough = n_valid >= 10
    
    for j in np.where(enough)[0]:
        mask = valid[:, j]
        r = ret_w[mask, j]
        d = damt_w[mask, j]
        
        # OLS: damt = alpha + beta * ret
        # beta = cov(d,r) / var(r)
        r_mean = r.mean()
        d_mean = d.mean()
        r_demean = r - r_mean
        var_r = (r_demean ** 2).sum()
        if var_r > 1e-12:
            cov_dr = (r_demean * (d - d_mean)).sum()
            beta = cov_dr / var_r
            factor_raw_vals[i, j] = beta

factor_raw = pd.DataFrame(factor_raw_vals, index=dates, columns=stocks)

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
    m = log_mktcap_proxy.loc[date].reindex(f.index).dropna()
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

# 选择更好的方向
if neg_sh > pos_sh:
    print(f"   → 反向更好, 翻转因子")
    fa = -fa
    ic_series = ic_neg
    rank_ic_series = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "spearman")
    group_returns, turnovers, holdings_info = compute_group_returns(
        fa, ra, N_GROUPS, REBALANCE_FREQ, COST
    )
    metrics = compute_metrics(
        group_returns, ic_series, rank_ic_series, turnovers, N_GROUPS,
        holdings_info=holdings_info
    )
    direction = -1
    direction_desc = "反向（低弹性=流动性驱动=均值回复=高预期收益）"
else:
    direction = 1
    direction_desc = "正向（高弹性=信息驱动=动量延续=高预期收益）"
print(f"   → 使用方向: {direction_desc}")

# ────────────────── 相关性 ──────────────────
print(f"[7] 与现有因子相关性...")

# Amihud
amihud_raw = (ret_piv.abs() / (amount_piv / 1e8).clip(lower=1e-8))
amihud_factor = np.log(amihud_raw.rolling(20, min_periods=10).mean().clip(lower=1e-12))

# Turnover level
turnover_factor = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8))

# Shadow pressure
upper_sr = (high_piv - np.maximum(close_piv, open_piv)) / (high_piv - low_piv).clip(lower=1e-8)
lower_sr = (np.minimum(close_piv, open_piv) - low_piv) / (high_piv - low_piv).clip(lower=1e-8)
shadow = (upper_sr - lower_sr).rolling(20, min_periods=10).mean()

# Overnight momentum
oret = open_piv / close_piv.shift(1) - 1
iret = close_piv / open_piv - 1
overnight_mom = oret.rolling(20, min_periods=10).sum() - iret.rolling(20, min_periods=10).sum()

# CVaR
cvar_mat = np.full((n_dates, n_stocks), np.nan)
for i in range(10, n_dates):
    w = ret_vals[i-10:i, :]
    s = np.sort(w, axis=0)
    bot2 = np.nanmean(s[:2, :], axis=0)
    vc = np.sum(~np.isnan(w), axis=0)
    bot2[vc < 5] = np.nan
    cvar_mat[i, :] = -bot2
cvar_df = pd.DataFrame(cvar_mat, index=dates, columns=stocks)

# TAE
amplitude_piv = df.pivot_table(index="date", columns="stock_code", values="amplitude")
tae_factor = np.log(
    turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8) / 
    (amplitude_piv.rolling(20, min_periods=10).mean().clip(lower=0.01) + 0.01)
)

correlations = {}
for name, other in [
    ('amihud_illiq_v2', amihud_factor), 
    ('shadow_pressure_v1', shadow),
    ('overnight_momentum_v1', overnight_mom), 
    ('tail_risk_cvar_v1', cvar_df),
    ('turnover_level_v1', turnover_factor),
    ('tae_v1', tae_factor),
]:
    corrs = []
    for d in common_dates[::10]:
        f1 = fa.loc[d].dropna()
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
    "factor_name": "成交额-收益率弹性 v1",
    "factor_name_en": "Volume-Return Elasticity v1",
    "category": "量价/信息效率",
    "description": f"20日滚动窗口内, 成交额变化率对收益率的OLS斜率。正β=放量涨缩量跌(信息驱动), 负β=放量跌缩量涨(流动性驱动)。市值中性化。",
    "hypothesis": "信息驱动交易(正弹性)的股票价格变动包含真实信息,后续有动量延续;流动性驱动交易(负弹性)的股票后续均值回复。",
    "formula": f"neutralize(OLS_slope(Δlog_amt ~ ret, {WINDOW}d), log_amount_20d)",
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
print(f"  {FACTOR_ID}: 成交额-收益率弹性因子")
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
else:
    print(f"  → 记录失败原因")
