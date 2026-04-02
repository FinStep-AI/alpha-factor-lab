#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: amt_volatility_v1 — 成交额波动率因子
============================================
方向: 流动性风险/信息不确定性

构造:
  1. 计算20日成交额对数收益率的标准差
  2. 成交额OLS中性化
  3. 5%缩尾
  4. 方向待验证:
     - 正向: 高成交额波动=高风险溢价补偿
     - 反向: 低成交额波动=稳定性溢价

逻辑:
  传统流动性因子(Amihud/换手率)看的是成交额"水平"或"价格冲击"。
  本因子看的是成交额的"稳定性"——
  
  高成交额波动性:
  → 交易行为不稳定，有时热有时冷
  → 信息不确定性高，做市商面临库存风险
  → 流动性时好时差，执行风险高
  → 投资者可能要求风险溢价(正向)
  → 或者意味着知情交易者的时机选择(方向不确定)
  
  低成交额波动性:
  → 交易行为稳定，流动性可预测
  → 持有成本低，执行确定性高

参考:
  - Chordia, Roll & Subrahmanyam (2001) "Market Liquidity and Trading Activity"
  - Hameed, Kang & Viswanathan (2010) "Stock Market Declines and Liquidity"
  - 陈志斌等 (2015) "成交额稳定性与个股收益" 金融研究
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
DATA_CUTOFF = "2026-03-27"
FACTOR_ID = "amt_volatility_v1"

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

ret_piv = close_piv.pct_change()
log_amt = np.log(amount_piv.rolling(WINDOW).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造成交额波动率因子...")

# 成交额对数收益率
log_amount = np.log(amount_piv.clip(lower=1))
amt_log_ret = log_amount.diff()

# 20日成交额对数收益率的标准差
factor_raw = amt_log_ret.rolling(WINDOW, min_periods=10).std()

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

# ────────────────── 确认方向 ──────────────────
print(f"[6b] 方向确认...")
ic_neg = compute_ic_dynamic(-fa, ra, FORWARD_DAYS, "pearson")
gr_neg, _, _ = compute_group_returns(-fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_neg = compute_metrics(gr_neg, ic_neg, ic_neg, [], N_GROUPS)
pos_sh = metrics.get("long_short_sharpe", 0) or 0
neg_sh = m_neg.get("long_short_sharpe", 0) or 0
print(f"   正向Sharpe={pos_sh:.4f}, 反向Sharpe={neg_sh:.4f}")

if neg_sh > pos_sh:
    print(f"   → 反向更好! 切换为反向...")
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
    direction_desc = "反向（做多低成交额波动=稳定性溢价）"
else:
    direction = 1
    direction_desc = "正向（做多高成交额波动=风险溢价）"
    print(f"   → 使用正向 ✓")

# ────────────────── 相关性 ──────────────────
print(f"[7] 与现有因子相关性...")

amihud_raw = (ret_piv.abs() / (amount_piv / 1e8).clip(lower=1e-8))
amihud_factor = np.log(amihud_raw.rolling(20, min_periods=10).mean().clip(lower=1e-12))

turnover_piv = df.pivot_table(index="date", columns="stock_code", values="turnover")
turnover_factor = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8))

# Amplitude
amp_piv = df.pivot_table(index="date", columns="stock_code", values="amplitude") / 100
high_piv = df.pivot_table(index="date", columns="stock_code", values="high")
low_piv = df.pivot_table(index="date", columns="stock_code", values="low")
amp_factor = np.log(((high_piv - low_piv) / close_piv.shift(1).clip(lower=0.01)).rolling(60, min_periods=30).mean().clip(lower=1e-8))

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

neg_freq = (ret_piv <= -0.03).astype(float).rolling(10, min_periods=5).mean()

tae_factor = np.log((turnover_piv.rolling(20, min_periods=10).mean() / (amp_piv.rolling(20, min_periods=10).mean() + 0.01)).clip(lower=1e-8))

# MA Disp
ma5 = close_piv.rolling(5).mean() / close_piv
ma10 = close_piv.rolling(10).mean() / close_piv
ma20 = close_piv.rolling(20).mean() / close_piv
ma40 = close_piv.rolling(40).mean() / close_piv
ma60 = close_piv.rolling(60).mean() / close_piv
ma120 = close_piv.rolling(120).mean() / close_piv
ma_stack = np.stack([ma5.values, ma10.values, ma20.values, ma40.values, ma60.values, ma120.values], axis=0)
ma_disp_vals = np.nanstd(ma_stack, axis=0)
ma_disp_factor = pd.DataFrame(ma_disp_vals, index=close_piv.index, columns=close_piv.columns)

correlations = {}
for name, other in [
    ('amihud_illiq_v2', amihud_factor),
    ('tail_risk_cvar_v1', cvar_df),
    ('neg_day_freq_v1', neg_freq),
    ('turnover_level_v1', turnover_factor),
    ('tae_v1', tae_factor),
    ('amp_level_v2', amp_factor),
    ('ma_disp_v1', ma_disp_factor),
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
    "factor_name": "成交额波动率 v1",
    "factor_name_en": "Amount Volatility v1",
    "category": "流动性风险/信息不确定性",
    "description": "20日成交额对数收益率的标准差，成交额OLS中性化。衡量成交额稳定性/交易行为不确定性。",
    "hypothesis": "成交额波动性高的股票交易行为不稳定，信息不确定性高，流动性时好时差，投资者要求风险溢价补偿。",
    "formula": f"neutralize(std(diff(log(amount)), {WINDOW}), log_amount_{WINDOW}d)",
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
print(f"  {FACTOR_ID}: 成交额波动率因子")
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

print(f"  相关性:")
for name, r in correlations.items():
    print(f"    vs {name}: {r:.3f}")

print(f"{'='*60}")
is_valid = abs(ic_m) > 0.015 and abs(ic_t) > 2 and abs(ls_sh) > 0.5
print(f"\n  ➤ 因子{'有效 ✓' if is_valid else '无效 ✗'}")
print(f"  评估标准: |IC|>0.015, |t|>2, |Sharpe|>0.5")
if is_valid:
    print(f"  → 准备写入factors.json并git提交")
else:
    print(f"  → 因子未达标，记录失败原因")
