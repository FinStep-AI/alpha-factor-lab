#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: vol_term_v1 — 波动率期限结构因子
==========================================

方向: 波动率/风险

构造:
  1. 短期波动率: std(daily_ret, 5d)
  2. 长期波动率: std(daily_ret, 60d)
  3. 期限比: short_vol / long_vol
  4. 成交额OLS中性化
  5. 5%缩尾

逻辑:
  波动率期限结构衡量短期vs长期风险的相对水平。
  - 高比值(短期>长期): 近期波动率骤升，可能正经历信息事件/恐慌，
    但长期来看波动率均值回复→短期高波后续降温=反弹机会
  - 低比值(短期<长期): 近期平静但长期波动高，处于暴风雨前的平静
  
  不同于amp_level(绝对波动率水平)和CVaR(尾部极端事件)，
  本因子关注的是波动率的时间序列形态(升降趋势)。
  
理论:
  - Christensen & Prabhala (1998): 隐含vs实现波动率期限结构有预测力
  - Adrian & Rosenberg (2008): 短期和长期波动率成分对截面收益有不同定价
  - 波动率均值回复是金融学最稳健的经验事实之一
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
SHORT_WINDOW = 5
LONG_WINDOW = 60
FORWARD_DAYS = 5
REBALANCE_FREQ = 5
N_GROUPS = 5
COST = 0.003
WINSORIZE_PCT = 0.05
DATA_CUTOFF = "2026-03-13"
FACTOR_ID = "vol_term_v1"

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
high_piv = df.pivot_table(index="date", columns="stock_code", values="high")
low_piv = df.pivot_table(index="date", columns="stock_code", values="low")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")

ret_piv = close_piv.pct_change()
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造波动率期限结构因子 (short={SHORT_WINDOW}d / long={LONG_WINDOW}d)...")

# 已实现波动率
short_vol = ret_piv.rolling(SHORT_WINDOW, min_periods=3).std()
long_vol = ret_piv.rolling(LONG_WINDOW, min_periods=30).std()

# 期限比: log(short/long) 取对数使分布更对称
factor_raw = np.log((short_vol / long_vol.clip(lower=1e-8)).clip(lower=1e-8))

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
    print(f"   → 反向更优，翻转因子")
    fa = -fa
    ic_series = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "pearson")
    rank_ic_series = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "spearman")
    group_returns, turnovers, holdings_info = compute_group_returns(
        fa, ra, N_GROUPS, REBALANCE_FREQ, COST
    )
    metrics = compute_metrics(
        group_returns, ic_series, rank_ic_series, turnovers, N_GROUPS,
        holdings_info=holdings_info
    )
    direction = -1
    direction_desc = "反向（低期限比=短期平静=低波环境=高预期收益）"
else:
    direction = 1
    direction_desc = "正向（高期限比=短期波动骤升=均值回复反弹机会）"
print(f"   → 使用{'正向' if direction == 1 else '反向'} ✓")

# ────────────────── 参数敏感性 ──────────────────
print(f"\n[6c] 参数敏感性分析...")
for test_fwd in [5, 10, 20]:
    test_ic = compute_ic_dynamic(fa, ra, test_fwd, "pearson")
    test_gr, _, _ = compute_group_returns(fa, ra, N_GROUPS, test_fwd, COST)
    test_m = compute_metrics(test_gr, test_ic, test_ic, [], N_GROUPS)
    ic_m = test_m.get("ic_mean", 0) or 0
    ic_t = test_m.get("ic_t_stat", 0) or 0
    sh = test_m.get("long_short_sharpe", 0) or 0
    mono = test_m.get("monotonicity", 0) or 0
    print(f"   fwd={test_fwd:2d}d: IC={ic_m:.4f}(t={ic_t:.2f}), Sharpe={sh:.4f}, Mono={mono:.2f}")

# ────────────────── 相关性 ──────────────────
print(f"\n[7] 与现有因子相关性...")
turnover_piv = df.pivot_table(index="date", columns="stock_code", values="turnover")

amihud_raw = (ret_piv.abs() / (amount_piv / 1e8).clip(lower=1e-8))
amihud_factor = np.log(amihud_raw.rolling(20, min_periods=10).mean().clip(lower=1e-12))

upper_sr = (high_piv - np.maximum(close_piv, open_piv)) / (high_piv - low_piv).clip(lower=1e-8)
lower_sr = (np.minimum(close_piv, open_piv) - low_piv) / (high_piv - low_piv).clip(lower=1e-8)
shadow = (upper_sr - lower_sr).rolling(20, min_periods=10).mean()

oret = open_piv / close_piv.shift(1) - 1
iret = close_piv / open_piv - 1
overnight_mom = oret.rolling(20, min_periods=10).sum() - iret.rolling(20, min_periods=10).sum()

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
turnover_level = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8))

amplitude_piv = (high_piv - low_piv) / close_piv.shift(1).clip(lower=0.01)
tae = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8) / 
             (amplitude_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8) + 0.01))

# amp_level_v2
amp_level = np.log(amplitude_piv.rolling(60, min_periods=30).mean().clip(lower=1e-8))

correlations = {}
for name, other in [('amihud_illiq_v2', amihud_factor), ('shadow_pressure_v1', shadow),
                     ('overnight_momentum_v1', overnight_mom), ('tail_risk_cvar_v1', cvar_df),
                     ('neg_day_freq_v1', neg_freq), ('turnover_level_v1', turnover_level),
                     ('tae_v1', tae), ('amp_level_v2', amp_level)]:
    corrs = []
    for d in common_dates[::10]:
        f1 = fa.loc[d].dropna()
        f2 = other.loc[d].reindex(f1.index).dropna() if d in other.index else pd.Series(dtype=float)
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
    "factor_name": "波动率期限结构 v1",
    "factor_name_en": "Volatility Term Structure v1",
    "category": "波动率/风险",
    "description": f"log(std(ret,{SHORT_WINDOW}d) / std(ret,{LONG_WINDOW}d)), 成交额中性化。衡量短期vs长期波动率的相对水平。",
    "hypothesis": "波动率期限结构包含关于未来风险变化的信息，短期波动率骤升后均值回复效应产生反弹收益。",
    "formula": f"neutralize(log(std(ret,{SHORT_WINDOW}d)/std(ret,{LONG_WINDOW}d)), log_amount_20d)",
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
print(f"  {FACTOR_ID}: 波动率期限结构因子")
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
    print(f"  → 因子未达标，记录失败原因")
