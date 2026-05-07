#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: vol_skew_v1 — 波动率偏度因子
==========================================
方向: 波动率偏度（收益分布尾部不对称）

构造:
  1. 过去20日日收益率的 Fisher-Pearson 偏度
  2. 成交额OLS中性化 + 5%截尾Winsorize

逻辑 (反向):
  正偏（右尾肥）= "彩票特征"，散户高估 → 后续收益差
  负偏（左尾肥）= 频繁小涨偶尔大跌，风险溢价补偿 → 后续收益好
  
A股中证1000散户比例高，彩票偏好效应更强
→ 高偏度（正偏）被高估 → 后续收益差
→ 低偏度（负偏）后续收益好

理论:
  - Boyer, Mitton & Vorkink (2010) "Expected Idiosyncratic Skewness" RFS
  - Bali, Cakici & Whitelaw (2011) "Maxing Out" JFE
  - Harvey & Siddique (2000) Conditional Skewness → JFE
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
DATA_CUTOFF = "2026-05-01"
FACTOR_ID = "vol_skew_v1"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "csi1000_kline_raw.csv"
SCRIPTS_DIR = PROJECT_ROOT / "skills" / "alpha-factor-lab" / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "output" / FACTOR_ID
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

ret_piv = close_piv.pct_change()
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股, 截至 {max(dates).strftime('%Y-%m-%d')}")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 计算 {WINDOW}日收益率偏度...")

factor_raw = ret_piv.rolling(WINDOW, min_periods=15).skew()

print(f"   非空率: {factor_raw.notna().mean().mean():.2%}")
print(f"   均值: {factor_raw.stack().mean():.4f}, std: {factor_raw.stack().std():.4f}")

# ────────────────── 缩尾 ──────────────────
print(f"[3] 5%截尾Winsorize...")
for date in dates:
    row = factor_raw.loc[date].dropna()
    if len(row) < 10:
        continue
    lo = row.quantile(0.05)
    hi = row.quantile(0.95)
    factor_raw.loc[date] = factor_raw.loc[date].clip(lo, hi)

# ────────────────── 中性化 ──────────────────
print(f"[4] 成交额OLS中性化...")
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
    except Exception:
        pass

print(f"   中性化后均值: {factor_neutral.stack().mean():.5f}, std: {factor_neutral.stack().std():.5f}")

# ────────────────── 回测 ──────────────────
print(f"[5] 回测 ({N_GROUPS}组, {REBALANCE_FREQ}d调仓, {COST*100:.1f}%成本)...")

sys.path.insert(0, str(SCRIPTS_DIR))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data
)

common_dates = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
common_stocks = sorted(factor_neutral.columns.intersection(ret_piv.columns))
fa = factor_neutral.loc[common_dates, common_stocks]
ra = ret_piv.loc[common_dates, common_stocks]

# ────────────────── 方向确认 ──────────────────
print(f"[6] 方向确认...")
ic_pos = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "pearson")
rank_ic_pos = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "spearman")
gr_pos, tv_pos, hi_pos = compute_group_returns(fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_pos = compute_metrics(gr_pos, ic_pos, rank_ic_pos, tv_pos, N_GROUPS, holdings_info=hi_pos)

fa_neg = -fa
ic_neg = compute_ic_dynamic(fa_neg, ra, FORWARD_DAYS, "pearson")
rank_ic_neg = compute_ic_dynamic(fa_neg, ra, FORWARD_DAYS, "spearman")
gr_neg, tv_neg, hi_neg = compute_group_returns(fa_neg, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_neg = compute_metrics(gr_neg, ic_neg, rank_ic_neg, tv_neg, N_GROUPS, holdings_info=hi_neg)

pos_sh = m_pos.get("long_short_sharpe", 0) or 0
neg_sh = m_neg.get("long_short_sharpe", 0) or 0
pos_t = m_pos.get("ic_t_stat", 0) or 0
neg_t = m_neg.get("ic_t_stat", 0) or 0

print(f"   正向Sharpe={pos_sh:.4f} (高偏度→高收益, t={pos_t:.2f})")
print(f"   反向Sharpe={neg_sh:.4f} (低偏度→高收益, t={neg_t:.2f})")

if neg_sh > pos_sh:
    direction = -1
    direction_desc = "反向（低偏度=高预期收益，彩票效应负溢价）"
    ic_series = ic_neg
    rank_ic_series = rank_ic_neg
    group_returns = gr_neg
    turnovers = tv_neg
    metrics = m_neg
    holdings_info = hi_neg
    fa_final = fa_neg
    print(f"   → 反向胜 ✓ (彩票偏好效应 → 高偏度被高估 → 低偏度后续更好)")
else:
    direction = 1
    direction_desc = "正向（高偏度=高预期收益）"
    ic_series = ic_pos
    rank_ic_series = rank_ic_pos
    group_returns = gr_pos
    turnovers = tv_pos
    metrics = m_pos
    holdings_info = hi_pos
    fa_final = fa
    print(f"   → 正向胜 ✓")

# ────────────────── 20d调仓测试 ──────────────────
print(f"[6b] 测试20d调仓周期...")
gr_20, tv_20, hi_20 = compute_group_returns(fa_final, ra, N_GROUPS, 20, COST)
ic_20 = compute_ic_dynamic(fa_final, ra, 20, "pearson")
rank_ic_20 = compute_ic_dynamic(fa_final, ra, 20, "spearman")
m_20 = compute_metrics(gr_20, ic_20, rank_ic_20, tv_20, N_GROUPS, holdings_info=hi_20)
sh_20 = m_20.get("long_short_sharpe", 0) or 0
t_20 = m_20.get("ic_t_stat", 0) or 0
print(f"   20d调仓: Sharpe={sh_20:.4f} t={t_20:.2f}")
if sh_20 > (metrics.get("long_short_sharpe", 0) or 0) * 1.05:
    print(f"   → 20d调仓更优，切换")
    group_returns = gr_20
    turnovers = tv_20
    metrics = m_20
    holdings_info = hi_20
    ic_series = ic_20
    rank_ic_series = rank_ic_20
    REBALANCE_FREQ = 20
    FORWARD_DAYS = 20
else:
    print(f"   → 保持5d调仓")

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
oret = (open_piv / close_piv.shift(1)).clip(lower=0.001, upper=2.0) - 1
iret = (close_piv / open_piv).clip(lower=0.001, upper=2.0) - 1
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

# Neg day freq
neg_freq = (ret_piv <= -0.03).astype(float).rolling(10, min_periods=5).mean()

# Turnover level
turnover_level = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8))

# TAE
amplitude_piv = df.pivot_table(index="date", columns="stock_code", values="amplitude")
tae = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8) / (amplitude_piv.rolling(20, min_periods=10).mean().clip(lower=0.01)))

# vol_log60d
vol_log60d = np.log(1 + ret_piv.rolling(60, min_periods=30).std())

correlations = {}
for name, other in [
    ('amihud_illiq_v2', amihud_factor),
    ('shadow_pressure_v1', shadow),
    ('overnight_momentum_v1', overnight_mom),
    ('tail_risk_cvar_v1', cvar_df),
    ('neg_day_freq_v1', neg_freq),
    ('turnover_level_v1', turnover_level),
    ('tae_v1', tae),
    ('vol_log60d_v4', vol_log60d),
]:
    corrs = []
    for d in common_dates[::10]:
        val = fa_final.loc[d].dropna()
        oth = other.loc[d].reindex(val.index).dropna()
        c = val.index.intersection(oth.index)
        if len(c) > 50:
            r, _ = sp_stats.spearmanr(val[c], oth[c])
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
    "factor_name": "波动率偏度 v1",
    "factor_name_en": "Return Skewness v1",
    "category": "波动率/偏度",
    "description": f"过去{WINDOW}日收益率Fisher-Pearson偏度, 成交额OLS中性化+5%截尾。衡量收益分布尾部不对称性(正偏=彩票特征, 负偏=风险暴露)。",
    "hypothesis": "A股中证1000散户占比高, 彩票偏好效应显著: 正偏度(右尾肥)股票被散户追捧高估→后续收益差; 负偏度(左尾肥)股价风险补偿→后续收益好。",
    "formula": f"neutralize(skew(daily_ret, {WINDOW}), log_amount_20d)",
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

# ────────────────── 因子CSV ──────────────────
print(f"[8] 导出因子数据CSV...")
factor_csv = PROJECT_ROOT / "data" / f"factor_{FACTOR_ID}.csv"
rows = []
for date in fa_final.index:
    s = fa_final.loc[date].dropna()
    for code, val in s.items():
        rows.append({"date": date.strftime("%Y-%m-%d"), "stock_code": code, "factor_value": round(float(val), 6)})
pd.DataFrame(rows).to_csv(factor_csv, index=False)
print(f"   写入: {factor_csv} ({len(rows):,} 行)")

# ────────────────── 摘要 ──────────────────
ic_m = metrics.get("ic_mean", 0) or 0
ic_t = metrics.get("ic_t_stat", 0) or 0
ls_sh = metrics.get("long_short_sharpe", 0) or 0
ls_md = metrics.get("long_short_mdd", 0) or 0
mono = metrics.get("monotonicity", 0) or 0
sig = "✓" if metrics.get("ic_significant_5pct") else "✗"

print(f"\n{'═'*64}")
print(f"  {FACTOR_ID}: 波动率偏度因子 (Return Skewness)")
print(f"  方向: {direction_desc}")
print(f"{'═'*64}")
print(f"  区间:     {report['period']}")
print(f"  股票数:  {len(common_stocks)}")
print(f"  IC均值:  {ic_m:.4f}   (t={ic_t:.2f}, {sig})")
print(f"  Rank IC: {metrics.get('rank_ic_mean', 0):.4f}")
print(f"  IR:      {metrics.get('ir', 0):.4f}")
print(f"  IC>0占比:{metrics.get('ic_positive_pct', 0):.1%}")
print(f"  IC观察数:{metrics.get('ic_count', 0)}")
print(f"  多空Sharpe: {ls_sh:.4f}")
print(f"  多空MDD:   {ls_md:.2%}")
print(f"  单调性:    {mono:.4f}")
print(f"  换手率:    {metrics.get('turnover_mean', 0):.2%}")
print(f"{'─'*64}")
print(f"  分层年化收益:")
for i, r in enumerate(metrics.get("group_returns_annualized", []), 1):
    r_str = f"{r:.2%}" if r is not None else "N/A"
    bar = "█" * max(int((r or 0) * 200), 0)
    print(f"    G{i}: {r_str}  {bar}")

print(f"{'─'*64}")
print(f"  与入库因子相关性:")
for name, corr in sorted(correlations.items()):
    print(f"    vs {name}: {corr:.3f}")

print(f"{'═'*64}")
is_valid = abs(ic_m) > 0.015 and abs(ic_t) > 2 and abs(ls_sh) > 0.5 and mono >= 0.8
print(f"\n  ➤ 因子{'✅ 有效' if is_valid else '❌ 无效'}")
print(f"  标准: |IC|>0.015(|{ic_m:.4f}|>{'✓' if abs(ic_m)>0.015 else '✗'})")
print(f"       |t|>2      (|{ic_t:.2f}|  {'✓' if abs(ic_t)>2 else '✗'})")
print(f"       |Sharpe|>0.5({abs(ls_sh):.4f} {'✓' if abs(ls_sh)>0.5 else '✗'})")
print(f"       单调性≥0.8 ({mono:.3f} {'✓' if mono>=0.8 else '✗'})")
if is_valid:
    print(f"  → 入库: 写入 factors.json, git add+commit+push")
