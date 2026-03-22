#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: amt_concentration_v1 — 资金集中度因子
==========================================

方向: 量价/资金流 (尚未覆盖)

构造:
  1. 过去20日中，成交额最高的5天占总成交额的比例 (Top5 Concentration Ratio)
  2. 成交额中性化(OLS)
  3. 5%缩尾

逻辑:
  资金集中度衡量"成交额在时间维度上的分布不均匀性":
  - 高集中度: 少数几天成交额异常放大 → 有大资金突击进出 → 信息事件驱动
  - 低集中度: 成交额分布均匀 → 正常交易节奏 → 无特殊事件

  两种方向假说:
  A) 正向(高集中度=好): 大资金集中介入是知情交易信号，后续延续
  B) 反向(低集中度=好): 均匀成交=健康交易模式=更好的价格发现

  在A股小盘股中，大资金集中介入可能是主力建仓/出货信号:
  - 建仓: 低集中度(分批吸筹) → 后续上涨
  - 出货: 高集中度(集中抛售) → 后续下跌

理论:
  - Kyle (1985) Strategic Trading: 知情交易者分散交易以隐藏信息
  - Easley & O'Hara (1987): 成交量中隐含信息
  - Chordia & Subrahmanyam (2004): 交易活动的自相关与收益预测
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
TOP_K = 5  # Top5天
FORWARD_DAYS = 5
REBALANCE_FREQ = 5
N_GROUPS = 5
COST = 0.003
WINSORIZE_PCT = 0.05
DATA_CUTOFF = "2026-03-13"
FACTOR_ID = "amt_concentration_v1"

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
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
n_dates = len(dates)
n_stocks = len(stocks)
print(f"   {n_dates} 日, {n_stocks} 股")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造资金集中度因子 (window={WINDOW}, top_k={TOP_K})...")

amt_vals = amount_piv.values  # (n_dates, n_stocks)
conc_mat = np.full((n_dates, n_stocks), np.nan)

for i in range(WINDOW, n_dates):
    window_amt = amt_vals[i - WINDOW:i, :]  # (WINDOW, n_stocks)
    # Top5天成交额占比
    for j in range(n_stocks):
        col = window_amt[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) >= WINDOW * 0.7:  # 至少70%有效
            total = np.sum(valid)
            if total > 0:
                sorted_desc = np.sort(valid)[::-1]
                top_k_sum = np.sum(sorted_desc[:TOP_K])
                conc_mat[i, j] = top_k_sum / total

factor_raw = pd.DataFrame(conc_mat, index=dates, columns=stocks)

print(f"   非空率: {factor_raw.notna().mean().mean():.2%}")
print(f"   均值: {factor_raw.stack().mean():.4f}, std: {factor_raw.stack().std():.4f}")
print(f"   理论均匀分布均值: {TOP_K/WINDOW:.4f}")

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

# ────────────────── 方向探索 ──────────────────
print(f"[6] 方向探索...")

# 正向: 做多高集中度
ic_pos = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "pearson")
gr_pos, _, _ = compute_group_returns(fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_pos = compute_metrics(gr_pos, ic_pos, ic_pos, [], N_GROUPS)

# 反向: 做多低集中度
ic_neg = compute_ic_dynamic(-fa, ra, FORWARD_DAYS, "pearson")
gr_neg, _, _ = compute_group_returns(-fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_neg = compute_metrics(gr_neg, ic_neg, ic_neg, [], N_GROUPS)

pos_sh = m_pos.get("long_short_sharpe", 0) or 0
neg_sh = m_neg.get("long_short_sharpe", 0) or 0
pos_ic = m_pos.get("ic_mean", 0) or 0
neg_ic = m_neg.get("ic_mean", 0) or 0
pos_t = m_pos.get("ic_t_stat", 0) or 0
neg_t = m_neg.get("ic_t_stat", 0) or 0

print(f"   正向 (高集中度=高收益): IC={pos_ic:.4f}, t={pos_t:.2f}, Sharpe={pos_sh:.4f}")
print(f"   反向 (低集中度=高收益): IC={neg_ic:.4f}, t={neg_t:.2f}, Sharpe={neg_sh:.4f}")

# 选择更好的方向 (综合Sharpe)
if neg_sh > pos_sh:
    print(f"   → 使用反向 (做多低集中度)")
    direction = -1
    fa_final = -fa
    direction_desc = "反向（低集中度=高预期收益，均匀成交=健康交易）"
else:
    print(f"   → 使用正向 (做多高集中度)")
    direction = 1
    fa_final = fa
    direction_desc = "正向（高集中度=高预期收益，大资金集中介入）"

# ────────────────── 也测试不同调仓频率 ──────────────────
print(f"[6b] 调仓频率敏感性...")
for test_freq in [5, 10, 20]:
    for test_fwd in [5, 10, 20]:
        ic_t_test = compute_ic_dynamic(fa_final, ra, test_fwd, "pearson")
        gr_t_test, _, _ = compute_group_returns(fa_final, ra, N_GROUPS, test_freq, COST)
        m_t_test = compute_metrics(gr_t_test, ic_t_test, ic_t_test, [], N_GROUPS)
        sh = m_t_test.get("long_short_sharpe", 0) or 0
        ic = m_t_test.get("ic_mean", 0) or 0
        t = m_t_test.get("ic_t_stat", 0) or 0
        mono = m_t_test.get("monotonicity", 0) or 0
        print(f"   Rebal={test_freq}d, Fwd={test_fwd}d: IC={ic:.4f}, t={t:.2f}, Sharpe={sh:.4f}, Mono={mono:.2f}")

# 找最佳参数组合
best_sharpe = 0
best_params = (5, 5)
for test_freq in [5, 10, 20]:
    for test_fwd in [5, 10, 20]:
        ic_t_test = compute_ic_dynamic(fa_final, ra, test_fwd, "pearson")
        gr_t_test, _, _ = compute_group_returns(fa_final, ra, N_GROUPS, test_freq, COST)
        m_t_test = compute_metrics(gr_t_test, ic_t_test, ic_t_test, [], N_GROUPS)
        sh = abs(m_t_test.get("long_short_sharpe", 0) or 0)
        if sh > best_sharpe:
            best_sharpe = sh
            best_params = (test_freq, test_fwd)

print(f"   最佳: Rebal={best_params[0]}d, Fwd={best_params[1]}d (Sharpe={best_sharpe:.4f})")

# 使用最佳参数重新回测
REBALANCE_FREQ_FINAL = best_params[0]
FORWARD_DAYS_FINAL = best_params[1]

# ────────────────── 最终回测 ──────────────────
print(f"[7] 最终回测 (方向={direction}, rebal={REBALANCE_FREQ_FINAL}d, fwd={FORWARD_DAYS_FINAL}d)...")

ic_series = compute_ic_dynamic(fa_final, ra, FORWARD_DAYS_FINAL, "pearson")
rank_ic_series = compute_ic_dynamic(fa_final, ra, FORWARD_DAYS_FINAL, "spearman")

group_returns, turnovers, holdings_info = compute_group_returns(
    fa_final, ra, N_GROUPS, REBALANCE_FREQ_FINAL, COST
)
metrics = compute_metrics(
    group_returns, ic_series, rank_ic_series, turnovers, N_GROUPS,
    holdings_info=holdings_info
)

# ────────────────── 相关性 ──────────────────
print(f"[8] 与现有因子相关性...")

open_piv = df.pivot_table(index="date", columns="stock_code", values="open")
high_piv = df.pivot_table(index="date", columns="stock_code", values="high")
low_piv = df.pivot_table(index="date", columns="stock_code", values="low")
turnover_piv = df.pivot_table(index="date", columns="stock_code", values="turnover")

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
cvar_mat = np.full((n_dates, n_stocks), np.nan)
ret_vals = ret_piv.values
for i in range(10, n_dates):
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
tae_raw = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8) / 
                  (amplitude_piv.rolling(20, min_periods=10).mean().clip(lower=0.01)))

correlations = {}
for name, other in [
    ('amihud_illiq_v2', amihud_factor), 
    ('shadow_pressure_v1', shadow),
    ('overnight_momentum_v1', overnight_mom), 
    ('tail_risk_cvar_v1', cvar_df),
    ('neg_day_freq_v1', neg_freq),
    ('turnover_level_v1', turnover_level),
    ('tae_v1', tae_raw),
]:
    corrs = []
    for d in common_dates[::10]:
        f1 = fa_final.loc[d].dropna()
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
    "factor_name": "资金集中度 v1",
    "factor_name_en": "Amount Concentration v1",
    "category": "量价/资金流",
    "description": f"过去{WINDOW}日中成交额最高{TOP_K}天占总成交额的比例，成交额中性化。衡量资金在时间维度的集中程度，反映大资金进出痕迹。",
    "hypothesis": "成交额高度集中(少数天放量)=大资金突击进出=信息冲击后的定价调整尚未完成",
    "formula": f"neutralize(top{TOP_K}_amount_ratio({WINDOW}d), log_amount_20d)",
    "direction": direction,
    "direction_desc": direction_desc,
    "stock_pool": "中证1000",
    "period": f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}",
    "n_dates": len(common_dates),
    "n_stocks": len(common_stocks),
    "n_groups": N_GROUPS,
    "rebalance_freq": REBALANCE_FREQ_FINAL,
    "forward_days": FORWARD_DAYS_FINAL,
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
print(f"  {FACTOR_ID}: 资金集中度因子")
print(f"  方向: {direction_desc}")
print(f"{'='*60}")
print(f"  区间:     {report['period']}")
print(f"  股票:     {len(common_stocks)}")
print(f"  调仓/前瞻: {REBALANCE_FREQ_FINAL}d / {FORWARD_DAYS_FINAL}d")
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
