#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: hl_asym_v1 — 高低价不对称性因子 (High-Low Asymmetry)
============================================================

方向: 量价/微观结构

构造:
  1. 每日计算: (high - open) / (open - low + ε)  → 日内上行空间 / 下行空间
  2. 对数变换: log(ratio)
  3. 20日滚动均值
  4. 成交额OLS中性化
  5. 5%双边缩尾 + z-score

逻辑:
  high - open = 日内最大上行空间 (开盘后能涨多高)
  open - low  = 日内最大下行空间 (开盘后能跌多深)
  
  比值 > 1: 上行空间 > 下行空间 → 日内有持续买入力量支撑
  比值 < 1: 下行空间 > 上行空间 → 日内卖压重
  
  持续上行空间大的股票:
    → 可能反映知情买入(机构在日内逐步建仓)
    → 也可能反映散户追涨(FOMO) → 要看方向
  
  持续下行空间大的股票:
    → 日内卖压重 → 均值回复? 或趋势延续?
    → 要看经验数据

学术依据:
  - Alizadeh, Brandt & Diebold (2002) "Range-Based Estimation of Stochastic Volatility"
    → 日内range包含丰富的波动率和方向性信息
  - Parkinson (1980) "The Extreme Value Method for Estimating the Variance of the Rate of Return"
    → 高低价包含比收盘价更多的价格发现信息
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
FACTOR_ID = "hl_asym_v1"

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "csi1000_kline_raw.csv"
SCRIPTS_DIR = BASE_DIR / "skills" / "alpha-factor-lab" / "scripts"
OUTPUT_DIR = BASE_DIR / "output" / f"{FACTOR_ID}_5d"
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
turnover_piv = df.pivot_table(index="date", columns="stock_code", values="turnover")

ret_piv = close_piv.pct_change()
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
n_dates = len(dates)
n_stocks = len(stocks)
print(f"   {n_dates} 日, {n_stocks} 股")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造高低价不对称性因子 (window={WINDOW})...")

# 日内上行空间和下行空间
up_range = high_piv - open_piv      # 日内最大上涨
dn_range = open_piv - low_piv       # 日内最大下跌

# 避免除零：加小常数
eps = 0.001  # 相对于价格，这是非常小的
hl_ratio = up_range / (dn_range + eps)

# 对数变换
hl_log_ratio = np.log(hl_ratio.clip(lower=1e-8))

# 20日滚动均值
factor_raw = hl_log_ratio.rolling(WINDOW, min_periods=10).mean()

non_null = factor_raw.notna().mean().mean()
print(f"   非空率: {non_null:.2%}")
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
sys.path.insert(0, str(SCRIPTS_DIR))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data
)

common_dates = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
common_stocks = sorted(factor_neutral.columns.intersection(ret_piv.columns))
fa = factor_neutral.loc[common_dates, common_stocks]
ra = ret_piv.loc[common_dates, common_stocks]

# ────────────────── 方向探索 ──────────────────
print(f"[5] 方向探索...")

# 正向: 做多上行空间大的
ic_pos = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "pearson")
gr_pos, _, _ = compute_group_returns(fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_pos = compute_metrics(gr_pos, ic_pos, ic_pos, [], N_GROUPS)

# 反向: 做多下行空间大的
ic_neg = compute_ic_dynamic(-fa, ra, FORWARD_DAYS, "pearson")
gr_neg, _, _ = compute_group_returns(-fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_neg = compute_metrics(gr_neg, ic_neg, ic_neg, [], N_GROUPS)

pos_sh = m_pos.get("long_short_sharpe", 0) or 0
neg_sh = m_neg.get("long_short_sharpe", 0) or 0
pos_ic = m_pos.get("ic_mean", 0) or 0
neg_ic = m_neg.get("ic_mean", 0) or 0
pos_t = m_pos.get("ic_t_stat", 0) or 0
neg_t = m_neg.get("ic_t_stat", 0) or 0
pos_mono = m_pos.get("monotonicity", 0) or 0
neg_mono = m_neg.get("monotonicity", 0) or 0

print(f"   正向 (上行空间大): IC={pos_ic:.4f}, t={pos_t:.2f}, Sharpe={pos_sh:.4f}, mono={pos_mono:.2f}")
print(f"   反向 (下行空间大): IC={neg_ic:.4f}, t={neg_t:.2f}, Sharpe={neg_sh:.4f}, mono={neg_mono:.2f}")

# 选最好的方向
if neg_sh > pos_sh:
    direction = -1
    fa_final = -fa
    direction_desc = "反向（下行空间大=日内卖压重→高预期收益，均值回复逻辑）"
else:
    direction = 1
    fa_final = fa
    direction_desc = "正向（上行空间大=日内买入力量强→高预期收益）"

print(f"   → 选择方向: {direction_desc}")

# ────────────────── 最终回测 ──────────────────
print(f"[6] 最终回测 (方向={direction})...")

ic_series = compute_ic_dynamic(fa_final, ra, FORWARD_DAYS, "pearson")
rank_ic_series = compute_ic_dynamic(fa_final, ra, FORWARD_DAYS, "spearman")

group_returns, turnovers, holdings_info = compute_group_returns(
    fa_final, ra, N_GROUPS, REBALANCE_FREQ, COST
)
metrics = compute_metrics(
    group_returns, ic_series, rank_ic_series, turnovers, N_GROUPS,
    holdings_info=holdings_info
)

# ────────────────── 窗口+前瞻期敏感性 ──────────────────
print(f"[7] 窗口敏感性分析...")
best_sharpe = metrics.get("long_short_sharpe", 0) or 0
best_config = f"Window={WINDOW}, Fwd={FORWARD_DAYS}"

for test_window in [5, 10, 30, 40, 60]:
    test_factor = hl_log_ratio.rolling(test_window, min_periods=max(test_window//2, 3)).mean()
    # 缩尾
    for d in dates:
        row = test_factor.loc[d].dropna()
        if len(row) < 10:
            continue
        lo, hi = row.quantile(0.05), row.quantile(0.95)
        test_factor.loc[d] = test_factor.loc[d].clip(lo, hi)
    # 中性化
    for d in dates:
        f_t = test_factor.loc[d].dropna()
        m_t = log_amt.loc[d].reindex(f_t.index).dropna()
        c = f_t.index.intersection(m_t.index)
        if len(c) < 30:
            continue
        fc = f_t[c].values
        mc = m_t[c].values
        X = np.column_stack([np.ones(len(mc)), mc])
        try:
            beta = np.linalg.lstsq(X, fc, rcond=None)[0]
            test_factor.loc[d, c] = fc - X @ beta
        except:
            pass
    
    cd = sorted(test_factor.dropna(how="all").index.intersection(ra.index))
    cs = sorted(test_factor.columns.intersection(ra.columns))
    if len(cd) < 50:
        continue
    
    fa_t = (direction * test_factor).loc[cd, cs]
    
    for fwd in [5, 10, 20]:
        rebal = fwd
        ic_t = compute_ic_dynamic(fa_t, ra.loc[cd, cs], fwd, "pearson")
        gr_t, _, _ = compute_group_returns(fa_t, ra.loc[cd, cs], N_GROUPS, rebal, COST)
        m_t = compute_metrics(gr_t, ic_t, ic_t, [], N_GROUPS)
        sh = m_t.get('long_short_sharpe', 0) or 0
        ic_val = m_t.get('ic_mean', 0) or 0
        t_val = m_t.get('ic_t_stat', 0) or 0
        mono_val = m_t.get('monotonicity', 0) or 0
        
        marker = " ★" if abs(sh) > 0.5 and abs(t_val) > 2 else ""
        print(f"   W={test_window:2d}d F={fwd:2d}d: IC={ic_val:.4f}, t={t_val:.2f}, "
              f"Sharpe={sh:.4f}, mono={mono_val:.2f}{marker}")
        
        if abs(sh) > abs(best_sharpe):
            best_sharpe = sh
            best_config = f"Window={test_window}, Fwd={fwd}"

print(f"\n   最佳配置: {best_config}, Sharpe={best_sharpe:.4f}")

# ────────────────── 相关性分析 ──────────────────
print(f"\n[8] 与现有因子相关性...")

amplitude_piv = df.pivot_table(index="date", columns="stock_code", values="amplitude")

existing_factors = {}
amihud_raw = (ret_piv.abs() / (amount_piv / 1e8).clip(lower=1e-8))
existing_factors['amihud_illiq_v2'] = np.log(amihud_raw.rolling(20, min_periods=10).mean().clip(lower=1e-12))

upper_sr = (high_piv - np.maximum(close_piv, open_piv)) / (high_piv - low_piv).clip(lower=1e-8)
lower_sr = (np.minimum(close_piv, open_piv) - low_piv) / (high_piv - low_piv).clip(lower=1e-8)
existing_factors['shadow_pressure_v1'] = (upper_sr - lower_sr).rolling(20, min_periods=10).mean()

oret = open_piv / close_piv.shift(1) - 1
iret = close_piv / open_piv - 1
existing_factors['overnight_momentum_v1'] = oret.rolling(20, min_periods=10).sum() - iret.rolling(20, min_periods=10).sum()

existing_factors['turnover_level_v1'] = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8))
existing_factors['tae_v1'] = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8) / 
                                     (amplitude_piv.rolling(20, min_periods=10).mean().clip(lower=0.01)))

ret_vals = ret_piv.values
cvar_mat = np.full((n_dates, n_stocks), np.nan)
for i in range(10, n_dates):
    w = ret_vals[i-10:i, :]
    s = np.sort(w, axis=0)
    bot2 = np.nanmean(s[:2, :], axis=0)
    vc = np.sum(~np.isnan(w), axis=0)
    bot2[vc < 5] = np.nan
    cvar_mat[i, :] = -bot2
existing_factors['tail_risk_cvar_v1'] = pd.DataFrame(cvar_mat, index=dates, columns=stocks)
existing_factors['neg_day_freq_v1'] = (ret_piv <= -0.03).astype(float).rolling(10, min_periods=5).mean()

existing_factors['turnover_decel_v1'] = -np.log(
    turnover_piv.rolling(5, min_periods=3).mean().clip(lower=1e-8) / 
    turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8)
)

# MA disp
ma_dict = {}
for w in [5, 10, 20, 40, 60, 120]:
    ma_dict[w] = close_piv.rolling(w, min_periods=max(w//2, 3)).mean()
ma_disp = pd.DataFrame(np.nan, index=dates, columns=stocks)
for idx, d in enumerate(dates):
    vals = np.array([ma_dict[w].loc[d].values / close_piv.loc[d].values for w in [5, 10, 20, 40, 60, 120]])
    ma_disp.iloc[idx] = np.nanstd(vals, axis=0)
existing_factors['ma_disp_v1'] = ma_disp

existing_factors['amp_level_v2'] = np.log((amplitude_piv / 100).rolling(60, min_periods=30).mean().clip(lower=1e-8))

vol_std = turnover_piv.rolling(20, min_periods=10).std()
vol_mean = turnover_piv.rolling(20, min_periods=10).mean()
existing_factors['vol_cv_neg_v1'] = -(vol_std / vol_mean.clip(lower=1e-8))

correlations = {}
for name, other in existing_factors.items():
    corrs = []
    for d in common_dates[::10]:
        if d not in fa_final.index or d not in other.index:
            continue
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
    "factor_name": "高低价不对称性 v1",
    "factor_name_en": "High-Low Asymmetry v1",
    "category": "量价/微观结构",
    "description": f"过去{WINDOW}日 log((high-open)/(open-low)) 的均值，成交额OLS中性化。衡量日内上行空间vs下行空间的不对称性。",
    "hypothesis": "日内上行空间持续大于下行空间→买入力量强→价格发现正向；下行空间大→卖压重→可能均值回复。",
    "formula": f"neutralize(MA{WINDOW}(log((high-open)/(open-low+ε))), log_amount_20d)",
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
to_mean = metrics.get("turnover_mean", 0) or 0

print(f"\n{'='*60}")
print(f"  {FACTOR_ID}: 高低价不对称性")
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
print(f"  换手率:     {to_mean:.2%}")
print(f"{'─'*60}")
print(f"  分层年化收益:")
for i, r in enumerate(metrics.get("group_returns_annualized", []), 1):
    r_str = f"{r:.2%}" if r is not None else "N/A"
    bar = "█" * max(int((r or 0) * 100), 0)
    print(f"    G{i}: {r_str}  {bar}")
print(f"{'='*60}")

is_valid = abs(ic_m) > 0.015 and abs(ic_t) > 2 and abs(ls_sh) > 0.5
print(f"\n  ➤ 因子{'有效 ✓' if is_valid else '无效 ✗'}")
print(f"  评估标准: |IC|>0.015, |t|>2, |Sharpe|>0.5")
if is_valid:
    print(f"  → 准备写入factors.json并git提交")
else:
    print(f"  → 因子未达标，记录失败原因")
