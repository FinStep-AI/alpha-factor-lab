#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: trend_day_ratio_v1 — 趋势日比率因子 (优化版)
====================================================
测试多种变体找最优构造

变体:
  A) 原始: MA20(|close-open|/(high-low)), 成交额中性化
  B) 成交量加权: MA20(volume * |close-open|/(high-low)) / MA20(volume)
  C) 只计数阈值: 过去20日中 |close-open|/(high-low) > 0.6 的天数占比
  D) 考虑方向: MA20(sign(ret) * |close-open|/(high-low))，看有方向的趋势效率
  E) 对数市值中性化（替代成交额）
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
N_GROUPS = 5
COST = 0.003
WINSORIZE_PCT = 0.05
DATA_CUTOFF = "2026-03-13"

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "csi1000_kline_raw.csv"
SCRIPTS_DIR = BASE_DIR / "skills" / "alpha-factor-lab" / "scripts"

# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] <= DATA_CUTOFF]
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
open_piv = df.pivot_table(index="date", columns="stock_code", values="open")
high_piv = df.pivot_table(index="date", columns="stock_code", values="high")
low_piv = df.pivot_table(index="date", columns="stock_code", values="low")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")
volume_piv = df.pivot_table(index="date", columns="stock_code", values="volume")
turnover_piv = df.pivot_table(index="date", columns="stock_code", values="turnover")

ret_piv = close_piv.pct_change()
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

sys.path.insert(0, str(SCRIPTS_DIR))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data
)

eps = 1e-8
daily_trend_eff = (close_piv - open_piv).abs() / (high_piv - low_piv + eps)
daily_trend_eff = daily_trend_eff.clip(0, 1)

def neutralize_factor(factor_raw, log_control, dates_list):
    """OLS中性化"""
    factor_neutral = factor_raw.copy()
    for date in dates_list:
        f = factor_raw.loc[date].dropna()
        m = log_control.loc[date].reindex(f.index).dropna()
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
    return factor_neutral

def winsorize_factor(factor_df, dates_list, pct=0.05):
    """缩尾"""
    for date in dates_list:
        row = factor_df.loc[date].dropna()
        if len(row) < 10:
            continue
        lo = row.quantile(pct)
        hi = row.quantile(1 - pct)
        factor_df.loc[date] = factor_df.loc[date].clip(lo, hi)
    return factor_df

def test_factor(name, factor_raw, ret_piv, log_control, dates_list):
    """测试一个因子变体"""
    factor_w = winsorize_factor(factor_raw.copy(), dates_list)
    factor_n = neutralize_factor(factor_w, log_control, dates_list)
    
    common_dates = sorted(factor_n.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
    common_stocks = sorted(factor_n.columns.intersection(ret_piv.columns))
    fa = factor_n.loc[common_dates, common_stocks]
    ra = ret_piv.loc[common_dates, common_stocks]
    
    results = {}
    for fwd, rebal in [(5, 5), (20, 20), (10, 10)]:
        ic = compute_ic_dynamic(fa, ra, fwd, "pearson")
        gr, to, hi = compute_group_returns(fa, ra, N_GROUPS, rebal, COST)
        m = compute_metrics(gr, ic, ic, to, N_GROUPS, holdings_info=hi)
        
        # 也测试反向
        ic_neg = compute_ic_dynamic(-fa, ra, fwd, "pearson")
        gr_neg, to_neg, hi_neg = compute_group_returns(-fa, ra, N_GROUPS, rebal, COST)
        m_neg = compute_metrics(gr_neg, ic_neg, ic_neg, to_neg, N_GROUPS, holdings_info=hi_neg)
        
        pos_sh = m.get("long_short_sharpe", 0) or 0
        neg_sh = m_neg.get("long_short_sharpe", 0) or 0
        
        if neg_sh > pos_sh:
            ic, gr, to, hi, m = ic_neg, gr_neg, to_neg, hi_neg, m_neg
            dirn = -1
        else:
            dirn = 1
        
        ic_m = m.get("ic_mean", 0) or 0
        ic_t = m.get("ic_t_stat", 0) or 0
        sh = m.get("long_short_sharpe", 0) or 0
        mono = m.get("monotonicity", 0) or 0
        mdd = m.get("long_short_mdd", 0) or 0
        
        results[f"{fwd}d"] = {
            "ic_mean": ic_m, "ic_t": ic_t, "sharpe": sh, 
            "mono": mono, "mdd": mdd, "direction": dirn,
            "group_returns": m.get("group_returns_annualized", []),
            "metrics": m, "ic_series": ic, "group_returns_series": gr,
            "turnovers": to, "holdings_info": hi, "fa": fa
        }
    
    return results

# ────────────────── 变体A: 原始MA20 ──────────────────
print(f"\n[A] 原始 MA20(趋势效率)...")
var_a = daily_trend_eff.rolling(WINDOW, min_periods=10).mean()
results_a = test_factor("var_a", var_a, ret_piv, log_amt, dates)
for k, v in results_a.items():
    print(f"   {k}: IC={v['ic_mean']:.4f}(t={v['ic_t']:.2f}), Sharpe={v['sharpe']:.4f}, Mono={v['mono']:.2f}, dir={v['direction']}")

# ────────────────── 变体B: 成交量加权 ──────────────────
print(f"\n[B] 成交量加权趋势效率...")
vol_weighted = (volume_piv * daily_trend_eff).rolling(WINDOW, min_periods=10).sum() / volume_piv.rolling(WINDOW, min_periods=10).sum().clip(lower=1)
results_b = test_factor("var_b", vol_weighted, ret_piv, log_amt, dates)
for k, v in results_b.items():
    print(f"   {k}: IC={v['ic_mean']:.4f}(t={v['ic_t']:.2f}), Sharpe={v['sharpe']:.4f}, Mono={v['mono']:.2f}, dir={v['direction']}")

# ────────────────── 变体C: 高趋势效率日占比 ──────────────────
print(f"\n[C] 高趋势效率日占比(>0.5)...")
trend_days = (daily_trend_eff > 0.5).astype(float).rolling(WINDOW, min_periods=10).mean()
results_c = test_factor("var_c", trend_days, ret_piv, log_amt, dates)
for k, v in results_c.items():
    print(f"   {k}: IC={v['ic_mean']:.4f}(t={v['ic_t']:.2f}), Sharpe={v['sharpe']:.4f}, Mono={v['mono']:.2f}, dir={v['direction']}")

# ────────────────── 变体D: 有方向的趋势效率 ──────────────────
print(f"\n[D] 有方向趋势效率 sign(ret)*效率...")
sign_ret = np.sign(ret_piv)
directional_eff = sign_ret * daily_trend_eff
directional_avg = directional_eff.rolling(WINDOW, min_periods=10).mean()
results_d = test_factor("var_d", directional_avg, ret_piv, log_amt, dates)
for k, v in results_d.items():
    print(f"   {k}: IC={v['ic_mean']:.4f}(t={v['ic_t']:.2f}), Sharpe={v['sharpe']:.4f}, Mono={v['mono']:.2f}, dir={v['direction']}")

# ────────────────── 变体E: 趋势效率标准差(波动) ──────────────────
print(f"\n[E] 趋势效率波动(std)...")
trend_std = daily_trend_eff.rolling(WINDOW, min_periods=10).std()
results_e = test_factor("var_e", trend_std, ret_piv, log_amt, dates)
for k, v in results_e.items():
    print(f"   {k}: IC={v['ic_mean']:.4f}(t={v['ic_t']:.2f}), Sharpe={v['sharpe']:.4f}, Mono={v['mono']:.2f}, dir={v['direction']}")

# ────────────────── 选最优 ──────────────────
print(f"\n{'='*60}")
print(f"  最优变体选择")
print(f"{'='*60}")
all_results = {
    'A_原始MA20': results_a,
    'B_量加权': results_b,
    'C_占比>0.5': results_c,
    'D_方向效率': results_d,
    'E_效率波动': results_e,
}

best_name = None
best_score = -999
best_fwd = None
for vname, vres in all_results.items():
    for fwd, r in vres.items():
        ic_t = abs(r['ic_t'])
        sh = abs(r['sharpe'])
        mono = r['mono']
        # 综合评分: 权重 IC_t=30% + Sharpe=40% + 单调性=30%
        score = 0.3 * min(ic_t / 3, 1) + 0.4 * min(sh / 2, 1) + 0.3 * min(mono / 1, 1)
        valid = abs(r['ic_mean']) > 0.015 and ic_t > 2 and sh > 0.5
        mark = "✓" if valid else "✗"
        print(f"  {vname} [{fwd}]: score={score:.3f} {mark} IC_t={ic_t:.2f} Sh={sh:.4f} Mono={mono:.2f}")
        if score > best_score and valid:
            best_score = score
            best_name = vname
            best_fwd = fwd

if best_name:
    print(f"\n  ★ 最优: {best_name} [{best_fwd}] (score={best_score:.3f})")
else:
    print(f"\n  ★ 无达标变体")
