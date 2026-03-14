#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: gap_fill_v1 — 跳空回补率
构造:
  1. gap = (open - prev_close) / prev_close
  2. fill = -(close - open) / (open - prev_close)  当 gap != 0
     fill > 0 = 日内朝反方向走(回补缺口)
     fill > 1 = 完全回补并反超
     fill < 0 = 日内顺着跳空方向走(加速)
  3. 20日均值(只统计|gap| > 0.5%的天,避免噪声)
  4. 正向使用: 做多高回补率(日内反转强 → 均值回复)
  5. 也测反向
  6. 市值中性化 + 5%缩尾

逻辑: 
- 高回补率 = 跳空后日内持续被纠正 = 有"纠错力量"(知情交易者or做市商)
- 与隔夜动量/跳空缺口动量互补:那些看方向,这个看回补速度
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WINDOW = 20
GAP_THRESHOLD = 0.005  # 只看|gap|>0.5%的天
WINSORIZE_PCT = 0.05
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "csi1000_kline_raw.csv"
BASE_OUTPUT = Path(__file__).resolve().parent.parent / "output"

print(f"[1] 加载数据...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
open_piv = df.pivot_table(index="date", columns="stock_code", values="open")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")

ret_piv = close_piv.pct_change()
log_mktcap = np.log(amount_piv.rolling(20).mean().clip(lower=1))

prev_close = close_piv.shift(1)
gap = (open_piv - prev_close) / prev_close.clip(lower=0.01)
intra_move = (close_piv - open_piv)  
gap_abs = (open_piv - prev_close)

# 回补率: -intra_move / gap_abs (正值=回补)
# 当gap>0: open>prev_close, 如果close<open则回补
# 当gap<0: open<prev_close, 如果close>open则回补
gap_fill_daily = -intra_move / gap_abs.clip(lower=1e-6)
# 只在有意义的缺口时计算
gap_fill_daily[gap.abs() < GAP_THRESHOLD] = np.nan
# clip极端值
gap_fill_daily = gap_fill_daily.clip(-5, 5)

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# 20日均值(忽略NaN即无缺口的天)
print(f"[2] 构造 gap_fill 因子 (window={WINDOW})...")
factor_matrix = gap_fill_daily.rolling(WINDOW, min_periods=5).mean()
print(f"   因子非空率: {factor_matrix.notna().mean().mean():.2%}")


def full_pipeline(factor_raw, direction, fwd, rebal, cost, factor_id, label):
    factor = factor_raw.copy()
    if direction == -1:
        factor = -factor
    
    for date in dates:
        row = factor.loc[date].dropna()
        if len(row) < 10:
            continue
        lo = row.quantile(WINSORIZE_PCT)
        hi = row.quantile(1 - WINSORIZE_PCT)
        factor.loc[date] = factor.loc[date].clip(lo, hi)
    
    for date in dates:
        f = factor.loc[date].dropna()
        m = log_mktcap.loc[date].reindex(f.index).dropna()
        common = f.index.intersection(m.index)
        if len(common) < 30:
            continue
        f_c = f[common].values
        m_c = m[common].values
        X = np.column_stack([np.ones(len(m_c)), m_c])
        try:
            beta = np.linalg.lstsq(X, f_c, rcond=None)[0]
            factor.loc[date, common] = f_c - X @ beta
        except:
            pass
    
    cd = sorted(factor.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
    cs = sorted(factor.columns.intersection(ret_piv.columns))
    fa = factor.loc[cd, cs]
    ra = ret_piv.loc[cd, cs]
    
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills" / "alpha-factor-lab" / "scripts"))
    from factor_backtest import (
        compute_group_returns, compute_ic_dynamic,
        compute_metrics, save_backtest_data
    )
    
    ic = compute_ic_dynamic(fa, ra, fwd, "pearson")
    ric = compute_ic_dynamic(fa, ra, fwd, "spearman")
    gr, to, hi = compute_group_returns(fa, ra, 5, rebal, cost)
    metrics = compute_metrics(gr, ic, ric, to, 5, holdings_info=hi)
    
    ic_m = metrics.get("ic_mean", 0) or 0
    ic_t = metrics.get("ic_t_stat", 0) or 0
    ls_sh = metrics.get("long_short_sharpe", 0) or 0
    mono = metrics.get("monotonicity", 0) or 0
    sig = "✓" if metrics.get("ic_significant_5pct") else "✗"
    
    print(f"  [{label}] IC={ic_m:.4f}(t={ic_t:.2f},{sig}) Sh={ls_sh:.2f} Mono={mono:.2f}")
    grp = metrics.get("group_returns_annualized", [])
    for i, r in enumerate(grp, 1):
        print(f"    G{i}: {r:.2%}" if r is not None else f"    G{i}: N/A")
    
    is_valid = abs(ic_m) > 0.015 and abs(ic_t) > 2 and abs(ls_sh) > 0.5
    print(f"  ➤ {'有效 ✓' if is_valid else '无效 ✗'}")
    
    if is_valid:
        out = BASE_OUTPUT / factor_id
        out.mkdir(parents=True, exist_ok=True)
        save_backtest_data(gr, ic, ric, str(out))
    
    return metrics, is_valid


print(f"\n[3] 回测各配置...")
configs = [
    (factor_matrix, 1, 5, 5, 0.003, "gap_fill_pos_5d", "正向(做多高回补) fwd=5d"),
    (factor_matrix, -1, 5, 5, 0.003, "gap_fill_neg_5d", "反向(做多低回补) fwd=5d"),
    (factor_matrix, 1, 20, 20, 0.002, "gap_fill_pos_20d", "正向(做多高回补) fwd=20d"),
    (factor_matrix, -1, 20, 20, 0.002, "gap_fill_neg_20d", "反向(做多低回补) fwd=20d"),
]

any_valid = False
for fraw, d, fwd, reb, cost, fid, lab in configs:
    m, v = full_pipeline(fraw, d, fwd, reb, cost, fid, lab)
    if v:
        any_valid = True

if not any_valid:
    print(f"\n❌ gap_fill 因子所有配置均未达标")
