#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: vra_rank_v1 — VRA 因子rank变换版
构造:
  1. 同 vol_ret_consistency_v1_20d: sign(vol_chg)*sign(ret) 20日均值
  2. 波动率标准化: /MA20(|ret|)
  3. 市值中性化 + MAD + **rank变换** (截面百分位排名)
  4. z-score

目标: 将G5极端优势(26%)转化为更高的线性IC(rank IC衡量)
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab/data")
OUTPUT_DIR = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab")
FACTOR_CSV = DATA_DIR / "factor_vra_rank_v1.csv"

print("=" * 60)
print("  VRA rank变换版")
print("=" * 60)

# [1] Load
print("\n[1] 加载数据...")
df = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

volume_piv = df.pivot_table(index="date", columns="stock_code", values="volume")
close_piv  = df.pivot_table(index="date", columns="stock_code", values="close")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")

ret_piv = close_piv.pct_change()

# [2] Alignment
print("\n[2] Sign alignment...")
sign_ret = np.sign(ret_piv)
sign_vol = np.sign(volume_piv.diff())
align = sign_ret * sign_vol
align_20d = align.rolling(20, min_periods=15).mean()

# [3] Vol normalization
print("\n[3] 波动率标准化...")
avg_abs_ret = ret_piv.abs().rolling(20, min_periods=15).mean().clip(lower=1e-6)
factor_raw = (align_20d / avg_abs_ret).clip(-5, 5)

# [4] Neutralize by market cap
print("\n[4] 市值中性化...")
log_amt_20d = np.log(amount_piv.rolling(20).mean().clip(lower=1))
factor_neutral = factor_raw.copy()

for date in sorted(factor_raw.dropna(how='all').index):
    f = factor_raw.loc[date].dropna()
    m = log_amt_20d.loc[date].reindex(f.index).dropna()
    common = f.index.intersection(m.index)
    if len(common) < 30:
        continue
    fv, mv = f[common].values.astype(float), m[common].values.astype(float)
    X = np.column_stack([np.ones(len(mv)), mv])
    try:
        beta = np.linalg.lstsq(X, fv, rcond=None)[0]
        factor_neutral.loc[date, common] = fv - X @ beta
    except:
        pass

# [5] **Rank transform** (key innovation vs vol_ret_align_v1)
print("\n[5] Rank变换 (放大G5极端优势)...")
factor_ranked = factor_neutral.copy()
for date in sorted(factor_neutral.dropna(how='all').index):
    f = factor_neutral.loc[date].dropna()
    if len(f) < 5:
        continue
    # Rank from 0 to 1
    ranks = f.rank(pct=True)
    mean_rank = ranks.mean()
    std_rank = ranks.std()
    if std_rank > 1e-10:
        factor_ranked.loc[date, f.index] = (ranks - mean_rank) / std_rank
    else:
        factor_ranked.loc[date, f.index] = 0

# [6] MAD + save
print("\n[6] 保存CSV...")
f_final = factor_ranked.copy()
for date in sorted(f_final.dropna(how='all').index):
    vals = f_final.loc[date].dropna()
    if len(vals) < 20:
        continue
    median = vals.median()
    mad = np.abs(vals - median).median()
    if mad < 1e-10:
        continue
    lo = median - 3.0 * 1.4826 * mad
    hi = median + 3.0 * 1.4826 * mad
    f_final.loc[date] = f_final.loc[date].clip(lo, hi)

long_df = f_final.stack().reset_index()
long_df.columns = ["date", "stock_code", "factor_value"]
long_df = long_df.sort_values(["date", "stock_code"]).dropna(subset=["factor_value"])
long_df.to_csv(FACTOR_CSV, index=False)

print(f"   → {FACTOR_CSV}")
print(f"   rows={len(long_df)}, stocks={long_df['stock_code'].nunique()}")
print(f"\n   rank范围: [{f_final.min().min():.3f}, {f_final.max().max():.3f}]")
print(f"   nonna_ratio: {f_final.notna().mean().mean():.2%}")
print(f"\n   Rank IC 目标: 将G5极端优势(26%)转化为>0.03的rank_ic")
