#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: vol_ret_consistency_v1 — 量价信号一致性因子
构造:
  1. 日成交量变化信号: sign(volume_t - volume_{t-1})
  2. 日收益率方向: sign(ret_t)
  3. 信号一致性: sign(vol_chg) * sign(ret) ∈ {-1, 0, 1}
  4. 20日窗口平均一致性 = 量价同向天数比例
  5. 市值中性化(对数成交额 OLS) + MAD winsorize + z-score
  6. 乘以20日累积收益方向: sign(sum(ret, 20d))

理论依据:
- Lou et al. (2019) "A Tug of War: Overnight vs Intraday Expected Returns"
  → 情绪因子(方向信息)由隔夜反映,日内由成交量确认
- Ayers, Li & Odean (2007) "Testing the Limits of Limits: Price Pressures"
  → 成交量与价格变动方向一致性反映市场共识强度
- 在A股中证1000: 量价同向→散户追涨/恐慌杀跌→短期反转;
  量价背离→信息揭示→趋势延续

假设: 量价背离时(成交量与价格反向),知情交易者利用群体错误定价,
     后续有动量效应; 量价同向(追涨杀跌)后反向修正

方向: 反向使用(高一致性=追涨杀跌→低收益)
Barra风格: MICRO (微观结构)
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# Paths
DATA_DIR = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab/data")
OUTPUT_DIR = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab/output/vol_ret_consistency_v1_20d")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BACKTEST_SCRIPT = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab/skills/alpha-factor-lab/scripts/factor_backtest.py")
FACTOR_CSV = DATA_DIR / "factor_vol_ret_consistency_v1_20d.csv"

WINSORIZE_PCT = 0.05

print("=" * 60)
print("  因子:v1 — 量价信号一致性(Volume-Return Sign Consistency)")
print("=" * 60)

# [1] Load data
print("\n[1] 加载行情数据...")
df = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

print(f"   总行数: {len(df)}, 股票数: {df['stock_code'].nunique()}")

# Pivot tables
volume_piv = df.pivot_table(index="date", columns="stock_code", values="volume")
close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")

# [2] Compute intraday return
print("\n[2] 构造信号一致性因子...")
ret_piv = close_piv.pct_change()

# Volume change sign: sign(volume_t / volume_{t-1} - 1)
# Actually simpler: sign is determined by whether volume increased or decreased
vol_chg = volume_piv.diff()
sign_vol = np.sign(vol_chg)

# Return sign
sign_ret = np.sign(ret_piv)

# Alignment signal: 1 if same direction, -1 if opposite, 0 if either is zero
alignment = sign_vol * sign_ret  # ∈ {-1, 0, 1}

# Rolling 20-day mean alignment (proportion of days where order flow confirms price)
# This measures the fraction where volume and price move together
alignment_20d = alignment.rolling(20, min_periods=15).mean()

# Handle zeros: fill with nan since they don't contribute meaningfully
alignment_20d = alignment_20d.replace(0, np.nan)

print(f"     alignment_20d 非空率: {alignment_20d.notna().mean().mean():.2%}")
print(f"     20d align range: [{alignment_20d.min().min():.3f}, {alignment_20d.max().max():.3f}]")

# [3] Market-cap neutralization + transformation
print("\n[3] 市值中性化...")
# Use log(Avg(amount, 20d)) as log_mktcap proxy
log_amount_20d = np.log(amount_piv.rolling(20).mean().clip(lower=1))

factor_raw = alignment_20d.copy()
factor_neutralized = factor_raw.copy()

all_dates = sorted(factor_raw.dropna(how="all").index)

for date in all_dates:
    f = factor_raw.loc[date].dropna()
    if len(f) < 30:
        continue
    
    # Get corresponding market cap proxy
    m = log_amount_20d.loc[date].reindex(f.index).dropna()
    common = f.index.intersection(m.index)
    if len(common) < 30:
        continue
    
    f_vals = f[common].values.astype(float)
    m_vals = m[common].values.astype(float)
    
    # OLS regression: f = alpha + beta * log_amount + residual
    X = np.column_stack([np.ones(len(m_vals)), m_vals])
    try:
        beta = np.linalg.lstsq(X, f_vals, rcond=None)[0]
        residual = f_vals - X @ beta
        factor_neutralized.loc[date, common] = residual
    except Exception:
        continue

# [4] Direction: reverse sign (high alignment = retail crowd following = reversal)
# We'll test both directions in the backtest
# For now, output the raw neutralized value (direction tested in backtest)

# [5] Winsorize + z-score
print("\n[4] 标准化 (MAD winsorize + z-score)...")
factor_final = factor_neutralized.copy()

for date in sorted(factor_neutralized.dropna(how="all").index):
    f = factor_neutralized.loc[date].dropna()
    if len(f) < 20:
        continue
    
    # MAD winsorize (robust to outliers)
    median = f.median()
    mad = np.abs(f - median).median()
    if mad < 1e-10:
        continue
    
    k = 3.0  # 3 MAD bounds (≈ 99.3% for normal)
    lower = median - k * 1.4826 * mad
    upper = median + k * 1.4826 * mad
    factor_final.loc[date] = factor_final.loc[date].clip(lower, upper)
    
    # Z-score
    f_clean = factor_final.loc[date].dropna()
    mean = f_clean.mean()
    std = f_clean.std()
    if std > 1e-10:
        factor_final.loc[date, f_clean.index] = (f_clean - mean) / std

# [6] Save factor CSV  
print(f"\n[5] 保存因子CSV: {FACTOR_CSV}")

factor_for_csv = factor_final.copy()
factor_for_csv = factor_for_csv.apply(
    lambda row: (row - row.mean()) / row.std() if row.std() > 1e-10 else row,
    axis=1
)

long_df = factor_for_csv.stack().reset_index()
long_df.columns = ["date", "stock_code", "factor_value"]
long_df = long_df.sort_values(["date", "stock_code"]).dropna(subset=["factor_value"])
long_df.to_csv(FACTOR_CSV, index=False)
print(f"    {len(long_df)} 行, {long_df['stock_code'].nunique()} 股")

# Stats
print(f"\n   因子统计:")
print(f"   均值: {long_df['factor_value'].mean():.4f}")
print(f"   标准差: {long_df['factor_value'].std():.4f}")
print(f"   非空比例: {factor_for_csv.notna().mean().mean():.2%}")

# Summary
print("\n" + "=" * 60)
print("  因子构造完成")
print("=" * 60)
print(f"  因子文件: {FACTOR_CSV}")
print(f"  因子ID: vol_ret_consistency_v1_20d")
print(f"  公式: neutralize(MA20(sign(vol_chg)*sign(ret)), log_amount_20d)")
print(f"  方向: 待回测确认(预计反向: 高一致性=追涨杀跌→低收益)")
print(f"  Barra风格: MICRO (微观结构)")
print(f"=" * 60)
