#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成交量激增频率因子 (Volume Spike Frequency v1)
===============================================

来源: 方正金工《成交量激增时刻蕴含的alpha信息—多因子选股系列研究之一》
核心假设: 成交量激增时刻蕴含着市场参与者的"适度冒险"信号.

构造:
  W = 20d 滚动窗口
  每日定义 volume_ratio = volume / mean(volume, 20d)
  volume_spike = volume_ratio > 1.5 (即成交量超过20日均值的1.5倍)
  因子值 = 过去20个交易日中 volume_spike=True 的天数占比

假说方向:
  - 正面: 成交量激增反映知情交易者积极入场, 短期正收益惯性(LSV羊群效应)
  - 反面: 成交量激增可能是噪音交易/追涨杀跌, 短期反转

中性化: log_amount_20d OLS neutralization + MAD winsorize + z-score
"""

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
import warnings, sys, os
warnings.filterwarnings("ignore")

DATA_PATH = "data/csi1000_kline_raw.csv"
OUTPUT_PATH = "data/factor_vol_spike_freq_v1.csv"
WINDOW = 20
SPIKE_THRESHOLD = 1.5  # volume must be > 1.5x rolling 20d mean to be a spike

print("Reading K-line data...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
df = df[df["date"] <= "2026-04-15"]

# Budget proxy for market cap neutralization
df["log_amount_20d"] = df.groupby("stock_code")["amount"].transform(
    lambda x: np.log(x.rolling(20, min_periods=10).mean().clip(lower=1) + 1)
)

print(f"Computing {WINDOW}d rolling volume stats...")
grouped = df.groupby("stock_code")

# Rolling mean and std of volume
df["vol_ma"] = grouped["volume"].transform(
    lambda x: x.rolling(WINDOW, min_periods=WINDOW-2).mean()
)
df["vol_std"] = grouped["volume"].transform(
    lambda x: x.rolling(WINDOW, min_periods=WINDOW-2).std().fillna(0)
)

# Volume ratio = current volume / rolling mean
df["vol_ratio"] = df["volume"] / df["vol_ma"].clip(lower=1)

# Spike indicator: volume > mean + threshold*std
df["is_spike"] = (df["volume"] > (df["vol_ma"] + SPIKE_THRESHOLD * df["vol_std"])).astype(int)

print(f"Computing {WINDOW}d rolling spike frequency...")
df["spike_count"] = grouped["is_spike"].transform(
    lambda x: x.rolling(WINDOW, min_periods=WINDOW-4).sum()
)
df["factor_raw"] = df["spike_count"] / WINDOW

# Drop NaN
df = df.dropna(subset=["factor_raw", "log_amount_20d"]).copy()

# MAD winsorize
median = df["factor_raw"].median()
mad = (df["factor_raw"] - median).abs().median()
upper = median + 5.2 * mad
lower = median - 5.2 * mad
df["factor_raw"] = df["factor_raw"].clip(lower, upper)

# OLS neutralization against log_amount_20d
mask = df["log_amount_20d"].notna() & df["factor_raw"].notna()
y = df.loc[mask, "factor_raw"].values
X = df.loc[mask, ["log_amount_20d"]].values
# Add constant
X_design = np.column_stack([np.ones(len(X)), X])
beta, _, _, _ = lstsq(X_design, y, rcond=None)
residual = y - X_design @ beta

# z-score
std = residual.std()
if std > 0:
    residual = (residual - residual.mean()) / std
else:
    residual = residual - residual.mean()

df.loc[mask, "factor_value"] = residual

# Output
out = df.dropna(subset=["factor_value"])[["date", "stock_code", "factor_value"]].copy()
out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
out.to_csv(OUTPUT_PATH, index=False)

print(f"\nFactor computed: {OUTPUT_PATH}")
print(f"  Records: {len(out)}")
print(f"  Date range: {out['date'].min()} ~ {out['date'].max()}")
print(f"  Stocks per date (avg): {out.groupby('date')['stock_code'].count().mean():.0f}")
print(f"  Factor stats: mean={out['factor_value'].mean():.4f}, std={out['factor_value'].std():.4f}")
