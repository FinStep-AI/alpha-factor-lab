#!/usr/bin/env python3
"""
因子: 振幅收缩因子 v2 (Amplitude Compression v2)

改进amp_compress_v1 (IC=0.022 t=2.06 mono=0.7):
- v1: -log(MA5_amp / MA20_amp)
- v2: 测试更长基准窗口: MA5/MA40, MA5/MA60, MA10/MA60

高因子值 = 短期振幅远低于长期 = 波动率收缩 = 蓄势状态
"""

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
import warnings
warnings.filterwarnings("ignore")

print("Loading data...")
df = pd.read_csv("data/csi1000_kline_raw.csv", parse_dates=["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

# 日内振幅 = (high - low) / prev_close
df["prev_close"] = df.groupby("stock_code")["close"].shift(1)
df["amplitude"] = (df["high"] - df["low"]) / df["prev_close"]

# 多窗口MA
for w in [5, 10, 20, 40, 60]:
    df[f"amp_ma{w}"] = df.groupby("stock_code")["amplitude"].transform(
        lambda x: x.rolling(w, min_periods=max(w//2, 3)).mean()
    )

# 多种比率组合 (取负, 即收缩为正)
CONFIGS = {
    "amp_comp_5_20": ("amp_ma5", "amp_ma20"),   # v1 baseline
    "amp_comp_5_40": ("amp_ma5", "amp_ma40"),
    "amp_comp_5_60": ("amp_ma5", "amp_ma60"),
    "amp_comp_10_40": ("amp_ma10", "amp_ma40"),
    "amp_comp_10_60": ("amp_ma10", "amp_ma60"),
}

for name, (short_col, long_col) in CONFIGS.items():
    df[name] = -np.log(df[short_col] / df[long_col].replace(0, np.nan))

# 成交额
df["log_amount"] = np.log1p(df["amount"])
df["log_amount_20d"] = df.groupby("stock_code")["log_amount"].transform(
    lambda x: x.rolling(20, min_periods=15).mean()
)

def neutralize_cs(group, col):
    y = group[col]
    x = group["log_amount_20d"]
    mask = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 30:
        return pd.Series(np.nan, index=group.index)
    X = np.column_stack([np.ones(mask.sum()), x[mask].values])
    beta, _, _, _ = lstsq(X, y[mask].values, rcond=None)
    resid = pd.Series(np.nan, index=group.index)
    resid[mask] = y[mask].values - X @ beta
    return resid

def mad_win(group, col, n_mad=5):
    vals = group[col]
    median = vals.median()
    mad = (vals - median).abs().median()
    if mad < 1e-10:
        return vals
    lower = median - n_mad * 1.4826 * mad
    upper = median + n_mad * 1.4826 * mad
    return vals.clip(lower, upper)

def zsc(group, col):
    vals = group[col]
    mu = vals.mean()
    std = vals.std()
    if std < 1e-10:
        return pd.Series(0.0, index=group.index)
    return (vals - mu) / std

for name in CONFIGS:
    print(f"\nProcessing {name}...")
    df["f_raw"] = df.groupby("date", group_keys=False).apply(lambda g, c=name: neutralize_cs(g, c))
    df["f_win"] = df.groupby("date", group_keys=False).apply(lambda g: mad_win(g, "f_raw"))
    df["f_z"] = df.groupby("date", group_keys=False).apply(lambda g: zsc(g, "f_win"))
    
    out = df[["date", "stock_code", "f_z"]].dropna(subset=["f_z"])
    out = out.rename(columns={"f_z": "factor_value"})
    out.to_csv(f"data/factor_{name}_v2.csv", index=False)
    print(f"  Output: data/factor_{name}_v2.csv ({len(out):,} records)")

print("\nDone!")
