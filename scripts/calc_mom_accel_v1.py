#!/usr/bin/env python3
"""
因子: 动量加速度 (Momentum Acceleration) v1

定义: 20日累计收益 - 60日累计收益的日均收益 × 20
      即: sum(ret,20) - (sum(ret,60)/60)*20 = sum(ret,20) - sum(ret,60)/3
      简化: 近20天收益 - 过去60天平均20天收益
      
      更直观: MA5/MA60 - 1 的斜率

实际公式: sum(ret,5) - sum(ret,20)/4 = 短期动量 - 中期平均动量
          或者用5/20的组合更短期

最终选择: ret_5d - ret_20d/4 = 5日动量加速度
          高值 = 最近5天涨幅超出20天平均速度 = 正在加速上涨

中性化: 成交额OLS中性化

Barra: Momentum
"""

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
import warnings
warnings.filterwarnings("ignore")

print("Loading data...")
df = pd.read_csv("data/csi1000_kline_raw.csv", parse_dates=["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

# 计算日收益率
df["ret"] = df.groupby("stock_code")["close"].pct_change()

# ── 多组窗口的动量加速度 ─────────────────────────────
# Variant 1: ret_5d - ret_20d/4 (5日加速度)
# Variant 2: ret_10d - ret_40d/4 (10日加速度)  
# Variant 3: ret_20d - ret_60d/3 (20日加速度)

CONFIGS = [
    ("mom_accel_5_20", 5, 20),
    ("mom_accel_10_40", 10, 40),
    ("mom_accel_20_60", 20, 60),
]

print("Computing momentum accelerations...")
for name, short, long in CONFIGS:
    df[f"ret_{short}d"] = df.groupby("stock_code")["ret"].transform(
        lambda x: x.rolling(short, min_periods=short).sum()
    )
    df[f"ret_{long}d"] = df.groupby("stock_code")["ret"].transform(
        lambda x: x.rolling(long, min_periods=long).sum()
    )
    ratio = long / short
    df[name] = df[f"ret_{short}d"] - df[f"ret_{long}d"] / ratio

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

# 处理最好的一个变体: 20_60 (最稳定)
TARGET = "mom_accel_20_60"
print(f"Processing {TARGET}...")
print("  Neutralizing...")
df["factor_raw"] = df.groupby("date", group_keys=False).apply(lambda g: neutralize_cs(g, TARGET))
print("  MAD winsorize...")
df["factor_win"] = df.groupby("date", group_keys=False).apply(lambda g: mad_win(g, "factor_raw"))
print("  Z-score...")
df["factor"] = df.groupby("date", group_keys=False).apply(lambda g: zsc(g, "factor_win"))

out = df[["date", "stock_code", "factor"]].dropna(subset=["factor"])
out = out.rename(columns={"factor": "factor_value"})
out.to_csv("data/factor_mom_accel_v1.csv", index=False)
print(f"Output: data/factor_mom_accel_v1.csv ({len(out):,} records)")
print(f"  Date range: {out['date'].min()} ~ {out['date'].max()}")

# 也输出 5_20 变体
for name, _, _ in CONFIGS:
    if name == TARGET:
        continue
    print(f"\nProcessing {name}...")
    df["factor_raw2"] = df.groupby("date", group_keys=False).apply(lambda g: neutralize_cs(g, name))
    df["factor_win2"] = df.groupby("date", group_keys=False).apply(lambda g: mad_win(g, "factor_raw2"))
    df["factor2"] = df.groupby("date", group_keys=False).apply(lambda g: zsc(g, "factor_win2"))
    out2 = df[["date", "stock_code", "factor2"]].dropna(subset=["factor2"])
    out2 = out2.rename(columns={"factor2": "factor_value"})
    out2.to_csv(f"data/factor_{name}_v1.csv", index=False)
    print(f"Output: data/factor_{name}_v1.csv ({len(out2):,} records)")

print("\nDone!")
