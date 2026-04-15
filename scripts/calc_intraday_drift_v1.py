#!/usr/bin/env python3
"""
因子: 日内累积漂移方向性 (Intraday Cumulative Drift Direction) v1

定义: sum(close - open, 20d) / sum(|close - open|, 20d)
      取值范围[-1, +1]
      +1 = 连续20天阳线(收盘>开盘)
      -1 = 连续20天阴线
       0 = 方向混乱

假说: 在A股中证1000上, 日内方向持续一致 = 信息持续释放
      做多正值(连续阳线=持续买入) or 做多负值(连续阴线后反弹)?
      先正向测试, 数据说话.

与已有因子区别:
  - gap_momentum: 看集合竞价跳空方向一致性
  - shadow_pressure: 看上下影线
  - overnight_momentum: 看隔夜收益
  本因子看的是日内open→close的方向一致性

中性化: 成交额OLS中性化 + MAD winsorize + z-score
"""

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
import warnings
warnings.filterwarnings("ignore")

print("Loading data...")
df = pd.read_csv("data/csi1000_kline_raw.csv", parse_dates=["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

# 日内方向
df["oc_diff"] = df["close"] - df["open"]
df["oc_abs"] = df["oc_diff"].abs()

WINDOW = 20
print(f"Computing {WINDOW}d intraday drift direction...")

df["oc_sum"] = df.groupby("stock_code")["oc_diff"].transform(
    lambda x: x.rolling(WINDOW, min_periods=15).sum()
)
df["oc_abs_sum"] = df.groupby("stock_code")["oc_abs"].transform(
    lambda x: x.rolling(WINDOW, min_periods=15).sum()
)
df["drift_dir"] = df["oc_sum"] / df["oc_abs_sum"].replace(0, np.nan)

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

print("Neutralizing...")
df["f_raw"] = df.groupby("date", group_keys=False).apply(lambda g: neutralize_cs(g, "drift_dir"))
print("MAD winsorize...")
df["f_win"] = df.groupby("date", group_keys=False).apply(lambda g: mad_win(g, "f_raw"))
print("Z-score...")
df["f_z"] = df.groupby("date", group_keys=False).apply(lambda g: zsc(g, "f_win"))

out = df[["date", "stock_code", "f_z"]].dropna(subset=["f_z"])
out = out.rename(columns={"f_z": "factor_value"})
out.to_csv("data/factor_intraday_drift_v1.csv", index=False)
print(f"Output: data/factor_intraday_drift_v1.csv ({len(out):,} records)")
print(f"  Date range: {out['date'].min()} ~ {out['date'].max()}")

print("\nDone!")
