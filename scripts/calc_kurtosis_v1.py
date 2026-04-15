#!/usr/bin/env python3
"""
因子: 收益率峰度 (Return Kurtosis) v1

定义: 过去40日日收益率的超额峰度(excess kurtosis), 成交额中性化.
      高峰度 = 胖尾分布 = 极端收益频繁 = 信息事件/风险集中
      低峰度 = 薄尾/均匀分布 = 收益波动平缓

假说A(正向): 高峰度=风险补偿(类似idio_vol/amp_level逻辑)
假说B(反向): 高峰度=彩票效应=被高估→后续收益低

中性化: 成交额OLS中性化 + MAD winsorize + z-score

另外测试: 下尾峰度(仅用负收益算峰度), 上尾峰度, 以及峰度变化(近期vs长期)
"""

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy.stats import kurtosis as sp_kurtosis
import warnings
warnings.filterwarnings("ignore")

print("Loading data...")
df = pd.read_csv("data/csi1000_kline_raw.csv", parse_dates=["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

df["ret"] = df.groupby("stock_code")["close"].pct_change()

# ── 滚动峰度 ───────────────────────────────────────────
WINDOW = 40
print(f"Computing {WINDOW}d rolling kurtosis...")

def rolling_kurtosis(group):
    g = group.sort_values("date")
    ret = g["ret"].values
    n = len(ret)
    kurt = np.full(n, np.nan)
    
    for i in range(WINDOW - 1, n):
        window = ret[i - WINDOW + 1: i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 20:
            continue
        # excess kurtosis (Fisher=True)
        mu = valid.mean()
        std = valid.std(ddof=1)
        if std < 1e-10:
            continue
        m4 = ((valid - mu) ** 4).mean()
        kurt[i] = m4 / (std ** 4) - 3.0
    
    return pd.Series(kurt, index=g.index)

kurt_vals = df.groupby("stock_code", group_keys=False).apply(rolling_kurtosis)
df["kurtosis_40d"] = kurt_vals.values

# ── 成交额中性化 ───────────────────────────────────────
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
df["f_raw"] = df.groupby("date", group_keys=False).apply(lambda g: neutralize_cs(g, "kurtosis_40d"))
print("MAD winsorize...")
df["f_win"] = df.groupby("date", group_keys=False).apply(lambda g: mad_win(g, "f_raw"))
print("Z-score...")
df["f_z"] = df.groupby("date", group_keys=False).apply(lambda g: zsc(g, "f_win"))

out = df[["date", "stock_code", "f_z"]].dropna(subset=["f_z"])
out = out.rename(columns={"f_z": "factor_value"})
out.to_csv("data/factor_kurtosis_v1.csv", index=False)
print(f"Output: data/factor_kurtosis_v1.csv ({len(out):,} records)")
print(f"  Date range: {out['date'].min()} ~ {out['date'].max()}")

print("\nDone!")
