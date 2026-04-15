#!/usr/bin/env python3
"""
因子: 异常成交日频率 (Abnormal Volume Day Frequency) v1

定义: 过去20日中, 成交额 > MA20_amount × 2.0 的天数占比
      高值 = 异常放量日频繁 = 事件密集/信息释放活跃

变体2: 异常缩量日频率 = 成交额 < MA20_amount × 0.5 的天数占比
       高值 = 异常缩量日频繁 = 市场关注度降低/流动性枯竭

逻辑: 
  - 异常放量频率高 → 近期信息事件密集, 方向取决于放量时的价格方向
  - 与vol_cv_neg(成交量变异系数)不同: CV衡量波动性, 这个衡量极端事件频率
  - 与turnover_level(换手水平)不同: 这里看的是尾部频率而非均值

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

# ── 计算异常成交日 ──────────────────────────────────────
WINDOW = 20
THRESHOLD_HIGH = 2.0  # 超过均值2倍算异常放量
THRESHOLD_LOW = 0.5   # 低于均值0.5倍算异常缩量

print("Computing rolling stats...")
df["amount_ma20"] = df.groupby("stock_code")["amount"].transform(
    lambda x: x.rolling(60, min_periods=40).mean()  # 用更长窗口(60d)作为"正常水平"
)

df["is_high_vol"] = (df["amount"] > df["amount_ma20"] * THRESHOLD_HIGH).astype(float)
df["is_low_vol"] = (df["amount"] < df["amount_ma20"] * THRESHOLD_LOW).astype(float)

# 20日滚动频率
df["high_vol_freq"] = df.groupby("stock_code")["is_high_vol"].transform(
    lambda x: x.rolling(WINDOW, min_periods=15).mean()
)
df["low_vol_freq"] = df.groupby("stock_code")["is_low_vol"].transform(
    lambda x: x.rolling(WINDOW, min_periods=15).mean()
)

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

# 处理两个因子
for factor_name, col in [("abnorm_high_freq", "high_vol_freq"), 
                          ("abnorm_low_freq", "low_vol_freq")]:
    print(f"\nProcessing {factor_name}...")
    df["f_raw"] = df.groupby("date", group_keys=False).apply(lambda g, c=col: neutralize_cs(g, c))
    df["f_win"] = df.groupby("date", group_keys=False).apply(lambda g: mad_win(g, "f_raw"))
    df["f_z"] = df.groupby("date", group_keys=False).apply(lambda g: zsc(g, "f_win"))
    
    out = df[["date", "stock_code", "f_z"]].dropna(subset=["f_z"])
    out = out.rename(columns={"f_z": "factor_value"})
    out.to_csv(f"data/factor_{factor_name}_v1.csv", index=False)
    print(f"  Output: data/factor_{factor_name}_v1.csv ({len(out):,} records)")

print("\nDone!")
