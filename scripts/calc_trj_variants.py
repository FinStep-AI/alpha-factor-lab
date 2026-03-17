#!/usr/bin/env python3
"""
成交额排名跃迁因子 - 窗口变体测试
测试不同短期窗口(3d, 5d, 10d)与长期窗口(20d, 40d)的组合
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/csi1000_kline_raw.csv"

print("读取数据...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
df = df[df["date"] <= "2026-03-07"]

# 截面排名
df["amt_rank"] = df.groupby("date")["amount"].rank(pct=True)
df = df.sort_values(["stock_code", "date"])

# 多种窗口
for short_w in [3, 5, 10]:
    for long_w in [20, 40]:
        col = f"jump_{short_w}_{long_w}"
        df[f"rank_{short_w}d"] = df.groupby("stock_code")["amt_rank"].transform(
            lambda x: x.rolling(short_w, min_periods=max(2,short_w-1)).mean()
        )
        df[f"rank_{long_w}d"] = df.groupby("stock_code")["amt_rank"].transform(
            lambda x: x.rolling(long_w, min_periods=int(long_w*0.75)).mean()
        )
        df[col] = -(df[f"rank_{short_w}d"] - df[f"rank_{long_w}d"])  # 反向：排名下降=正因子

# 中性化
df["log_amount_20d"] = df.groupby("stock_code")["amount"].transform(
    lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
)

from numpy.linalg import lstsq

def neutralize(group, col):
    g = group.copy()
    f = g[col].values
    amt = g["log_amount_20d"].values
    valid = ~(np.isnan(f) | np.isnan(amt))
    if valid.sum() < 30:
        return pd.Series(np.nan, index=g.index)
    X = np.column_stack([np.ones(valid.sum()), amt[valid]])
    y = f[valid]
    try:
        beta, _, _, _ = lstsq(X, y, rcond=None)
        residual = np.full(len(f), np.nan)
        residual[valid] = y - X @ beta
    except:
        return pd.Series(np.nan, index=g.index)
    med = np.nanmedian(residual)
    mad = np.nanmedian(np.abs(residual - med))
    if mad < 1e-10:
        return pd.Series(np.nan, index=g.index)
    upper = med + 5 * 1.4826 * mad
    lower = med - 5 * 1.4826 * mad
    residual = np.clip(residual, lower, upper)
    mu = np.nanmean(residual)
    std = np.nanstd(residual)
    if std < 1e-10:
        return pd.Series(np.nan, index=g.index)
    return pd.Series((residual - mu) / std, index=g.index)

configs = [(3,20), (3,40), (5,20), (5,40), (10,20), (10,40)]
for short_w, long_w in configs:
    col = f"jump_{short_w}_{long_w}"
    print(f"处理 {col}...")
    df[f"{col}_factor"] = df.groupby("date", group_keys=False).apply(
        lambda g: neutralize(g, col)
    ).values
    out = df[["date", "stock_code", f"{col}_factor"]].dropna()
    out.columns = ["date", "stock_code", "factor_value"]
    path = f"data/factor_trj_{short_w}_{long_w}.csv"
    out.to_csv(path, index=False)
    print(f"  -> {path}: {len(out)}行")

print("完成!")
