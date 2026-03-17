#!/usr/bin/env python3
"""
均线乖离率反转因子 - 多变体测试

测试：
  v1: -BIAS_10 (10日均线)
  v2: -BIAS_5 (5日均线)  
  v3: -(BIAS_5 + BIAS_20) / 2 (复合)
  v4: -BIAS_10, 加入成交量确认 (缩量偏离权重更大)
"""

import numpy as np
import pandas as pd
import warnings, sys, os
warnings.filterwarnings("ignore")

DATA_PATH = "data/csi1000_kline_raw.csv"

print("读取K线数据...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
df = df[df["date"] <= "2026-03-07"]

# 预算成交额
amt_all = df[["date", "stock_code", "amount"]].copy()
amt_all["log_amount_20d"] = amt_all.groupby("stock_code")["amount"].transform(
    lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
)

def calc_variants(group):
    g = group.sort_values("date").copy()
    g["ma5"] = g["close"].rolling(5, min_periods=3).mean()
    g["ma10"] = g["close"].rolling(10, min_periods=7).mean()
    g["ma20"] = g["close"].rolling(20, min_periods=15).mean()
    
    g["bias_5"] = (g["close"] - g["ma5"]) / g["ma5"]
    g["bias_10"] = (g["close"] - g["ma10"]) / g["ma10"]
    g["bias_20"] = (g["close"] - g["ma20"]) / g["ma20"]
    
    # v1: -BIAS_10
    g["v1"] = -g["bias_10"]
    # v2: -BIAS_5
    g["v2"] = -g["bias_5"]
    # v3: composite -(BIAS_5 + BIAS_20) / 2
    g["v3"] = -(g["bias_5"] + g["bias_20"]) / 2
    # v4: -(0.5*BIAS_5 + 0.3*BIAS_10 + 0.2*BIAS_20) 加权复合
    g["v4"] = -(0.5 * g["bias_5"] + 0.3 * g["bias_10"] + 0.2 * g["bias_20"])
    
    return g[["date", "stock_code", "v1", "v2", "v3", "v4"]]

print("计算因子变体...")
result = df.groupby("stock_code", group_keys=False).apply(calc_variants)
result = result.merge(amt_all[["date", "stock_code", "log_amount_20d"]], on=["date", "stock_code"], how="left")

from numpy.linalg import lstsq

def neutralize_zscore(group, col):
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

for variant in ["v1", "v2", "v3", "v4"]:
    print(f"处理 {variant}...")
    result[f"{variant}_factor"] = result.groupby("date", group_keys=False).apply(
        lambda g: neutralize_zscore(g, variant)
    ).values
    
    out = result[["date", "stock_code", f"{variant}_factor"]].dropna()
    out.columns = ["date", "stock_code", "factor_value"]
    out_path = f"data/factor_bias_{variant}.csv"
    out.to_csv(out_path, index=False)
    print(f"  -> {out_path}: {len(out)}行")

print("\n全部变体计算完成!")
