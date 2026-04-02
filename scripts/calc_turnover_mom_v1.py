#!/usr/bin/env python3
"""
因子：换手率动量 (Turnover Momentum) v1
公式：log(MA5_turnover / MA20_turnover)，成交额OLS中性化 + MAD winsorize + z-score
逻辑：短期换手率相对长期的加速度。高=最近5日换手率相对20日均值显著放大=新资金涌入/关注度飙升。
      和turnover_level(水平)、vol_cv_neg(稳定性)正交的第三个维度：变化方向。
Barra风格：Liquidity/Sentiment
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from numpy.linalg import lstsq

# ---------- 参数 ----------
SHORT_WINDOW = 5
LONG_WINDOW = 20

# ---------- 读取数据 ----------
print("读取数据...")
df = pd.read_csv("data/csi1000_kline_raw.csv", parse_dates=["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

# ---------- 计算因子 ----------
print("计算因子...")

def calc_turnover_momentum(group):
    t = group["turnover"]
    ma_short = t.rolling(SHORT_WINDOW, min_periods=3).mean()
    ma_long = t.rolling(LONG_WINDOW, min_periods=10).mean()
    # 避免log(0)
    ratio = ma_short / ma_long.replace(0, np.nan)
    ratio = ratio.clip(lower=0.01)  # 避免log负值
    return np.log(ratio)

df["raw_factor"] = df.groupby("stock_code", group_keys=False).apply(
    lambda g: calc_turnover_momentum(g)
)

# ---------- 计算中性化变量：log_amount_20d ----------
print("计算中性化变量...")
df["log_amount"] = np.log(df["amount"].clip(lower=1))
df["log_amount_20d"] = df.groupby("stock_code")["log_amount"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

# ---------- 截面处理：MAD winsorize + 成交额OLS中性化 + z-score ----------
print("截面处理...")

def process_cross_section(sub):
    factor = sub["raw_factor"].copy()
    neutralizer = sub["log_amount_20d"]
    
    mask = factor.notna() & neutralizer.notna()
    if mask.sum() < 50:
        return pd.Series(np.nan, index=sub.index)
    
    f = factor[mask].copy()
    n = neutralizer[mask].copy()
    
    # MAD winsorize
    med = f.median()
    mad = (f - med).abs().median()
    if mad < 1e-10:
        mad = f.std()
    if mad < 1e-10:
        return pd.Series(np.nan, index=sub.index)
    upper = med + 5 * 1.4826 * mad
    lower = med - 5 * 1.4826 * mad
    f = f.clip(lower, upper)
    
    # OLS中性化
    X = np.column_stack([np.ones(len(n)), n.values])
    y = f.values
    coef, _, _, _ = lstsq(X, y, rcond=None)
    residual = y - X @ coef
    
    # z-score
    mu = residual.mean()
    sigma = residual.std()
    if sigma < 1e-10:
        return pd.Series(np.nan, index=sub.index)
    z = (residual - mu) / sigma
    
    result = pd.Series(np.nan, index=sub.index)
    result[mask] = z
    return result

factor_values = df.groupby("date", group_keys=False).apply(process_cross_section)
df["factor"] = factor_values

# ---------- 输出 ----------
print("输出因子文件...")
output = df[["date", "stock_code", "factor"]].dropna(subset=["factor"])
output = output.rename(columns={"factor": "factor_value"})
output.to_csv("data/factor_turnover_mom_v1.csv", index=False)

n_dates = output["date"].nunique()
n_stocks = output["stock_code"].nunique()
print(f"完成！{n_dates}个交易日，{n_stocks}只股票，共{len(output)}行")
print(f"因子均值: {output['factor_value'].mean():.4f}, 标准差: {output['factor_value'].std():.4f}")
print(f"日期范围: {output['date'].min()} ~ {output['date'].max()}")
