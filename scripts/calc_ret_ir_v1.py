#!/usr/bin/env python3
"""
因子：收益率信息比率 (Return Information Ratio) v1
公式：mean(daily_ret, 20d) / std(daily_ret, 20d)，成交额OLS中性化 + MAD winsorize + z-score
逻辑：衡量近期收益率的信噪比。高IR=持续稳定上涨(动量+低波)，低IR=方向不明或剧烈波动。
      本质是风险调整后的短期动量，惩罚高波动的不确定方向。
Barra风格：Momentum (风险调整变体)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ---------- 参数 ----------
WINDOW = 20
MIN_OBS = 15  # 至少需要15天数据

# ---------- 读取数据 ----------
print("读取数据...")
df = pd.read_csv("data/csi1000_kline_raw.csv", parse_dates=["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

# 计算日收益率
df["ret"] = df["pct_change"] / 100.0  # pct_change 是百分比

# ---------- 计算因子 ----------
print("计算因子...")

def calc_ir(group):
    """计算滚动IR = mean(ret) / std(ret)"""
    ret = group["ret"]
    roll_mean = ret.rolling(WINDOW, min_periods=MIN_OBS).mean()
    roll_std = ret.rolling(WINDOW, min_periods=MIN_OBS).std()
    # 避免除以0
    ir = roll_mean / roll_std.replace(0, np.nan)
    return ir

df["raw_factor"] = df.groupby("stock_code", group_keys=False).apply(
    lambda g: calc_ir(g)
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
    """每个截面日的处理"""
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
    from numpy.linalg import lstsq
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
output.to_csv("data/factor_ret_ir_v1.csv", index=False)

n_dates = output["date"].nunique()
n_stocks = output["stock_code"].nunique()
print(f"完成！{n_dates}个交易日，{n_stocks}只股票，共{len(output)}行")
print(f"因子均值: {output['factor_value'].mean():.4f}, 标准差: {output['factor_value'].std():.4f}")
print(f"日期范围: {output['date'].min()} ~ {output['date'].max()}")
