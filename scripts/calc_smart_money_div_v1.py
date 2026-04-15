#!/usr/bin/env python3
"""
聪明钱分歧因子 v1 (Smart Money Divergence)

公式: mean(ret | 换手率排名前50%, 20d) - mean(ret | 换手率排名后50%, 20d)
含义: 高活跃日平均收益 - 低活跃日平均收益
  - 高值 = 活跃交易日涨、安静日跌 → 知情资金推升 → 动量延续
  - 低值 = 活跃日跌、安静日涨 → 散户追涨被套、知情资金撤退

中性化: 成交额OLS中性化 + MAD winsorize + z-score
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

print("读取K线数据...")
df = pd.read_csv("data/csi1000_kline_raw.csv", parse_dates=["date"])

# 收益率
df = df.sort_values(["stock_code", "date"])
if "pct_change" not in df.columns or df["pct_change"].isna().all():
    df["pct_change"] = df.groupby("stock_code")["close"].pct_change() * 100

# log_amount for neutralization
df["log_amount"] = np.log1p(df["amount"])
df["log_amount_20d"] = df.groupby("stock_code")["log_amount"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

print("计算聪明钱分歧因子...")

WINDOW = 20

def calc_smart_money_div(group):
    group = group.sort_values("date")
    ret = group["pct_change"].values
    turnover = group["turnover"].values
    n = len(ret)
    
    factor_vals = np.full(n, np.nan)
    
    for i in range(WINDOW - 1, n):
        w_ret = ret[i - WINDOW + 1: i + 1]
        w_turn = turnover[i - WINDOW + 1: i + 1]
        
        # 去掉NaN
        valid_mask = ~(np.isnan(w_ret) | np.isnan(w_turn))
        if valid_mask.sum() < 10:
            continue
            
        vr = w_ret[valid_mask]
        vt = w_turn[valid_mask]
        
        # 按换手率中位数分两组
        median_turn = np.median(vt)
        high_mask = vt >= median_turn
        low_mask = vt < median_turn
        
        if high_mask.sum() >= 3 and low_mask.sum() >= 3:
            high_ret = np.mean(vr[high_mask])
            low_ret = np.mean(vr[low_mask])
            factor_vals[i] = high_ret - low_ret
    
    group["raw_factor"] = factor_vals
    return group

df = df.groupby("stock_code", group_keys=False).apply(calc_smart_money_div)

print(f"  原始因子非空: {df['raw_factor'].notna().sum()}")
print(f"  原始因子统计:\n{df['raw_factor'].describe()}")

# ── 截面处理 ──
print("截面处理...")

def process_cross_section(group):
    factor = group["raw_factor"].copy()
    amount = group["log_amount_20d"].copy()
    
    mask = factor.notna() & amount.notna()
    if mask.sum() < 30:
        group["factor"] = np.nan
        return group
    
    f = factor[mask].values.astype(float)
    a = amount[mask].values.astype(float)
    
    # MAD winsorize
    med = np.median(f)
    mad = np.median(np.abs(f - med))
    if mad < 1e-10:
        mad = np.std(f)
    upper = med + 5 * 1.4826 * mad
    lower = med - 5 * 1.4826 * mad
    f = np.clip(f, lower, upper)
    
    # OLS中性化
    X = np.column_stack([np.ones(len(a)), a])
    try:
        beta = np.linalg.lstsq(X, f, rcond=None)[0]
        residual = f - X @ beta
    except:
        residual = f
    
    # z-score
    mu = np.mean(residual)
    sigma = np.std(residual)
    if sigma > 1e-10:
        z = (residual - mu) / sigma
    else:
        z = np.zeros_like(residual)
    
    result = np.full(len(factor), np.nan)
    result[mask.values] = z
    group["factor"] = result
    return group

df = df.groupby("date", group_keys=False).apply(process_cross_section)

output = df[["date", "stock_code", "factor"]].dropna(subset=["factor"])
output = output.rename(columns={"factor": "factor_value"})
output["date"] = output["date"].dt.strftime("%Y-%m-%d")
output = output.sort_values(["date", "stock_code"])

output.to_csv("data/factor_smart_money_div_v1.csv", index=False)
print(f"输出: data/factor_smart_money_div_v1.csv ({len(output)} rows)")
print(f"日期范围: {output['date'].min()} ~ {output['date'].max()}")
