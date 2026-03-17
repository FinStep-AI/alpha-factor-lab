#!/usr/bin/env python3
"""
成交额排名跃迁因子 (Turnover Rank Jump)

逻辑：
  1. 每天计算每只股票成交额在截面中的百分位排名 (0~1)
  2. 近5日均排名 vs 近20日均排名的差值
  3. 正值 = 近期成交额排名跃升(被市场关注) → 做多(信息驱动)
     或反向(均值回复,热度退潮后下跌) → 看回测结果
  4. 成交额OLS中性化 + MAD + z-score

与turnover_decay不同：
  - turnover_decay是绝对换手率的短/长比
  - 本因子是相对排名的变化(截面排名跃迁)
  - 排名变化更稳健,不受个股绝对换手率水平影响
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/csi1000_kline_raw.csv"
OUTPUT_PATH = "data/factor_turnover_rank_jump_v1.csv"

print("读取数据...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
df = df[df["date"] <= "2026-03-07"]

# 计算每日成交额截面百分位排名
print("计算截面排名...")
df["amt_rank"] = df.groupby("date")["amount"].rank(pct=True)

# 计算5日均排名和20日均排名
print("计算排名跃迁...")
df = df.sort_values(["stock_code", "date"])
df["rank_5d"] = df.groupby("stock_code")["amt_rank"].transform(
    lambda x: x.rolling(5, min_periods=3).mean()
)
df["rank_20d"] = df.groupby("stock_code")["amt_rank"].transform(
    lambda x: x.rolling(20, min_periods=15).mean()
)

# 排名跃迁 = 5d均排名 - 20d均排名
df["rank_jump"] = df["rank_5d"] - df["rank_20d"]
# 正值 = 排名上升(更多关注)

# 成交额中性化用20日均成交额
df["log_amount_20d"] = df.groupby("stock_code")["amount"].transform(
    lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
)

from numpy.linalg import lstsq

def cross_section_process(group):
    g = group.copy()
    f = g["rank_jump"].values
    amt = g["log_amount_20d"].values
    valid = ~(np.isnan(f) | np.isnan(amt))
    if valid.sum() < 30:
        g["factor"] = np.nan
        return g
    X = np.column_stack([np.ones(valid.sum()), amt[valid]])
    y = f[valid]
    try:
        beta, _, _, _ = lstsq(X, y, rcond=None)
        residual = np.full(len(f), np.nan)
        residual[valid] = y - X @ beta
    except:
        g["factor"] = np.nan
        return g
    med = np.nanmedian(residual)
    mad = np.nanmedian(np.abs(residual - med))
    if mad < 1e-10:
        g["factor"] = np.nan
        return g
    upper = med + 5 * 1.4826 * mad
    lower = med - 5 * 1.4826 * mad
    residual = np.clip(residual, lower, upper)
    mu = np.nanmean(residual)
    std = np.nanstd(residual)
    if std < 1e-10:
        g["factor"] = np.nan
        return g
    g["factor"] = (residual - mu) / std
    return g

print("横截面处理...")
result = df.groupby("date", group_keys=False).apply(cross_section_process)
result = result[["date", "stock_code", "factor"]].dropna(subset=["factor"])

# 同时输出反向版本
result_pos = result.copy()
result_pos.columns = ["date", "stock_code", "factor_value"]
result_pos.to_csv(OUTPUT_PATH, index=False)
print(f"正向输出: {OUTPUT_PATH}, {len(result_pos)}行")

result_neg = result.copy()
result_neg["factor"] = -result_neg["factor"]
result_neg.columns = ["date", "stock_code", "factor_value"]
result_neg.to_csv("data/factor_turnover_rank_jump_neg_v1.csv", index=False)
print(f"反向输出: data/factor_turnover_rank_jump_neg_v1.csv, {len(result_neg)}行")

print("完成!")
