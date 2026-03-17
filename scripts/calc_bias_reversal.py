#!/usr/bin/env python3
"""
均线乖离率反转因子 (BIAS Reversal Factor)

因子构造：
  - BIAS_20 = (close - MA20) / MA20  （20日均线乖离率）
  - 反向使用: factor = -BIAS_20  （做多负乖离=超卖股，做空正乖离=超买股）
  - 成交额OLS中性化 + MAD winsorize + z-score

逻辑：
  中证1000小盘股中，偏离20日均线过多的股票倾向均值回复。
  负乖离（价格低于均线）→ 超卖 → 后续反弹
  正乖离（价格高于均线）→ 超买 → 后续回调
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ---------- 配置 ----------
DATA_PATH = "data/csi1000_kline_raw.csv"
OUTPUT_PATH = "data/factor_bias_reversal_v1.csv"
MA_WINDOW = 20     # 均线窗口
MIN_OBS = 15       # 最少观测数

# ---------- 读取数据 ----------
print("读取K线数据...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

# 数据截断: 排除2026-03-09后的异常数据
df = df[df["date"] <= "2026-03-07"]
print(f"数据范围: {df['date'].min()} ~ {df['date'].max()}, 共{len(df)}行")

# ---------- 计算因子 ----------
print("计算均线乖离率...")

def calc_bias(group):
    """对单只股票计算BIAS"""
    g = group.sort_values("date").copy()
    # MA20
    g["ma20"] = g["close"].rolling(MA_WINDOW, min_periods=MIN_OBS).mean()
    # BIAS_20 = (close - MA20) / MA20
    g["bias_20"] = (g["close"] - g["ma20"]) / g["ma20"]
    return g[["date", "stock_code", "bias_20"]]

result = df.groupby("stock_code", group_keys=False).apply(calc_bias)
result = result.dropna(subset=["bias_20"])
print(f"有效因子值: {len(result)}行")

# ---------- 反向: factor = -BIAS_20 ----------
result["factor_raw"] = -result["bias_20"]

# ---------- 横截面处理 ----------
print("横截面标准化...")

# 需要成交额做中性化
amt_df = df[["date", "stock_code", "amount"]].copy()
amt_df["log_amount_20d"] = amt_df.groupby("stock_code")["amount"].transform(
    lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
)
result = result.merge(amt_df[["date", "stock_code", "log_amount_20d"]], on=["date", "stock_code"], how="left")

def cross_section_process(group):
    """横截面处理: 成交额OLS中性化 + MAD winsorize + z-score"""
    g = group.copy()
    f = g["factor_raw"].values
    amt = g["log_amount_20d"].values
    
    # 去掉NaN
    valid = ~(np.isnan(f) | np.isnan(amt))
    if valid.sum() < 30:
        g["factor"] = np.nan
        return g
    
    # OLS中性化: f = a + b*amt + residual
    from numpy.linalg import lstsq
    X = np.column_stack([np.ones(valid.sum()), amt[valid]])
    y = f[valid]
    try:
        beta, _, _, _ = lstsq(X, y, rcond=None)
        residual = np.full(len(f), np.nan)
        residual[valid] = y - X @ beta
    except:
        g["factor"] = np.nan
        return g
    
    # MAD winsorize
    med = np.nanmedian(residual)
    mad = np.nanmedian(np.abs(residual - med))
    if mad < 1e-10:
        g["factor"] = np.nan
        return g
    upper = med + 5 * 1.4826 * mad
    lower = med - 5 * 1.4826 * mad
    residual = np.clip(residual, lower, upper)
    
    # z-score
    mu = np.nanmean(residual)
    std = np.nanstd(residual)
    if std < 1e-10:
        g["factor"] = np.nan
        return g
    g["factor"] = (residual - mu) / std
    return g

result = result.groupby("date", group_keys=False).apply(cross_section_process)

# ---------- 输出 ----------
output = result[["date", "stock_code", "factor"]].dropna(subset=["factor"])
output.columns = ["date", "stock_code", "factor_value"]
output = output.sort_values(["date", "stock_code"])
output.to_csv(OUTPUT_PATH, index=False)
print(f"输出: {OUTPUT_PATH}, 共{len(output)}行")
print(f"因子日期范围: {output['date'].min()} ~ {output['date'].max()}")
print(f"每日平均股票数: {output.groupby('date')['stock_code'].count().mean():.0f}")

# 简单统计
print("\n因子分布:")
print(output["factor_value"].describe())
