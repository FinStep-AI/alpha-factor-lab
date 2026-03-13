#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
价格效率因子 (price_efficiency_v1)

逻辑：
  衡量股票价格运动的"效率"——净位移与总路径之比。
  
  定义：
  - 过去N日的净价格变化（收盘价位移）：|close_t - close_{t-N}|
  - 过去N日的总价格路径（逐日绝对变化之和）：sum(|close_i - close_{i-1}|)
  - 价格效率 = 净位移 / 总路径
  
  效率高(接近1)：价格单方向运动，趋势明确
  效率低(接近0)：价格来回震荡，方向不明
  
  文献表明：低价格效率（高噪音）的股票往往有更高的alpha，
  可能是因为这些股票被关注较少、信息传播慢。
  
  反向使用：低效率 → 做多（反转/信息不对称逻辑）
  
  参数：N=20日

Barra风格: Quality（信息质量/价格发现效率）
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import warnings

warnings.filterwarnings("ignore")

# === 参数 ===
N = 20  # 回看窗口

# === 1. 加载数据 ===
print("[1/4] 加载数据...")
kline = pd.read_csv("data/csi1000_kline_raw.csv")
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline['date'] = pd.to_datetime(kline['date'])
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)

# === 2. 计算价格效率因子 ===
print("[2/4] 计算价格效率因子...")

def calc_price_efficiency(group):
    group = group.sort_values('date')
    close = group['close'].values
    dates = group['date'].values
    
    # 逐日绝对变化
    daily_abs_change = np.abs(np.diff(close))
    
    # 滚动计算
    n = len(close)
    efficiency = np.full(n, np.nan)
    
    for i in range(N, n):
        net_displacement = abs(close[i] - close[i - N])
        total_path = np.sum(daily_abs_change[i-N:i])
        if total_path > 0:
            efficiency[i] = net_displacement / total_path
        else:
            efficiency[i] = np.nan
    
    group['raw_efficiency'] = efficiency
    return group

results = []
stocks = kline['stock_code'].unique()
for i, stock in enumerate(stocks):
    grp = kline[kline['stock_code'] == stock].copy()
    res = calc_price_efficiency(grp)
    results.append(res)
    if (i + 1) % 200 == 0:
        print(f"  处理: {i+1}/{len(stocks)}")

factor_df = pd.concat(results, ignore_index=True)
factor_df = factor_df.dropna(subset=['raw_efficiency'])

# 反向使用：低效率 → 高因子值 → 做多
factor_df['raw_factor'] = -factor_df['raw_efficiency']

print(f"  有效记录: {len(factor_df)}")

# === 3. 市值中性化 + 标准化 ===
print("[3/4] 市值中性化 + 标准化...")

# 成交额20日均值作为市值代理
factor_df['amount_ma20'] = factor_df.groupby('stock_code')['amount'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

def neutralize_and_normalize(group):
    y = group['raw_factor'].values
    x = np.log(group['amount_ma20'].values + 1)
    mask = np.isfinite(y) & np.isfinite(x) & (x > 0)
    if mask.sum() < 30:
        group['factor_value'] = np.nan
        return group
    
    # 市值中性化
    slope, intercept, _, _, _ = sp_stats.linregress(x[mask], y[mask])
    residuals = np.full(len(y), np.nan)
    residuals[mask] = y[mask] - (intercept + slope * x[mask])
    
    # Winsorize (MAD)
    res_series = pd.Series(residuals[mask])
    median = res_series.median()
    mad = (res_series - median).abs().median()
    if mad > 0:
        lower = median - 5 * mad * 1.4826
        upper = median + 5 * mad * 1.4826
        residuals[mask] = np.clip(residuals[mask], lower, upper)
    
    # Z-score
    valid = residuals[mask]
    mean = np.nanmean(valid)
    std = np.nanstd(valid)
    if std > 0:
        residuals[mask] = (residuals[mask] - mean) / std
    else:
        residuals[mask] = 0.0
    
    group['factor_value'] = residuals
    return group

factor_df = factor_df.groupby('date', group_keys=False).apply(neutralize_and_normalize)
factor_df = factor_df.dropna(subset=['factor_value'])

# === 4. 输出 ===
print("[4/4] 输出因子CSV...")
output = factor_df[['date', 'stock_code', 'factor_value']].copy()
output['date'] = output['date'].dt.strftime('%Y-%m-%d')
output = output.sort_values(['date', 'stock_code'])
output.to_csv('data/factor_price_efficiency_v1.csv', index=False)
print(f"  输出: data/factor_price_efficiency_v1.csv ({len(output)} 行)")
print(f"  日期范围: {output['date'].min()} ~ {output['date'].max()}")
print(f"  每日平均股票数: {output.groupby('date').size().mean():.0f}")
print("完成!")
