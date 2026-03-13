#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最大日收益因子 (max_ret_v1)

逻辑（Bali, Cakici, Whitelaw 2011 "Maxing Out"）：
  过去20个交易日内最大单日收益率（MAX20）。
  
  文献表明：MAX高的股票被"彩票偏好"投资者追捧，
  导致定价偏高，未来收益率较低。
  
  反向因子：-MAX 作为因子值
  低MAX → 高因子值 → 做多（避开"彩票股"）

Barra风格: Volatility / 反转
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import warnings

warnings.filterwarnings("ignore")

N = 20

print("[1/4] 加载数据...")
kline = pd.read_csv("data/csi1000_kline_raw.csv")
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline['date'] = pd.to_datetime(kline['date'])
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)

# 计算日收益率
kline['ret'] = kline.groupby('stock_code')['close'].pct_change()

print("[2/4] 计算最大日收益因子...")

kline['max_ret'] = kline.groupby('stock_code')['ret'].transform(
    lambda x: x.rolling(N, min_periods=15).max()
)

factor_df = kline.dropna(subset=['max_ret']).copy()

# 反向：低MAX → 高因子值
factor_df['raw_factor'] = -factor_df['max_ret']
print(f"  有效记录: {len(factor_df)}")

print("[3/4] 市值中性化 + 标准化...")

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
    slope, intercept, _, _, _ = sp_stats.linregress(x[mask], y[mask])
    residuals = np.full(len(y), np.nan)
    residuals[mask] = y[mask] - (intercept + slope * x[mask])
    
    res_series = pd.Series(residuals[mask])
    median = res_series.median()
    mad = (res_series - median).abs().median()
    if mad > 0:
        lower = median - 5 * mad * 1.4826
        upper = median + 5 * mad * 1.4826
        residuals[mask] = np.clip(residuals[mask], lower, upper)
    
    valid = residuals[mask]
    mean, std = np.nanmean(valid), np.nanstd(valid)
    if std > 0:
        residuals[mask] = (residuals[mask] - mean) / std
    group['factor_value'] = residuals
    return group

factor_df = factor_df.groupby('date', group_keys=False).apply(neutralize_and_normalize)
factor_df = factor_df.dropna(subset=['factor_value'])

print("[4/4] 输出...")
output = factor_df[['date', 'stock_code', 'factor_value']].copy()
output['date'] = output['date'].dt.strftime('%Y-%m-%d')
output = output.sort_values(['date', 'stock_code'])
output.to_csv('data/factor_max_ret_v1.csv', index=False)
print(f"  输出: data/factor_max_ret_v1.csv ({len(output)} 行)")
print(f"  日期范围: {output['date'].min()} ~ {output['date'].max()}")
print(f"  每日平均: {output.groupby('date').size().mean():.0f} 股")
print("完成!")
