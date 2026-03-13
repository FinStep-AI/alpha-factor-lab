#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日内动量因子 (intraday_mom_v1)

逻辑：
  衡量每日开盘到收盘的收益率相对于全天收益率的比例。
  
  intraday_return = (close - open) / open
  total_range = (high - low) / low
  
  因子 = rolling_mean(intraday_return / total_range, 20)
  
  当日内收益占振幅比例大，说明趋势性强；
  当日内收益占振幅比例小，说明有大量日内反转。
  
  强日内趋势 → 信息快速反映 → 机构主导
  弱日内趋势 → 散户博弈为主

Barra风格: 微观结构
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

print("[2/4] 计算日内动量因子...")

# 日内收益 / 振幅
kline['intraday_ret'] = (kline['close'] - kline['open']) / kline['open']
kline['range_pct'] = (kline['high'] - kline['low']) / kline['low']
# 避免除零
kline['intraday_ratio'] = np.where(
    kline['range_pct'] > 0.001,
    kline['intraday_ret'] / kline['range_pct'],
    np.nan
)

# 滚动均值
kline['raw_factor'] = kline.groupby('stock_code')['intraday_ratio'].transform(
    lambda x: x.rolling(N, min_periods=15).mean()
)

factor_df = kline.dropna(subset=['raw_factor']).copy()
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
output.to_csv('data/factor_intraday_mom_v1.csv', index=False)

# 也输出反向
output_neg = output.copy()
output_neg['factor_value'] = -output_neg['factor_value']
output_neg.to_csv('data/factor_intraday_mom_neg_v1.csv', index=False)

print(f"  输出: data/factor_intraday_mom_v1.csv ({len(output)} 行)")
print(f"  日期范围: {output['date'].min()} ~ {output['date'].max()}")
print(f"  每日平均: {output.groupby('date').size().mean():.0f} 股")
print("完成!")
