#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BPS/Price动量因子 (bp_momentum_v1)

逻辑：
  BP比率(Book-to-Price) = BPS / 收盘价
  BP Momentum = BP_t / BP_{t-60d的BP} - 1
  
  即BP比率的60日变化率。BP上升可能因为：
  1. BPS上升（公司增值）→ 好信号
  2. 股价下跌（市场低估）→ 价值回归信号
  
  两种情况都是正面的alpha信号。
  
  这是一个融合基本面(BPS)和价格(close)的复合因子。

Barra风格: Value/Growth
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import warnings

warnings.filterwarnings("ignore")

N = 60  # 回看窗口

print("[1/4] 加载数据...")
fund = pd.read_csv("data/csi1000_fundamental_cache.csv")
kline = pd.read_csv("data/csi1000_kline_raw.csv")

fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline['date'] = pd.to_datetime(kline['date'])
fund['report_date'] = pd.to_datetime(fund['report_date'])
fund = fund.sort_values(['stock_code', 'report_date'])

print("[2/4] 计算BP比率的动量...")

# 将BPS映射到每日（用merge_asof，45天延迟）
fund['avail_date'] = fund['report_date'] + pd.Timedelta(days=45)

stocks = sorted(kline['stock_code'].unique())
daily_bps_list = []

for stock in stocks:
    stock_fund = fund[fund['stock_code'] == stock].sort_values('avail_date')
    stock_kline = kline[kline['stock_code'] == stock][['date']].drop_duplicates().sort_values('date')
    if len(stock_fund) == 0:
        continue
    right = stock_fund[['avail_date', 'bps']].rename(columns={'avail_date': 'date'}).sort_values('date')
    merged = pd.merge_asof(stock_kline, right, on='date', direction='backward')
    merged['stock_code'] = stock
    daily_bps_list.append(merged)

daily_bps = pd.concat(daily_bps_list, ignore_index=True)
daily_bps = daily_bps.dropna(subset=['bps'])

# 合并价格
factor_df = daily_bps.merge(
    kline[['date', 'stock_code', 'close', 'amount']],
    on=['date', 'stock_code'],
    how='left'
)

# 计算BP ratio
factor_df['bp'] = factor_df['bps'] / factor_df['close']
factor_df = factor_df.dropna(subset=['bp'])
factor_df = factor_df[factor_df['bp'] > 0]

# BP momentum (60日变化率)
factor_df = factor_df.sort_values(['stock_code', 'date'])
factor_df['bp_lag'] = factor_df.groupby('stock_code')['bp'].shift(N)
factor_df['bp_mom'] = (factor_df['bp'] / factor_df['bp_lag']) - 1
factor_df = factor_df.dropna(subset=['bp_mom'])

# 去极端值
factor_df['raw_factor'] = factor_df['bp_mom']
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
output.to_csv('data/factor_bp_momentum_v1.csv', index=False)

# 反向
output_neg = output.copy()
output_neg['factor_value'] = -output_neg['factor_value']
output_neg.to_csv('data/factor_bp_momentum_neg_v1.csv', index=False)

print(f"  输出: data/factor_bp_momentum_v1.csv ({len(output)} 行)")
print(f"  日期范围: {output['date'].min()} ~ {output['date'].max()}")
print(f"  每日平均: {output.groupby('date').size().mean():.0f} 股")
print("完成!")
