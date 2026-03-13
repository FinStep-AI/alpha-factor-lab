#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROE同比变化因子 (roe_delta_yoy_v1)

逻辑：
  对每只股票，计算最新季报ROE与去年同期ROE的差值。
  ROE改善(delta>0)的股票应跑赢ROE恶化(delta<0)的股票。
  
  为避免极端值影响，对delta做winsorize后z-score标准化。
  做市值中性化（用成交额20日均值代理市值）。

Barra风格: Growth
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import warnings

warnings.filterwarnings("ignore")

# === 1. 加载数据 ===
print("[1/5] 加载数据...")
fund = pd.read_csv("data/csi1000_fundamental_cache.csv")
kline = pd.read_csv("data/csi1000_kline_raw.csv")

fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline['date'] = pd.to_datetime(kline['date'])
fund['report_date'] = pd.to_datetime(fund['report_date'])

fund = fund.sort_values(['stock_code', 'report_date']).reset_index(drop=True)

# === 2. 计算ROE同比变化 ===
print("[2/5] 计算ROE同比变化...")

# 提取季度标识
fund['quarter'] = fund['report_date'].dt.month  # 3,6,9,12

# 对每个stock，找同季度的去年数据
results = []
for stock, grp in fund.groupby('stock_code'):
    grp = grp.sort_values('report_date')
    for i, row in grp.iterrows():
        # 找同季度、前一年
        yoy_date = row['report_date'] - pd.DateOffset(years=1)
        # 允许+-15天的匹配
        prev = grp[(grp['report_date'] >= yoy_date - pd.Timedelta(days=15)) & 
                    (grp['report_date'] <= yoy_date + pd.Timedelta(days=15))]
        if len(prev) == 0:
            continue
        prev_roe = prev.iloc[0]['roe']
        curr_roe = row['roe']
        if pd.isna(prev_roe) or pd.isna(curr_roe):
            continue
        delta = curr_roe - prev_roe
        results.append({
            'stock_code': stock,
            'report_date': row['report_date'],
            'roe_delta': delta,
            'roe_current': curr_roe,
            'roe_prev': prev_roe
        })

delta_df = pd.DataFrame(results)
print(f"  计算完成: {len(delta_df)} 条ROE同比变化记录")

# === 3. 映射到日频 ===
print("[3/5] 映射到日频交易日...")

# 假设数据在report_date后45天可用（避免前视偏差）
delta_df['avail_date'] = delta_df['report_date'] + pd.Timedelta(days=45)

stocks = sorted(kline['stock_code'].unique())
daily_factors = []

for stock in stocks:
    stock_deltas = delta_df[delta_df['stock_code'] == stock].sort_values('avail_date')
    stock_dates = kline[kline['stock_code'] == stock][['date']].drop_duplicates().sort_values('date')
    
    if len(stock_deltas) == 0:
        continue
    
    right = stock_deltas[['avail_date', 'roe_delta']].rename(columns={'avail_date': 'date'}).sort_values('date')
    merged = pd.merge_asof(stock_dates, right, on='date', direction='backward')
    merged['stock_code'] = stock
    daily_factors.append(merged)

factor_df = pd.concat(daily_factors, ignore_index=True)
factor_df = factor_df.dropna(subset=['roe_delta'])
print(f"  日频因子值: {len(factor_df)} 条")

# === 4. 市值中性化 ===
print("[4/5] 市值中性化...")

kline_sorted = kline.sort_values(['stock_code', 'date'])
kline_sorted['amount_ma20'] = kline_sorted.groupby('stock_code')['amount'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

factor_df = factor_df.merge(
    kline_sorted[['date', 'stock_code', 'amount_ma20']],
    on=['date', 'stock_code'],
    how='left'
)

def neutralize_cross_section(group):
    y = group['roe_delta'].values
    x = np.log(group['amount_ma20'].values + 1)
    mask = np.isfinite(y) & np.isfinite(x) & (x > 0)
    if mask.sum() < 30:
        group['factor_value'] = np.nan
        return group
    slope, intercept, _, _, _ = sp_stats.linregress(x[mask], y[mask])
    residuals = np.full(len(y), np.nan)
    residuals[mask] = y[mask] - (intercept + slope * x[mask])
    group['factor_value'] = residuals
    return group

factor_df = factor_df.groupby('date', group_keys=False).apply(neutralize_cross_section)

# Winsorize (MAD) + Z-score
def winsorize_mad(s, n_mad=5):
    median = s.median()
    mad = (s - median).abs().median()
    if mad == 0:
        return s
    lower = median - n_mad * mad * 1.4826
    upper = median + n_mad * mad * 1.4826
    return s.clip(lower, upper)

def cross_section_normalize(group):
    vals = group['factor_value']
    vals = winsorize_mad(vals)
    mean = vals.mean()
    std = vals.std()
    if std == 0 or np.isnan(std):
        group['factor_value'] = 0.0
    else:
        group['factor_value'] = (vals - mean) / std
    return group

factor_df = factor_df.groupby('date', group_keys=False).apply(cross_section_normalize)
factor_df = factor_df.dropna(subset=['factor_value'])

# === 5. 输出 ===
print("[5/5] 输出因子CSV...")
output = factor_df[['date', 'stock_code', 'factor_value']].copy()
output['date'] = output['date'].dt.strftime('%Y-%m-%d')
output = output.sort_values(['date', 'stock_code'])
output.to_csv('data/factor_roe_delta_yoy_v1.csv', index=False)
print(f"  输出: data/factor_roe_delta_yoy_v1.csv ({len(output)} 行)")
print(f"  日期范围: {output['date'].min()} ~ {output['date'].max()}")
print(f"  每日平均股票数: {output.groupby('date').size().mean():.0f}")
print("完成!")
