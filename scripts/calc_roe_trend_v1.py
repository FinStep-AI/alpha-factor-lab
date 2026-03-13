#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROE趋势因子 (roe_trend_v1)

逻辑：
  对每只股票，在每个季报发布后，取最近4个季度的ROE，
  对季度序号做OLS回归，斜率即为ROE趋势。
  斜率>0 表示盈利改善，<0 表示盈利恶化。
  
  因子映射到日频：
  每当新季报发布后，更新因子值，直到下一个季报。
  
  市值中性化：用 amount (成交额) 的近期均值作为市值代理。

Barra风格: Growth / Quality
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import warnings, sys

warnings.filterwarnings("ignore")

# === 1. 加载数据 ===
print("[1/5] 加载数据...")
fund = pd.read_csv("data/csi1000_fundamental_cache.csv")
kline = pd.read_csv("data/csi1000_kline_raw.csv")

# stock_code 统一为6位字符串
fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline['date'] = pd.to_datetime(kline['date'])
fund['report_date'] = pd.to_datetime(fund['report_date'])

# 排序
fund = fund.sort_values(['stock_code', 'report_date']).reset_index(drop=True)

# === 2. 计算每个(stock, report_date)的ROE趋势斜率 ===
print("[2/5] 计算ROE趋势斜率（滚动4季度回归）...")

MIN_QUARTERS = 3  # 至少需要3个季度数据

def calc_roe_slope(group):
    """对单只股票的所有季报，滚动计算ROE趋势斜率"""
    group = group.sort_values('report_date')
    results = []
    for i in range(len(group)):
        # 取最近4个季度（含当前）
        start = max(0, i - 3)
        window = group.iloc[start:i+1]
        if len(window) < MIN_QUARTERS:
            results.append({'report_date': group.iloc[i]['report_date'], 'roe_slope': np.nan})
            continue
        # OLS: ROE ~ quarter_index
        y = window['roe'].values
        x = np.arange(len(y))
        # 跳过全NaN
        mask = ~np.isnan(y)
        if mask.sum() < MIN_QUARTERS:
            results.append({'report_date': group.iloc[i]['report_date'], 'roe_slope': np.nan})
            continue
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x[mask], y[mask])
        results.append({'report_date': group.iloc[i]['report_date'], 'roe_slope': slope})
    return pd.DataFrame(results)

slopes_list = []
for stock, grp in fund.groupby('stock_code'):
    res = calc_roe_slope(grp)
    res['stock_code'] = stock
    slopes_list.append(res)

slopes_df = pd.concat(slopes_list, ignore_index=True)
slopes_df = slopes_df.dropna(subset=['roe_slope'])
print(f"  计算完成: {len(slopes_df)} 条ROE斜率记录")

# === 3. 映射到日频 ===
print("[3/5] 映射到日频交易日...")

# 获取所有交易日
trade_dates = sorted(kline['date'].unique())
stocks = sorted(kline['stock_code'].unique())

# 对每个季报日期，估计其实际发布日（季报+1.5个月）
# 实际上用report_date直接作为信号日期（保守假设：数据在report_date后可用）
# 为避免前视偏差，我们假设数据在report_date后45天可用
slopes_df['report_date'] = pd.to_datetime(slopes_df['report_date'])
slopes_df['avail_date'] = slopes_df['report_date'] + pd.Timedelta(days=45)

# 对每只股票、每个交易日，取最近可用的roe_slope
daily_factors = []

for stock in stocks:
    stock_slopes = slopes_df[slopes_df['stock_code'] == stock].sort_values('avail_date')
    stock_dates = kline[kline['stock_code'] == stock][['date']].drop_duplicates().sort_values('date')
    
    if len(stock_slopes) == 0:
        continue
    
    # Merge_asof: 对每个交易日找最近的avail_date <= date
    stock_dates = stock_dates.copy()
    right = stock_slopes[['avail_date', 'roe_slope']].rename(columns={'avail_date': 'date'}).sort_values('date')
    merged = pd.merge_asof(
        stock_dates, 
        right,
        on='date',
        direction='backward'
    )
    merged['stock_code'] = stock
    daily_factors.append(merged)

factor_df = pd.concat(daily_factors, ignore_index=True)
factor_df = factor_df.dropna(subset=['roe_slope'])
print(f"  日频因子值: {len(factor_df)} 条")

# === 4. 市值中性化 ===
print("[4/5] 市值中性化...")

# 用成交额20日均值作为市值代理
kline_sorted = kline.sort_values(['stock_code', 'date'])
kline_sorted['amount_ma20'] = kline_sorted.groupby('stock_code')['amount'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

# 合并
factor_df = factor_df.merge(
    kline_sorted[['date', 'stock_code', 'amount_ma20']],
    on=['date', 'stock_code'],
    how='left'
)

# 截面中性化: 每日对因子值做 roe_slope ~ log(amount_ma20) 回归，取残差
def neutralize_cross_section(group):
    y = group['roe_slope'].values
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

# Winsorize (MAD方法) + Z-score标准化
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
output.to_csv('data/factor_roe_trend_v1.csv', index=False)
print(f"  输出: data/factor_roe_trend_v1.csv ({len(output)} 行)")
print(f"  日期范围: {output['date'].min()} ~ {output['date'].max()}")
print(f"  每日平均股票数: {output.groupby('date').size().mean():.0f}")
print("完成!")
