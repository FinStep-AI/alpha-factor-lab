#!/usr/bin/env python3
"""
因子: 成交额增长斜率 (Amount Trend Slope)
逻辑: 对过去20日成交额做OLS线性回归, 取标准化斜率(slope/mean_amount)
高斜率 = 成交额持续放大 = 资金持续流入 = 市场关注度上升
中性化: 成交额OLS中性化 + MAD winsorize + z-score
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys
import os

# 参数
LOOKBACK = 20
OUTPUT_PATH = "data/factor_amt_trend_slope_v1.csv"

def compute_slope(arr):
    """对arr做OLS回归, 返回标准化斜率"""
    n = len(arr)
    if n < LOOKBACK or np.all(np.isnan(arr)):
        return np.nan
    valid = ~np.isnan(arr)
    if valid.sum() < LOOKBACK * 0.7:
        return np.nan
    x = np.arange(n)
    # 用valid数据做回归
    a = arr[valid]
    t = x[valid]
    if len(a) < 5:
        return np.nan
    slope, intercept, r, p, se = stats.linregress(t, a)
    # 标准化: slope / mean(amount), 使不同量级的股票可比
    mean_val = np.mean(a)
    if mean_val <= 0:
        return np.nan
    return slope / mean_val

def mad_winsorize(s, n_mad=5):
    """MAD winsorize"""
    median = s.median()
    mad = (s - median).abs().median()
    if mad == 0:
        return s
    upper = median + n_mad * 1.4826 * mad
    lower = median - n_mad * 1.4826 * mad
    return s.clip(lower=lower, upper=upper)

def neutralize_ols(factor_series, neutralizer_series):
    """OLS中性化"""
    mask = factor_series.notna() & neutralizer_series.notna()
    if mask.sum() < 10:
        return factor_series
    from numpy.linalg import lstsq
    X = np.column_stack([np.ones(mask.sum()), neutralizer_series[mask].values])
    y = factor_series[mask].values
    beta, _, _, _ = lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    result = factor_series.copy()
    result[mask] = residuals
    return result

def main():
    print("Loading data...")
    kline = pd.read_csv("data/csi1000_kline_raw.csv")
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # 计算成交额(amount)的20日OLS斜率
    print(f"Computing {LOOKBACK}d amount trend slope...")
    
    all_results = []
    stocks = kline['stock_code'].unique()
    
    for i, stock in enumerate(stocks):
        if (i + 1) % 200 == 0:
            print(f"  Processing {i+1}/{len(stocks)} stocks...")
        
        stock_data = kline[kline['stock_code'] == stock].sort_values('date')
        amounts = stock_data['amount'].values
        dates = stock_data['date'].values
        codes = stock_data['stock_code'].values
        
        slopes = np.full(len(amounts), np.nan)
        
        for j in range(LOOKBACK - 1, len(amounts)):
            window = amounts[j - LOOKBACK + 1: j + 1]
            slopes[j] = compute_slope(window)
        
        for j in range(len(dates)):
            if not np.isnan(slopes[j]):
                all_results.append({
                    'date': dates[j],
                    'stock_code': codes[j],
                    'raw_factor': slopes[j]
                })
    
    df = pd.DataFrame(all_results)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Raw factor computed: {len(df)} rows")
    
    # 合并成交额用于中性化
    amt_20d = kline.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )
    kline['log_amount_20d'] = np.log(amt_20d.clip(lower=1))
    
    df = df.merge(
        kline[['date', 'stock_code', 'log_amount_20d']],
        on=['date', 'stock_code'],
        how='left'
    )
    
    # 截面处理: MAD winsorize -> OLS中性化 -> z-score
    print("Cross-sectional processing...")
    processed = []
    
    for date, group in df.groupby('date'):
        g = group.copy()
        
        # MAD winsorize
        g['factor'] = mad_winsorize(g['raw_factor'])
        
        # OLS neutralize by log_amount_20d
        g['factor'] = neutralize_ols(g['factor'], g['log_amount_20d'])
        
        # z-score
        mean = g['factor'].mean()
        std = g['factor'].std()
        if std > 0:
            g['factor'] = (g['factor'] - mean) / std
        else:
            g['factor'] = 0
        
        processed.append(g[['date', 'stock_code', 'factor']])
    
    result = pd.concat(processed, ignore_index=True)
    result = result.sort_values(['date', 'stock_code']).reset_index(drop=True)
    
    # 保存
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}: {len(result)} rows")
    print(f"Date range: {result['date'].min()} ~ {result['date'].max()}")
    print(f"Stocks per day: {result.groupby('date')['stock_code'].count().median():.0f}")
    print(f"Factor stats:\n{result['factor'].describe()}")

if __name__ == "__main__":
    main()
