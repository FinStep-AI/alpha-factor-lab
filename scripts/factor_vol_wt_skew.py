#!/usr/bin/env python3
"""
因子: 成交额加权收益偏度 (Volume-Weighted Return Skewness)
逻辑: 过去20天日收益率的成交额加权偏度
高偏度 = 大成交额日集中在正收益 = 资金流入偏向上涨日 = 看多信号
中性化: 成交额OLS中性化 + MAD winsorize + z-score
"""

import pandas as pd
import numpy as np
import sys

LOOKBACK = 20
OUTPUT_PATH = "data/factor_vol_wt_skew_v1.csv"

def weighted_skewness(returns, weights):
    """计算加权偏度"""
    n = len(returns)
    if n < 10:
        return np.nan
    
    # 标准化权重
    w = weights / weights.sum()
    
    # 加权均值
    mu = np.sum(w * returns)
    
    # 加权方差
    var = np.sum(w * (returns - mu) ** 2)
    if var <= 0:
        return np.nan
    
    std = np.sqrt(var)
    
    # 加权偏度 (三阶矩)
    skew = np.sum(w * ((returns - mu) / std) ** 3)
    
    return skew

def mad_winsorize(s, n_mad=5):
    median = s.median()
    mad = (s - median).abs().median()
    if mad == 0:
        return s
    upper = median + n_mad * 1.4826 * mad
    lower = median - n_mad * 1.4826 * mad
    return s.clip(lower=lower, upper=upper)

def neutralize_ols(factor_series, neutralizer_series):
    from numpy.linalg import lstsq
    mask = factor_series.notna() & neutralizer_series.notna()
    if mask.sum() < 10:
        return factor_series
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
    
    # 计算日收益率
    kline['ret'] = kline.groupby('stock_code')['close'].pct_change()
    
    print(f"Computing {LOOKBACK}d volume-weighted return skewness...")
    
    all_results = []
    stocks = kline['stock_code'].unique()
    
    for i, stock in enumerate(stocks):
        if (i + 1) % 200 == 0:
            print(f"  Processing {i+1}/{len(stocks)} stocks...")
        
        stock_data = kline[kline['stock_code'] == stock].sort_values('date')
        rets = stock_data['ret'].values
        amounts = stock_data['amount'].values
        dates = stock_data['date'].values
        codes = stock_data['stock_code'].values
        
        for j in range(LOOKBACK - 1, len(rets)):
            r_window = rets[j - LOOKBACK + 1: j + 1]
            a_window = amounts[j - LOOKBACK + 1: j + 1]
            
            # 跳过含NaN
            valid = ~(np.isnan(r_window) | np.isnan(a_window) | (a_window <= 0))
            if valid.sum() < 15:
                continue
            
            skew_val = weighted_skewness(r_window[valid], a_window[valid])
            if not np.isnan(skew_val):
                all_results.append({
                    'date': dates[j],
                    'stock_code': codes[j],
                    'raw_factor': skew_val
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
    
    # 截面处理
    print("Cross-sectional processing...")
    processed = []
    
    for date, group in df.groupby('date'):
        g = group.copy()
        g['factor'] = mad_winsorize(g['raw_factor'])
        g['factor'] = neutralize_ols(g['factor'], g['log_amount_20d'])
        mean = g['factor'].mean()
        std = g['factor'].std()
        if std > 0:
            g['factor'] = (g['factor'] - mean) / std
        else:
            g['factor'] = 0
        processed.append(g[['date', 'stock_code', 'factor']])
    
    result = pd.concat(processed, ignore_index=True)
    result = result.sort_values(['date', 'stock_code']).reset_index(drop=True)
    
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}: {len(result)} rows")
    print(f"Date range: {result['date'].min()} ~ {result['date'].max()}")
    print(f"Factor stats:\n{result['factor'].describe()}")

if __name__ == "__main__":
    main()
