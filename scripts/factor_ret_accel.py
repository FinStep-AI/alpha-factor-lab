#!/usr/bin/env python3
"""
因子: 收益率加速度 (Return Acceleration)
逻辑: 过去20天收益率的"加速度" = 后10天累计收益 - 前10天累计收益
正加速度 = 最近涨幅在扩大(加速上行) = 动量加速
负加速度 = 最近涨幅在缩小或转跌(减速/加速下行)
市值中性化 + MAD winsorize + z-score
"""

import pandas as pd
import numpy as np
import sys

LOOKBACK = 20
HALF = LOOKBACK // 2  # 10
OUTPUT_PATH = "data/factor_ret_accel_v1.csv"

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
    
    # 日收益率
    kline['ret'] = kline.groupby('stock_code')['close'].pct_change()
    
    print(f"Computing {LOOKBACK}d return acceleration (half={HALF})...")
    
    all_results = []
    stocks = kline['stock_code'].unique()
    
    for i, stock in enumerate(stocks):
        if (i + 1) % 200 == 0:
            print(f"  Processing {i+1}/{len(stocks)} stocks...")
        
        stock_data = kline[kline['stock_code'] == stock].sort_values('date')
        rets = stock_data['ret'].values
        dates = stock_data['date'].values
        codes = stock_data['stock_code'].values
        amounts = stock_data['amount'].values
        
        for j in range(LOOKBACK - 1, len(rets)):
            window = rets[j - LOOKBACK + 1: j + 1]
            if np.isnan(window).sum() > 3:
                continue
            
            # 前半段(更早的10天) vs 后半段(更近的10天)
            first_half = window[:HALF]
            second_half = window[HALF:]
            
            r1 = np.nansum(first_half)  # 前10天累计收益
            r2 = np.nansum(second_half)  # 后10天累计收益
            
            accel = r2 - r1  # 加速度: 后半段超越前半段的幅度
            
            all_results.append({
                'date': dates[j],
                'stock_code': codes[j],
                'raw_factor': accel
            })
    
    df = pd.DataFrame(all_results)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Raw factor computed: {len(df)} rows")
    
    # 合并成交额
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
