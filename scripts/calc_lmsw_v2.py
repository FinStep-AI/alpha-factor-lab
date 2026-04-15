#!/usr/bin/env python3
"""
LMSW因子v2: Volume-Conditional Return Autocorrelation (改进版)
=============================================================
改进:
1. 60天窗口(原v1用20天噪音太大)
2. 用detrended volume (vol/MA60) 替代z-score 
3. 尝试两个方向: 正向(知情交易/动量) + 反向(投机反转)
4. 同时输出一个简化版: sign(ret)*ret_lag1*vol_z 的均值
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression


def compute_lmsw_v2(df, window=60):
    """Compute LMSW v2 factor."""
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    results = []
    for code, grp in df.groupby('stock_code'):
        grp = grp.sort_values('date').reset_index(drop=True)
        if len(grp) < window + 5:
            continue
        
        ret = grp['close'].pct_change()
        
        # Detrended log volume
        log_vol = np.log(grp['amount'].clip(lower=1))
        log_vol_ma = log_vol.rolling(window, min_periods=window).mean()
        vol_detrend = log_vol - log_vol_ma  # positive = above average volume
        
        # Volume-conditional return autocorrelation
        # LMSW = mean( vol_detrend_t * ret_t * ret_{t-1} )  over window
        interaction = vol_detrend * ret * ret.shift(1)
        lmsw = interaction.rolling(window, min_periods=window).mean()
        
        for i in range(len(grp)):
            if pd.notna(lmsw.iloc[i]):
                results.append({
                    'date': grp['date'].iloc[i],
                    'stock_code': code,
                    'lmsw_raw': lmsw.iloc[i]
                })
    
    return pd.DataFrame(results)


def neutralize(factor_df, kline_df, col='lmsw_raw', window=60):
    """OLS neutralize on log_amount, MAD winsorize, z-score."""
    kline_sorted = kline_df.sort_values(['stock_code', 'date'])
    amt_data = []
    for code, grp in kline_sorted.groupby('stock_code'):
        grp = grp.sort_values('date')
        log_amt = np.log(grp['amount'].clip(lower=1))
        log_amt_ma = log_amt.rolling(window, min_periods=window).mean()
        for i in range(len(grp)):
            if pd.notna(log_amt_ma.iloc[i]):
                amt_data.append({
                    'date': grp['date'].iloc[i],
                    'stock_code': code,
                    'log_amount': log_amt_ma.iloc[i]
                })
    amt_df = pd.DataFrame(amt_data)
    merged = factor_df.merge(amt_df, on=['date', 'stock_code'], how='inner')
    
    output = []
    for dt, day_df in merged.groupby('date'):
        day_df = day_df.dropna(subset=[col, 'log_amount'])
        if len(day_df) < 30:
            continue
        
        X = day_df[['log_amount']].values
        y = day_df[col].values
        
        lr = LinearRegression()
        lr.fit(X, y)
        residual = y - lr.predict(X)
        
        med = np.median(residual)
        mad = np.median(np.abs(residual - med))
        if mad < 1e-10:
            mad = np.std(residual)
        upper = med + 5 * 1.4826 * mad
        lower = med - 5 * 1.4826 * mad
        residual = np.clip(residual, lower, upper)
        
        mean_r = np.mean(residual)
        std_r = np.std(residual)
        if std_r < 1e-10:
            continue
        z = (residual - mean_r) / std_r
        z = np.clip(z, -3, 3)
        
        for j, idx in enumerate(day_df.index):
            output.append({
                'date': dt,
                'stock_code': day_df.loc[idx, 'stock_code'],
                'factor_value': z[j]
            })
    
    return pd.DataFrame(output)


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    print("Loading kline data...")
    kline = pd.read_csv(os.path.join(data_dir, 'csi1000_kline_raw.csv'),
                        dtype={'stock_code': str})
    kline['date'] = pd.to_datetime(kline['date'])
    kline['stock_code'] = kline['stock_code'].str.zfill(6)
    
    print(f"Loaded {len(kline)} rows, {kline['stock_code'].nunique()} stocks")
    
    print("Computing LMSW v2 (60d window)...")
    factor_raw = compute_lmsw_v2(kline, window=60)
    print(f"Raw: {len(factor_raw)} rows")
    
    print("Neutralizing...")
    factor_final = neutralize(factor_raw, kline, col='lmsw_raw', window=60)
    print(f"Final: {len(factor_final)} rows")
    
    out_path = os.path.join(data_dir, 'factor_lmsw_v2.csv')
    factor_final.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    
    # Also output negative version for reversal direction
    factor_neg = factor_final.copy()
    factor_neg['factor_value'] = -factor_neg['factor_value']
    out_neg = os.path.join(data_dir, 'factor_lmsw_v2_neg.csv')
    factor_neg.to_csv(out_neg, index=False)
    print(f"Saved negative version to {out_neg}")


if __name__ == '__main__':
    main()
