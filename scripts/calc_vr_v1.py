#!/usr/bin/env python3
"""
方差比因子 (Variance Ratio Factor)
===================================
Based on: Lo & MacKinlay (1988) "Stock Market Prices Do Not Follow Random Walks"

Factor construction:
  VR_i,t = Var(r_i, 5d) / (5 * Var(r_i, 1d))
  
  computed over rolling 60-day window.
  
  VR > 1: positive autocorrelation (momentum/trending)
  VR < 1: negative autocorrelation (mean-reverting)
  VR = 1: random walk (efficient)

We test BOTH directions:
  v1: positive (high VR = trending = momentum hypothesis)
  v2: negative (low VR = mean-reverting = efficiency hypothesis)

Neutralization: log transform + OLS on log_amount_60d + MAD winsorize + z-score
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression


def compute_variance_ratio(df, short=1, long=5, window=60):
    """Compute Variance Ratio for all stocks."""
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    results = []
    for code, grp in df.groupby('stock_code'):
        grp = grp.sort_values('date').reset_index(drop=True)
        if len(grp) < window + long + 5:
            continue
        
        ret_1d = grp['close'].pct_change()
        
        # Multi-day returns (sum of log returns for accuracy)
        log_ret = np.log(grp['close'] / grp['close'].shift(1))
        log_ret_kd = log_ret.rolling(long, min_periods=long).sum()
        
        # Rolling variance
        var_1d = ret_1d.rolling(window, min_periods=window).var()
        var_kd = log_ret_kd.rolling(window - long + 1, min_periods=window - long + 1).var()
        
        # Variance ratio
        vr = var_kd / (long * var_1d)
        
        for i in range(len(grp)):
            if pd.notna(vr.iloc[i]) and np.isfinite(vr.iloc[i]) and vr.iloc[i] > 0:
                results.append({
                    'date': grp['date'].iloc[i],
                    'stock_code': code,
                    'vr_raw': vr.iloc[i]
                })
    
    return pd.DataFrame(results)


def neutralize(factor_df, kline_df, col='vr_raw', window=60):
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
        y = np.log(day_df[col].values)  # log transform VR
        
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
    
    print("Computing Variance Ratio (k=5, window=60)...")
    factor_raw = compute_variance_ratio(kline, short=1, long=5, window=60)
    print(f"Raw: {len(factor_raw)} rows")
    
    # Quick check VR distribution
    vr_vals = factor_raw['vr_raw']
    print(f"VR stats: mean={vr_vals.mean():.3f}, median={vr_vals.median():.3f}, "
          f"std={vr_vals.std():.3f}, min={vr_vals.min():.3f}, max={vr_vals.max():.3f}")
    
    print("Neutralizing (positive direction: high VR = trending)...")
    factor_pos = neutralize(factor_raw, kline, col='vr_raw', window=60)
    print(f"Final positive: {len(factor_pos)} rows")
    
    out_pos = os.path.join(data_dir, 'factor_vr_pos_v1.csv')
    factor_pos.to_csv(out_pos, index=False)
    print(f"Saved to {out_pos}")
    
    # Also create negative version (low VR = efficient/mean-reverting)
    factor_neg = factor_pos.copy()
    factor_neg['factor_value'] = -factor_neg['factor_value']
    out_neg = os.path.join(data_dir, 'factor_vr_neg_v1.csv')
    factor_neg.to_csv(out_neg, index=False)
    print(f"Saved negative version to {out_neg}")


if __name__ == '__main__':
    main()
