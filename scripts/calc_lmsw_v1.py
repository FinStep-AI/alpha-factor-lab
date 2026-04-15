#!/usr/bin/env python3
"""
LMSW因子: Volume-Conditional Return Autocorrelation
===================================================
Based on: Llorente, Michaely, Saar & Wang (2002) JFE
"Dynamic Volume-Return Relation of Individual Stocks"

Core idea: When volume is high, the return autocorrelation direction reveals 
whether trading is informed (continuation/momentum) or speculative (reversal).

Factor construction:
  LMSW_i,t = mean( vol_z_{i,t-j} * r_{i,t-j} * r_{i,t-j-1} ) for j=0..N-1
  
  where vol_z = (log_vol - MA(log_vol, 20)) / std(log_vol, 20)
  
  High LMSW = high volume days see return continuation = informed trading
  Low LMSW = high volume days see return reversal = speculative trading

Neutralization: OLS on log_amount_20d + MAD winsorize + z-score
"""

import pandas as pd
import numpy as np
import sys
import os

def compute_lmsw(df, window=20):
    """Compute LMSW factor for all stocks."""
    
    # Sort by stock and date
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    results = []
    
    for code, grp in df.groupby('stock_code'):
        grp = grp.sort_values('date').reset_index(drop=True)
        
        if len(grp) < window + 5:
            continue
        
        # Compute returns
        ret = grp['close'].pct_change()
        
        # Compute log volume z-score (rolling)
        log_vol = np.log(grp['amount'].clip(lower=1))
        log_vol_ma = log_vol.rolling(window, min_periods=window).mean()
        log_vol_std = log_vol.rolling(window, min_periods=window).std()
        vol_z = (log_vol - log_vol_ma) / log_vol_std.clip(lower=0.01)
        
        # Compute volume-conditional return autocorrelation
        # LMSW_t = mean(vol_z_{t-j} * ret_{t-j} * ret_{t-j-1}) for j=0..window-1
        interaction = vol_z * ret * ret.shift(1)
        lmsw = interaction.rolling(window, min_periods=window).mean()
        
        for i in range(len(grp)):
            if pd.notna(lmsw.iloc[i]):
                results.append({
                    'date': grp['date'].iloc[i],
                    'stock_code': code,
                    'lmsw_raw': lmsw.iloc[i]
                })
    
    return pd.DataFrame(results)


def neutralize_and_standardize(factor_df, kline_df, window=20):
    """OLS neutralize on log_amount, MAD winsorize, z-score."""
    from sklearn.linear_model import LinearRegression
    
    # Compute 20d average log amount
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
                    'log_amount_20d': log_amt_ma.iloc[i]
                })
    amt_df = pd.DataFrame(amt_data)
    
    merged = factor_df.merge(amt_df, on=['date', 'stock_code'], how='inner')
    
    output = []
    for dt, day_df in merged.groupby('date'):
        day_df = day_df.dropna(subset=['lmsw_raw', 'log_amount_20d'])
        if len(day_df) < 30:
            continue
        
        X = day_df[['log_amount_20d']].values
        y = day_df['lmsw_raw'].values
        
        # OLS neutralize
        lr = LinearRegression()
        lr.fit(X, y)
        residual = y - lr.predict(X)
        
        # MAD winsorize
        med = np.median(residual)
        mad = np.median(np.abs(residual - med))
        if mad < 1e-10:
            mad = np.std(residual)
        upper = med + 5 * 1.4826 * mad
        lower = med - 5 * 1.4826 * mad
        residual = np.clip(residual, lower, upper)
        
        # Z-score
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
    
    # Pad stock_code to 6 digits
    kline['stock_code'] = kline['stock_code'].str.zfill(6)
    
    print(f"Loaded {len(kline)} rows, {kline['stock_code'].nunique()} stocks")
    
    print("Computing LMSW factor (window=20)...")
    factor_raw = compute_lmsw(kline, window=20)
    print(f"Raw factor: {len(factor_raw)} rows")
    
    print("Neutralizing and standardizing...")
    factor_final = neutralize_and_standardize(factor_raw, kline, window=20)
    print(f"Final factor: {len(factor_final)} rows")
    
    # Save
    out_path = os.path.join(data_dir, 'factor_lmsw_v1.csv')
    factor_final.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    
    # Quick stats
    print("\n=== Factor Stats ===")
    for dt in factor_final['date'].unique()[-5:]:
        day = factor_final[factor_final['date'] == dt]
        print(f"{dt.strftime('%Y-%m-%d')}: n={len(day)}, mean={day['factor_value'].mean():.4f}, std={day['factor_value'].std():.4f}")


if __name__ == '__main__':
    main()
