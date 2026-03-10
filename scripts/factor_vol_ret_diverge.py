#!/usr/bin/env python3
"""
Volume-Return Divergence Factor (vol_ret_diverge_v1)
====================================================
Concept: Compare volume changes with absolute return changes.

When volume is abnormally high but returns are small → institutional accumulation/distribution
When volume is normal but returns are large → momentum continuation

Metric: Rolling 20-day correlation between |return| and volume_ratio
- volume_ratio = volume / MA20_volume
- Low correlation = divergence (volume doesn't track returns)

Direction: NEGATIVE correlation → stocks with high volume but small returns 
tend to be accumulated → future outperformance

Alternatively, we compute:
  volume_surprise = z-score of today's volume vs 20-day MA
  return_surprise = z-score of today's |return| vs 20-day MA
  
  Factor = rolling mean of (volume_surprise - return_surprise)
  
  High factor = volume consistently surprising up more than returns → accumulation
  
This is distinct from:
- pv_corr (correlation of price and volume levels, not surprises)
- Amihud (price impact per dollar volume)
- turnover_decay (temporal pattern of turnover)
"""

import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'vol_ret_diverge_v1')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'csi1000_kline_raw.csv'))
    if 'stock_code' in df.columns:
        df = df.rename(columns={'stock_code': 'code'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    return df


def compute_vol_ret_divergence(df, window=20):
    """
    Compute volume-return divergence.
    """
    print(f"Computing {window}-day volume-return divergence...")
    
    df = df.copy()
    
    # Daily return
    df['ret'] = df.groupby('code')['close'].pct_change()
    df['abs_ret'] = df['ret'].abs()
    
    # Volume z-score (relative to own 20-day rolling)
    vol_ma = df.groupby('code')['volume'].transform(
        lambda x: x.rolling(window, min_periods=int(window*0.8)).mean()
    )
    vol_std = df.groupby('code')['volume'].transform(
        lambda x: x.rolling(window, min_periods=int(window*0.8)).std()
    )
    df['vol_zscore'] = (df['volume'] - vol_ma) / vol_std.clip(lower=1e-10)
    
    # Return z-score (absolute)
    ret_ma = df.groupby('code')['abs_ret'].transform(
        lambda x: x.rolling(window, min_periods=int(window*0.8)).mean()
    )
    ret_std = df.groupby('code')['abs_ret'].transform(
        lambda x: x.rolling(window, min_periods=int(window*0.8)).std()
    )
    df['ret_zscore'] = (df['abs_ret'] - ret_ma) / ret_std.clip(lower=1e-10)
    
    # Divergence: volume surprise - return surprise
    df['daily_diverge'] = df['vol_zscore'] - df['ret_zscore']
    
    # Rolling mean of divergence over the window
    df['vol_ret_diverge'] = df.groupby('code')['daily_diverge'].transform(
        lambda x: x.rolling(window, min_periods=int(window*0.8)).mean()
    )
    
    return df


def neutralize_by_mktcap(df):
    """OLS neutralize by log market cap proxy"""
    print("Market cap neutralization...")
    
    if 'amount' in df.columns:
        df['log_mktcap_proxy'] = np.log(
            df.groupby('code')['amount'].transform(
                lambda x: x.rolling(20).mean()
            ).clip(lower=1)
        )
    else:
        df['log_mktcap_proxy'] = np.log((df['close'] * df['volume']).clip(lower=1))
    
    def neutralize_cs(group):
        factor = group['vol_ret_diverge'].values
        mktcap = group['log_mktcap_proxy'].values
        valid = ~(np.isnan(factor) | np.isnan(mktcap) | np.isinf(factor) | np.isinf(mktcap))
        if valid.sum() < 30:
            return pd.Series(np.nan, index=group.index)
        result = np.full(len(factor), np.nan)
        X = np.column_stack([np.ones(valid.sum()), mktcap[valid]])
        y = factor[valid]
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            result[valid] = y - X @ beta
        except:
            result[valid] = y - np.nanmean(y)
        return pd.Series(result, index=group.index)
    
    df['factor'] = df.groupby('date', group_keys=False).apply(neutralize_cs)
    return df


def main():
    print("=" * 60)
    print("Volume-Return Divergence Factor (vol_ret_diverge_v1)")
    print("=" * 60)
    
    df = load_data()
    print(f"Loaded {len(df)} rows, {df['code'].nunique()} stocks")
    print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")
    
    df = compute_vol_ret_divergence(df, window=20)
    df = neutralize_by_mktcap(df)
    
    # Direction: POSITIVE (high divergence = accumulation → higher future returns)
    
    df['stock_code'] = df['code']
    factor_df = df[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    
    factor_path = os.path.join(DATA_DIR, 'factor_vol_ret_diverge_v1.csv')
    factor_df.to_csv(factor_path, index=False)
    
    print(f"\nFactor saved: {factor_path}")
    print(f"  Rows: {len(factor_df)}")
    print(f"  Stocks: {factor_df['stock_code'].nunique()}")
    print(f"  Dates: {factor_df['date'].min()} ~ {factor_df['date'].max()}")
    print(f"\nFactor stats:")
    print(f"  Mean: {factor_df['factor'].mean():.6f}")
    print(f"  Std:  {factor_df['factor'].std():.6f}")
    
    return factor_df


if __name__ == '__main__':
    main()
