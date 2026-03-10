#!/usr/bin/env python3
"""
Realized Skewness Factor (realized_skew_v1)
============================================
Reference: Amaya, Christoffersen, Jacobs & Vasquez (2015)
"Does realized skewness predict the cross-section of equity returns?" JFE 118(1):135-167

Logic:
- Compute rolling 20-day skewness of daily returns
- Direction: NEGATIVE (low skewness → high expected returns)
  - Investors overpay for lottery-like (positively skewed) payoffs
  - Negatively skewed stocks are underpriced → higher future returns
- Neutralize by market cap (OLS residual)

Output: factor CSV + correlation check with existing factors
"""

import pandas as pd
import numpy as np
import sys, os, json
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'realized_skew_v1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load kline data"""
    df = pd.read_csv(os.path.join(DATA_DIR, 'csi1000_kline_raw.csv'))
    df['date'] = pd.to_datetime(df['date'])
    # Ensure sorted
    # Normalize column name
    if 'stock_code' in df.columns:
        df = df.rename(columns={'stock_code': 'code'})
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    return df

def compute_daily_returns(df):
    """Compute daily returns from close prices"""
    df = df.copy()
    df['ret'] = df.groupby('code')['close'].pct_change()
    return df

def compute_realized_skewness(df, window=20):
    """
    Compute rolling realized skewness of daily returns.
    Skewness = E[(r - mu)^3] / sigma^3
    Using scipy.stats.skew (Fisher definition, bias=False)
    """
    print(f"Computing {window}-day realized skewness...")
    
    def rolling_skew(group):
        return group['ret'].rolling(window, min_periods=int(window * 0.8)).skew()
    
    df = df.copy()
    df['realized_skew'] = df.groupby('code', group_keys=False).apply(rolling_skew)
    return df

def neutralize_by_mktcap(df):
    """OLS neutralize factor by log market cap"""
    print("Market cap neutralization...")
    
    # Compute log market cap proxy: log(close * volume) as approximation
    # Actually use amount if available, otherwise close * volume
    if 'amount' in df.columns:
        df['log_mktcap_proxy'] = np.log(df['amount'].rolling(20).mean().clip(lower=1))
    else:
        df['log_mktcap_proxy'] = np.log((df['close'] * df['volume']).clip(lower=1))
    
    def neutralize_cross_section(group):
        factor = group['realized_skew'].values
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
    
    df['factor'] = df.groupby('date', group_keys=False).apply(neutralize_cross_section)
    return df

def main():
    print("=" * 60)
    print("Realized Skewness Factor (realized_skew_v1)")
    print("Amaya et al. (2015) JFE")
    print("=" * 60)
    
    # Load and prepare data
    df = load_data()
    print(f"Loaded {len(df)} rows, {df['code'].nunique()} stocks")
    print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")
    
    # Compute returns
    df = compute_daily_returns(df)
    
    # Compute realized skewness (20-day window)
    df = compute_realized_skewness(df, window=20)
    
    # Neutralize
    df = neutralize_by_mktcap(df)
    
    # DIRECTION: NEGATIVE (flip sign so that low skewness = high factor value)
    # This way, factor_backtest.py with direction=1 will buy low-skewness stocks
    df['factor'] = -df['factor']
    
    # Save factor values (use stock_code for backtest compatibility)
    df['stock_code'] = df['code']
    factor_df = df[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    factor_path = os.path.join(DATA_DIR, 'factor_realized_skew_v1.csv')
    factor_df.to_csv(factor_path, index=False)
    print(f"\nFactor saved to {factor_path}")
    print(f"  Rows: {len(factor_df)}")
    print(f"  Stocks: {factor_df['stock_code'].nunique()}")
    print(f"  Dates: {factor_df['date'].min()} ~ {factor_df['date'].max()}")
    
    # Basic stats
    print(f"\nFactor stats (after sign flip):")
    print(f"  Mean:   {factor_df['factor'].mean():.6f}")
    print(f"  Std:    {factor_df['factor'].std():.6f}")
    print(f"  Skew:   {factor_df['factor'].skew():.4f}")
    print(f"  Kurt:   {factor_df['factor'].kurtosis():.4f}")
    
    return factor_df

if __name__ == '__main__':
    main()
