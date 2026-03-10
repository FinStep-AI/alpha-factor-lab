#!/usr/bin/env python3
"""
Range Efficiency Factor (range_efficiency_v1)
=============================================
Concept: Intraday price efficiency = |close - open| / (high - low)

High efficiency = directional move (price went one way)
Low efficiency = noisy/choppy (large range but small net move)

Hypothesis: Stocks with LOW range efficiency (choppy, uncertain) tend to
revert as uncertainty resolves → do WORSE in the short term.
Stocks with HIGH range efficiency (clear directional moves) continue
their trend → do BETTER.

This is similar to "directional accuracy" and captures informed trading
conviction vs noise trading.

Direction: POSITIVE (high efficiency → higher future returns)
Neutralize: market cap (OLS)
Window: 20 days

Unique vs existing factors:
- Not volume-based (different from Amihud, turnover_decay)
- Not shadow-based (different from shadow_pressure)
- Not close location (CLV uses (C-L)/(H-L), we use |C-O|/(H-L))
- Captures directional conviction, not price level within range
"""

import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'range_efficiency_v1')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'csi1000_kline_raw.csv'))
    if 'stock_code' in df.columns:
        df = df.rename(columns={'stock_code': 'code'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    return df


def compute_range_efficiency(df, window=20):
    """
    Range Efficiency = |close - open| / (high - low)
    Then take 20-day rolling mean.
    """
    print(f"Computing {window}-day range efficiency...")
    
    # Daily range efficiency
    daily_range = df['high'] - df['low']
    body = (df['close'] - df['open']).abs()
    
    # Avoid division by zero (flat days)
    df['daily_re'] = np.where(daily_range > 0, body / daily_range, 0.5)
    
    # Cap outliers
    df['daily_re'] = df['daily_re'].clip(0, 1)
    
    # Rolling mean
    df['range_eff'] = df.groupby('code')['daily_re'].transform(
        lambda x: x.rolling(window, min_periods=int(window * 0.8)).mean()
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
        factor = group['range_eff'].values
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
    print("Range Efficiency Factor (range_efficiency_v1)")
    print("=" * 60)
    
    df = load_data()
    print(f"Loaded {len(df)} rows, {df['code'].nunique()} stocks")
    print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")
    
    df = compute_range_efficiency(df, window=20)
    df = neutralize_by_mktcap(df)
    
    # Direction: POSITIVE (high efficiency → high factor → high expected return)
    # No sign flip needed
    
    df['stock_code'] = df['code']
    factor_df = df[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    
    factor_path = os.path.join(DATA_DIR, 'factor_range_efficiency_v1.csv')
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
