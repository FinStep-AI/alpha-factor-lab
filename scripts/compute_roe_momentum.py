#!/usr/bin/env python3
"""
ROE Momentum Factor (ROE动量/盈利改善因子)
==========================================

Barra Style: Growth
Factor ID: roe_momentum_v1

Logic:
------
ROE Momentum = ROE(当季) - ROE(去年同季)  [YoY change]

对于每个季度财报:
  - Q2 2023 ROE - Q2 2022 ROE = YoY change
  - 去除ROE异常值(>100 or <-100)
  - 用最近可用的YoY ROE change作为因子值
  - 考虑财报披露滞后: 用2个月滞后(1Q报4月底前, 2Q/半年报8月底前, 3Q报10月底前, 年报4月底前)

Point-in-time mapping:
  - Q1 (3-31) → available after 2024-04-30 → usable from 2024-05-01
  - Q2 (6-30) → available after 2024-08-31 → usable from 2024-09-01
  - Q3 (9-30) → available after 2024-10-31 → usable from 2024-11-01
  - Q4 (12-31) → available after 2025-04-30 → usable from 2025-05-01

Neutralization: OLS regression on log(market_cap), take residual
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Paths
BASE = Path(__file__).resolve().parent.parent
FUND_PATH = BASE / "data" / "csi1000_fundamental_cache.csv"
KLINE_PATH = BASE / "data" / "csi1000_kline_raw.csv"
OUTPUT_PATH = BASE / "data" / "factor_roe_momentum_v1.csv"


def load_data():
    fund = pd.read_csv(FUND_PATH)
    kline = pd.read_csv(KLINE_PATH)
    kline['date'] = pd.to_datetime(kline['date'])
    fund['report_date'] = pd.to_datetime(fund['report_date'])
    return fund, kline


def compute_roe_yoy(fund):
    """Compute YoY ROE change for each stock-quarter."""
    # Filter extreme ROE values
    fund = fund.copy()
    fund.loc[fund['roe'].abs() > 100, 'roe'] = np.nan
    
    # Extract quarter info
    fund['quarter'] = fund['report_date'].dt.quarter
    fund['year'] = fund['report_date'].dt.year
    
    # Sort
    fund = fund.sort_values(['stock_code', 'year', 'quarter'])
    
    # For each stock-quarter, find same quarter last year
    records = []
    for (sc, q), grp in fund.groupby(['stock_code', 'quarter']):
        grp = grp.sort_values('year')
        grp['roe_yoy'] = grp['roe'].diff()  # diff between consecutive years for same quarter
        # Only keep rows where we have YoY (i.e., have last year's data)
        valid = grp.dropna(subset=['roe_yoy'])
        for _, row in valid.iterrows():
            records.append({
                'stock_code': row['stock_code'],
                'report_date': row['report_date'],
                'roe': row['roe'],
                'roe_yoy': row['roe_yoy'],
                'year': row['year'],
                'quarter': row['quarter']
            })
    
    roe_yoy = pd.DataFrame(records)
    print(f"ROE YoY records: {len(roe_yoy)}")
    print(f"ROE YoY stats:\n{roe_yoy['roe_yoy'].describe()}")
    return roe_yoy


def pit_mapping(report_date):
    """Point-in-time: when is this report available for trading?"""
    q = report_date.quarter
    y = report_date.year
    if q == 1:  # Q1 report available by April 30
        return pd.Timestamp(y, 5, 1)
    elif q == 2:  # Semi-annual available by August 31
        return pd.Timestamp(y, 9, 1)
    elif q == 3:  # Q3 report available by October 31
        return pd.Timestamp(y, 11, 1)
    else:  # Annual report available by April 30 next year
        return pd.Timestamp(y + 1, 5, 1)


def map_to_trading_days(roe_yoy, kline):
    """Map ROE YoY to trading days using point-in-time."""
    # Get all trading dates
    trading_dates = sorted(kline['date'].unique())
    
    # Add available_from date
    roe_yoy['available_from'] = roe_yoy['report_date'].apply(pit_mapping)
    
    # Sort by stock and available_from
    roe_yoy = roe_yoy.sort_values(['stock_code', 'available_from'])
    
    # For each stock, forward-fill the latest available ROE YoY to each trading day
    stocks = kline['stock_code'].unique()
    results = []
    
    for sc in stocks:
        sc_fund = roe_yoy[roe_yoy['stock_code'] == sc].copy()
        if len(sc_fund) == 0:
            continue
        
        sc_kline = kline[kline['stock_code'] == sc][['date', 'stock_code']].copy()
        
        # Merge: for each trading date, find the latest available ROE YoY
        sc_kline = sc_kline.sort_values('date')
        sc_fund = sc_fund.sort_values('available_from')
        
        # Use merge_asof to get the latest available fundamental data for each trading day
        merged = pd.merge_asof(
            sc_kline,
            sc_fund[['available_from', 'roe_yoy', 'stock_code']].rename(columns={'available_from': 'date'}),
            on='date',
            by='stock_code',
            direction='backward'
        )
        results.append(merged)
    
    factor_df = pd.concat(results, ignore_index=True)
    factor_df = factor_df.dropna(subset=['roe_yoy'])
    print(f"\nFactor mapped to trading days: {len(factor_df)} rows")
    print(f"Date range: {factor_df['date'].min()} ~ {factor_df['date'].max()}")
    print(f"Stocks per day (sample):")
    sample_counts = factor_df.groupby('date')['stock_code'].count()
    print(f"  Mean: {sample_counts.mean():.0f}, Min: {sample_counts.min()}, Max: {sample_counts.max()}")
    return factor_df


def neutralize(factor_df, kline):
    """Market cap neutralization using OLS regression on log(amount_20d)."""
    # Compute 20d average amount as market cap proxy
    kline_sorted = kline.sort_values(['stock_code', 'date'])
    kline_sorted['amount_20d'] = kline_sorted.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    kline_sorted['log_amount_20d'] = np.log(kline_sorted['amount_20d'].clip(lower=1))
    
    # Merge
    factor_df = factor_df.merge(
        kline_sorted[['date', 'stock_code', 'log_amount_20d']],
        on=['date', 'stock_code'],
        how='left'
    )
    factor_df = factor_df.dropna(subset=['log_amount_20d', 'roe_yoy'])
    
    # Winsorize: MAD 3x
    def winsorize_mad(s, k=3):
        med = s.median()
        mad = (s - med).abs().median()
        if mad == 0:
            return s
        lower = med - k * 1.4826 * mad
        upper = med + k * 1.4826 * mad
        return s.clip(lower, upper)
    
    # Cross-sectional neutralization
    results = []
    for dt, grp in factor_df.groupby('date'):
        if len(grp) < 50:
            continue
        
        raw = grp['roe_yoy'].copy()
        raw = winsorize_mad(raw)
        
        x = grp['log_amount_20d'].values
        y = raw.values
        
        # OLS: y = a + b*x + residual
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 50:
            continue
        
        x_m, y_m = x[mask], y[mask]
        X = np.column_stack([np.ones(len(x_m)), x_m])
        try:
            beta = np.linalg.lstsq(X, y_m, rcond=None)[0]
        except:
            continue
        
        residual = np.full(len(x), np.nan)
        residual[mask] = y_m - X @ beta
        
        # Z-score standardize
        res_std = np.nanstd(residual)
        if res_std > 0:
            residual = (residual - np.nanmean(residual)) / res_std
        
        grp_out = grp[['date', 'stock_code']].copy()
        grp_out['factor_value'] = residual
        results.append(grp_out)
    
    result = pd.concat(results, ignore_index=True)
    result = result.dropna(subset=['factor_value'])
    print(f"\nAfter neutralization: {len(result)} rows")
    return result


def main():
    print("=" * 60)
    print("ROE Momentum Factor (盈利改善因子)")
    print("=" * 60)
    
    # Load data
    fund, kline = load_data()
    print(f"Fundamental data: {len(fund)} rows, {fund['stock_code'].nunique()} stocks")
    print(f"Kline data: {len(kline)} rows, {kline['date'].min()} ~ {kline['date'].max()}")
    
    # Step 1: Compute ROE YoY
    roe_yoy = compute_roe_yoy(fund)
    
    # Step 2: Map to trading days (point-in-time)
    factor_raw = map_to_trading_days(roe_yoy, kline)
    
    # Step 3: Neutralize
    factor_final = neutralize(factor_raw, kline)
    
    # Save
    factor_final.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Final shape: {factor_final.shape}")
    print(f"Date range: {factor_final['date'].min()} ~ {factor_final['date'].max()}")
    
    # Summary stats
    print(f"\nFactor value stats:")
    print(factor_final['factor_value'].describe())


if __name__ == "__main__":
    main()
