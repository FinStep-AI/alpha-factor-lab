#!/usr/bin/env python3
"""
Price Efficiency Ratio Factor (价格效率因子)
=============================================

Barra Style: Momentum / Microstructure
Factor ID: price_efficiency_v1

Logic:
------
Price Efficiency = |Close(t) - Close(t-N)| / sum(|Close(i) - Close(i-1)|, i=t-N+1..t)

This measures how "efficiently" the price moved from A to B.
- High efficiency = trending (straight line move)
- Low efficiency = meandering (lots of back and forth, noise)

In efficient markets, trending stocks continue (momentum).
In CSI1000 (less efficient), this could go either way.

Hypothesis: 
- High price efficiency = strong trend → momentum continuation
- Low price efficiency = choppy = uncertainty → possibly negative

Test: 20-day window, positive direction first.
If doesn't work, try negative direction.

Neutralize by log(amount_20d).
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
KLINE_PATH = BASE / "data" / "csi1000_kline_raw.csv"
OUTPUT_PATH = BASE / "data" / "factor_price_efficiency_v1.csv"


def main():
    print("=" * 60)
    print("Price Efficiency Ratio Factor")
    print("=" * 60)
    
    kline = pd.read_csv(KLINE_PATH)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date'])
    print(f"Kline: {len(kline)} rows")
    
    N = 20  # window
    
    results_list = []
    for sc, grp in kline.groupby('stock_code'):
        grp = grp.copy().reset_index(drop=True)
        close = grp['close'].values
        
        # Net price change over N days
        net_change = np.abs(close[N:] - close[:-N])
        
        # Total absolute daily changes
        daily_abs_change = np.abs(np.diff(close))
        total_path = np.array([
            daily_abs_change[max(0, i-N+1):i+1].sum() 
            for i in range(len(daily_abs_change))
        ])
        total_path_aligned = total_path[N-1:]  # align with net_change
        
        # Efficiency ratio
        efficiency = np.where(total_path_aligned > 0, net_change / total_path_aligned, np.nan)
        
        # Align dates
        dates = grp['date'].values[N:]
        
        out = pd.DataFrame({
            'date': dates,
            'stock_code': sc,
            'raw_factor': efficiency
        })
        results_list.append(out)
    
    df = pd.concat(results_list, ignore_index=True)
    df = df.dropna(subset=['raw_factor'])
    df = df[np.isfinite(df['raw_factor'])]
    print(f"Raw factor rows: {len(df)}")
    print(f"Raw factor stats:\n{df['raw_factor'].describe()}")
    
    # 20d amount for neutralization
    kline_sorted = kline.sort_values(['stock_code', 'date'])
    kline_sorted['amount_20d'] = kline_sorted.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    kline_sorted['log_amount_20d'] = np.log(kline_sorted['amount_20d'].clip(lower=1))
    
    df = df.merge(
        kline_sorted[['date', 'stock_code', 'log_amount_20d']],
        on=['date', 'stock_code'],
        how='left'
    )
    df = df.dropna(subset=['log_amount_20d'])
    
    # Winsorize MAD
    def winsorize_mad(s, k=3):
        med = s.median()
        mad = (s - med).abs().median()
        if mad == 0:
            return s
        lower = med - k * 1.4826 * mad
        upper = med + k * 1.4826 * mad
        return s.clip(lower, upper)
    
    # Cross-sectional neutralization
    out_records = []
    for dt, grp in df.groupby('date'):
        if len(grp) < 100:
            continue
        
        raw = winsorize_mad(grp['raw_factor'])
        x = grp['log_amount_20d'].values
        y = raw.values
        
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 100:
            continue
        
        x_m, y_m = x[mask], y[mask]
        X = np.column_stack([np.ones(len(x_m)), x_m])
        try:
            beta = np.linalg.lstsq(X, y_m, rcond=None)[0]
        except:
            continue
        
        residual = np.full(len(x), np.nan)
        residual[mask] = y_m - X @ beta
        
        res_std = np.nanstd(residual)
        if res_std > 0:
            residual = (residual - np.nanmean(residual)) / res_std
        
        grp_out = grp[['date', 'stock_code']].copy()
        grp_out['factor_value'] = residual
        out_records.append(grp_out)
    
    result = pd.concat(out_records, ignore_index=True)
    result = result.dropna(subset=['factor_value'])
    
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Shape: {result.shape}")
    print(f"Date range: {result['date'].min()} ~ {result['date'].max()}")
    print(f"Factor stats:\n{result['factor_value'].describe()}")


if __name__ == "__main__":
    main()
