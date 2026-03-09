#!/usr/bin/env python3
"""
Volatility Term Structure Factor (波动率期限结构因子)
=====================================================

Barra Style: Volatility (fills gap alongside idio_vol_v1)
Factor ID: vol_term_structure_v1

Logic:
------
Vol_Ratio = realized_vol_5d / realized_vol_20d

When short-term vol >> long-term vol → selling climax → reversal likely
Use negative direction: low Vol_Ratio (calm period) → do well? OR high Vol_Ratio → reverse?

Actually, let's test: 
- REVERSE direction: high short-term vol spikes are followed by mean reversion
- neutralize by log(amount_20d)

Construction:
- realized_vol_5d = std(daily_returns, 5d)
- realized_vol_20d = std(daily_returns, 20d)  
- factor = -log(vol_5d / vol_20d)  [negative = high vol spike → do LONG for reversal]
- neutralize by log_amount_20d
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
KLINE_PATH = BASE / "data" / "csi1000_kline_raw.csv"
OUTPUT_PATH = BASE / "data" / "factor_vol_term_v1.csv"


def main():
    print("=" * 60)
    print("Volatility Term Structure Factor (波动率期限结构)")
    print("=" * 60)
    
    # Load
    kline = pd.read_csv(KLINE_PATH)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date'])
    print(f"Kline: {len(kline)} rows, {kline['date'].min()} ~ {kline['date'].max()}")
    
    # Compute daily returns
    kline['ret'] = kline.groupby('stock_code')['close'].pct_change()
    
    # Realized vol: 5d and 20d
    kline['vol_5d'] = kline.groupby('stock_code')['ret'].transform(
        lambda x: x.rolling(5, min_periods=4).std()
    )
    kline['vol_20d'] = kline.groupby('stock_code')['ret'].transform(
        lambda x: x.rolling(20, min_periods=15).std()
    )
    
    # Vol ratio: short/long
    kline['vol_ratio'] = kline['vol_5d'] / kline['vol_20d']
    
    # Replace inf/nan
    kline['vol_ratio'] = kline['vol_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Take log for better distribution
    kline['log_vol_ratio'] = np.log(kline['vol_ratio'].clip(lower=0.01))
    
    # Factor: NEGATIVE of log_vol_ratio → high short-term vol spike → long for reversal
    kline['raw_factor'] = -kline['log_vol_ratio']
    
    # 20d average amount for neutralization
    kline['amount_20d'] = kline.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    kline['log_amount_20d'] = np.log(kline['amount_20d'].clip(lower=1))
    
    # Drop NaN
    valid = kline.dropna(subset=['raw_factor', 'log_amount_20d']).copy()
    print(f"Valid rows: {len(valid)}")
    
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
    results = []
    for dt, grp in valid.groupby('date'):
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
        results.append(grp_out)
    
    result = pd.concat(results, ignore_index=True)
    result = result.dropna(subset=['factor_value'])
    
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Shape: {result.shape}")
    print(f"Date range: {result['date'].min()} ~ {result['date'].max()}")
    print(f"Factor stats:\n{result['factor_value'].describe()}")


if __name__ == "__main__":
    main()
