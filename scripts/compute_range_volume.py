#!/usr/bin/env python3
"""
Range-Volume Ratio Factor (振幅流动性因子)
==========================================

Barra Style: Liquidity / Volatility
Factor ID: range_volume_v1

Logic:
------
Range_Volume = (High - Low) / (Amount in 亿)

This is like Amihud but uses intraday range instead of absolute returns.
Range captures true price volatility better than close-to-close returns 
(which miss intraday mean-reversion).

20d rolling mean, log transform, neutralize by market cap.

Hypothesis: High range-volume ratio = wide swings on low volume = illiquid + volatile.
In CSI1000, both illiquidity and high volatility are rewarded (Amihud + idio_vol both positive).
So this compound factor should be positive direction.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
KLINE_PATH = BASE / "data" / "csi1000_kline_raw.csv"
OUTPUT_PATH = BASE / "data" / "factor_range_volume_v1.csv"


def main():
    print("=" * 60)
    print("Range-Volume Ratio Factor")
    print("=" * 60)
    
    kline = pd.read_csv(KLINE_PATH)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date'])
    print(f"Kline: {len(kline)} rows")
    
    # Range = (High - Low) / Close (normalized)
    kline['range_norm'] = (kline['high'] - kline['low']) / kline['close'].clip(lower=0.01)
    
    # Range / Amount (in 亿)
    kline['range_vol'] = kline['range_norm'] / (kline['amount'].clip(lower=1) / 1e8)
    kline['range_vol'] = kline['range_vol'].replace([np.inf, -np.inf], np.nan)
    
    # 20d rolling mean
    kline['rv_20d'] = kline.groupby('stock_code')['range_vol'].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )
    
    # Log transform
    kline['raw_factor'] = np.log(kline['rv_20d'].clip(lower=1e-10))
    
    # 20d amount for neutralization
    kline['amount_20d'] = kline.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    kline['log_amount_20d'] = np.log(kline['amount_20d'].clip(lower=1))
    
    valid = kline.dropna(subset=['raw_factor', 'log_amount_20d']).copy()
    valid = valid[np.isfinite(valid['raw_factor'])]
    print(f"Valid rows: {len(valid)}")
    
    # Winsorize MAD
    def winsorize_mad(s, k=3):
        med = s.median()
        mad = (s - med).abs().median()
        if mad == 0:
            return s
        return s.clip(med - k * 1.4826 * mad, med + k * 1.4826 * mad)
    
    # Cross-sectional neutralization
    out_records = []
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
