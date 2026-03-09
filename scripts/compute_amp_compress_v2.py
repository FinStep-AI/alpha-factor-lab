#!/usr/bin/env python3
"""
Amplitude Compression v2 — with 10d/40d window and combined with volume compression
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
KLINE_PATH = BASE / "data" / "csi1000_kline_raw.csv"
OUTPUT_PATH = BASE / "data" / "factor_amp_compress_v2.csv"


def main():
    print("=" * 60)
    print("Amplitude Compression v2 (10d/40d + volume)")
    print("=" * 60)
    
    kline = pd.read_csv(KLINE_PATH)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date'])
    
    # Daily amplitude
    kline['amplitude_raw'] = (kline['high'] - kline['low']) / kline['close'].clip(lower=0.01)
    
    # Rolling means: 10d / 40d
    kline['amp_10d'] = kline.groupby('stock_code')['amplitude_raw'].transform(
        lambda x: x.rolling(10, min_periods=8).mean()
    )
    kline['amp_40d'] = kline.groupby('stock_code')['amplitude_raw'].transform(
        lambda x: x.rolling(40, min_periods=30).mean()
    )
    
    # Amplitude compression ratio
    kline['amp_ratio'] = kline['amp_10d'] / kline['amp_40d'].clip(lower=1e-6)
    
    # Also compute volume compression
    kline['vol_10d'] = kline.groupby('stock_code')['turnover'].transform(
        lambda x: x.rolling(10, min_periods=8).mean()
    )
    kline['vol_40d'] = kline.groupby('stock_code')['turnover'].transform(
        lambda x: x.rolling(40, min_periods=30).mean()
    )
    kline['vol_ratio'] = kline['vol_10d'] / kline['vol_40d'].clip(lower=1e-6)
    
    # Combined: amplitude + volume compression (both low = double compression)
    kline['combined'] = np.log(kline['amp_ratio'].clip(lower=0.01)) + np.log(kline['vol_ratio'].clip(lower=0.01))
    
    # Factor: NEGATIVE (more compression = higher factor value)
    kline['raw_factor'] = -kline['combined']
    kline['raw_factor'] = kline['raw_factor'].replace([np.inf, -np.inf], np.nan)
    
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


if __name__ == "__main__":
    main()
