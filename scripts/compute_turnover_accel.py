#!/usr/bin/env python3
"""
Turnover Acceleration Factor (换手率加速度因子)
================================================

Barra Style: 微观结构/资金流
Factor ID: turnover_accel_v1

Logic:
------
Turnover_Accel = MA5(turnover) / MA20(turnover) - 1

When recent turnover is rising relative to history → attention/capital inflow
In CSI1000: rising attention on small caps often precedes momentum

Factor direction: POSITIVE (rising turnover = good for future returns)
Combine with return direction: use turnover accel × sign(5d return)
This way: rising turnover + up trend = buy signal
           rising turnover + down trend = sell signal

Neutralize by log(amount_20d).
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
KLINE_PATH = BASE / "data" / "csi1000_kline_raw.csv"
OUTPUT_PATH = BASE / "data" / "factor_turnover_accel_v1.csv"


def main():
    print("=" * 60)
    print("Turnover Acceleration Factor")
    print("=" * 60)
    
    kline = pd.read_csv(KLINE_PATH)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date'])
    print(f"Kline: {len(kline)} rows")
    
    # Returns for direction
    kline['ret'] = kline.groupby('stock_code')['close'].pct_change()
    kline['ret_5d'] = kline.groupby('stock_code')['close'].pct_change(5)
    
    # Turnover means
    kline['turn_5d'] = kline.groupby('stock_code')['turnover'].transform(
        lambda x: x.rolling(5, min_periods=4).mean()
    )
    kline['turn_20d'] = kline.groupby('stock_code')['turnover'].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )
    
    # Acceleration: (short-term / long-term) - 1
    kline['turn_accel'] = kline['turn_5d'] / kline['turn_20d'].clip(lower=0.01) - 1
    kline['turn_accel'] = kline['turn_accel'].replace([np.inf, -np.inf], np.nan)
    
    # Combine with return direction: turn_accel × sign(ret_5d)
    # Positive when: volume rising + price rising (bullish) or volume falling + price falling
    # Negative when: volume rising + price falling (bearish) or volume falling + price rising
    kline['raw_factor'] = kline['turn_accel'] * np.sign(kline['ret_5d'])
    
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
