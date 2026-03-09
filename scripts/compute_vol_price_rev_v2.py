#!/usr/bin/env python3
"""
Volume-Price Reversal Factor v2 (量价反转因子 v2)
==================================================

Factor ID: vol_price_rev_v2

Improved version: use rank-based turnover acceleration and signed returns,
then take their product. Rank-based approach reduces outlier impact.

Key improvements:
1. Rank turnover acceleration cross-sectionally (more stable)
2. Use 3d return rank instead of sign (captures magnitude)
3. Take negative product as factor (contrarian)

Neutralize by log(amount_20d).
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
KLINE_PATH = BASE / "data" / "csi1000_kline_raw.csv"
OUTPUT_PATH = BASE / "data" / "factor_vol_price_rev_v2.csv"


def main():
    print("=" * 60)
    print("Volume-Price Reversal v2 (Rank-based)")
    print("=" * 60)
    
    kline = pd.read_csv(KLINE_PATH)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date'])
    
    # 3d return
    kline['ret_3d'] = kline.groupby('stock_code')['close'].pct_change(3)
    
    # Turnover acceleration: MA3/MA20 - 1
    kline['turn_3d'] = kline.groupby('stock_code')['turnover'].transform(
        lambda x: x.rolling(3, min_periods=2).mean()
    )
    kline['turn_20d'] = kline.groupby('stock_code')['turnover'].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )
    kline['turn_accel'] = kline['turn_3d'] / kline['turn_20d'].clip(lower=0.01) - 1
    kline['turn_accel'] = kline['turn_accel'].replace([np.inf, -np.inf], np.nan)
    
    # 20d amount for neutralization
    kline['amount_20d'] = kline.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    kline['log_amount_20d'] = np.log(kline['amount_20d'].clip(lower=1))
    
    valid = kline.dropna(subset=['ret_3d', 'turn_accel', 'log_amount_20d']).copy()
    valid = valid[np.isfinite(valid['turn_accel']) & np.isfinite(valid['ret_3d'])]
    
    def winsorize_mad(s, k=3):
        med = s.median()
        mad = (s - med).abs().median()
        if mad == 0:
            return s
        return s.clip(med - k * 1.4826 * mad, med + k * 1.4826 * mad)
    
    # Cross-sectional processing
    out_records = []
    for dt, grp in valid.groupby('date'):
        if len(grp) < 200:
            continue
        
        # Rank turnover acceleration (0 to 1)
        ta_rank = grp['turn_accel'].rank(pct=True)
        
        # Rank return (0 to 1)
        ret_rank = grp['ret_3d'].rank(pct=True)
        
        # Product of ranks, centered: high-rank both = volume confirmed momentum
        # Factor = negative of (ta_rank - 0.5) * (ret_rank - 0.5)
        # This captures: high turnover accel + high return → negative (sell)
        #                high turnover accel + low return → positive (contrarian buy after sell-off with volume)
        #                low turnover accel + high return → positive (stealth rally)
        raw = -4 * (ta_rank - 0.5) * (ret_rank - 0.5)  # scale by 4 for [-1, 1] range
        
        # Neutralize by market cap
        x = grp['log_amount_20d'].values
        y = raw.values
        
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 200:
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
