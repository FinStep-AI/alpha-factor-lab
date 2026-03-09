#!/usr/bin/env python3
"""
Volume-Price Reversal Factor (量价反转因子)
============================================

Factor ID: vol_price_reversal_v1

Core insight: When turnover accelerates in the direction of price, 
the move is exhausted → contrarian signal.

Factor = -(turnover_acceleration × sign(return))
       = NEGATIVE of volume-confirmed momentum

Test variations:
- v1: 3d/10d windows, 5d forward
- v2: 5d/20d windows, 5d forward (already tested)
- v3: 3d/10d windows, 20d forward

Neutralize by log(amount_20d).
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
KLINE_PATH = BASE / "data" / "csi1000_kline_raw.csv"


def compute_factor(short_w=3, long_w=10, ret_w=3, output_suffix=""):
    kline = pd.read_csv(KLINE_PATH)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date'])
    
    kline['ret'] = kline.groupby('stock_code')['close'].pct_change(ret_w)
    
    kline['turn_short'] = kline.groupby('stock_code')['turnover'].transform(
        lambda x: x.rolling(short_w, min_periods=max(2, short_w-1)).mean()
    )
    kline['turn_long'] = kline.groupby('stock_code')['turnover'].transform(
        lambda x: x.rolling(long_w, min_periods=max(5, long_w-2)).mean()
    )
    
    kline['turn_accel'] = kline['turn_short'] / kline['turn_long'].clip(lower=0.01) - 1
    kline['turn_accel'] = kline['turn_accel'].replace([np.inf, -np.inf], np.nan)
    
    # Factor: NEGATIVE of volume-price confirmation
    # When turnover rises + price rises → contrarian → sell (negative factor)
    kline['raw_factor'] = -(kline['turn_accel'] * np.sign(kline['ret']))
    
    kline['amount_20d'] = kline.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    kline['log_amount_20d'] = np.log(kline['amount_20d'].clip(lower=1))
    
    valid = kline.dropna(subset=['raw_factor', 'log_amount_20d']).copy()
    valid = valid[np.isfinite(valid['raw_factor'])]
    
    def winsorize_mad(s, k=3):
        med = s.median()
        mad = (s - med).abs().median()
        if mad == 0:
            return s
        return s.clip(med - k * 1.4826 * mad, med + k * 1.4826 * mad)
    
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
    
    out_path = BASE / f"data/factor_vol_price_rev{output_suffix}.csv"
    result.to_csv(out_path, index=False)
    print(f"\n[{output_suffix}] short={short_w}d long={long_w}d ret={ret_w}d → {len(result)} rows, {result['date'].min()} ~ {result['date'].max()}")
    return str(out_path)


if __name__ == "__main__":
    # Test multiple window combinations
    paths = {}
    for short_w, long_w, ret_w, suffix in [
        (3, 10, 3, "_3_10_3"),
        (3, 10, 5, "_3_10_5"),
        (3, 20, 3, "_3_20_3"),
        (5, 20, 3, "_5_20_3"),
        (5, 10, 5, "_5_10_5"),
    ]:
        p = compute_factor(short_w, long_w, ret_w, suffix)
        paths[suffix] = p
    
    print("\nAll factor files saved. Run backtest on each.")
