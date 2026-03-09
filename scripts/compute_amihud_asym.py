#!/usr/bin/env python3
"""
Amihud Asymmetry Factor (Amihud不对称性因子)
=============================================

Barra Style: Liquidity / Microstructure
Factor ID: amihud_asym_v1

Logic:
------
Split Amihud illiquidity into up-day and down-day components:
  - Amihud_up = mean(|ret|/amount on up days, 20d)  
  - Amihud_down = mean(|ret|/amount on down days, 20d)
  - Asymmetry = Amihud_down - Amihud_up

Interpretation:
- High asymmetry (down more illiquid) → selling pressure on thin volume → distress
- Low asymmetry (up more illiquid) → buying on thin volume → accumulation?

Hypothesis: Stocks where DOWN moves are more illiquid are under selling pressure 
and will continue to underperform (momentum-like). OR the reverse: they're oversold.
Test both directions.

Alternative: Use the RATIO instead of difference for better scale invariance.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
KLINE_PATH = BASE / "data" / "csi1000_kline_raw.csv"
OUTPUT_PATH = BASE / "data" / "factor_amihud_asym_v1.csv"


def main():
    print("=" * 60)
    print("Amihud Asymmetry Factor")
    print("=" * 60)
    
    kline = pd.read_csv(KLINE_PATH)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date'])
    print(f"Kline: {len(kline)} rows")
    
    # Daily returns
    kline['ret'] = kline.groupby('stock_code')['close'].pct_change()
    
    # Amihud = |ret| / amount (in 亿)
    kline['amihud'] = kline['ret'].abs() / (kline['amount'].clip(lower=1) / 1e8)
    kline['amihud'] = kline['amihud'].replace([np.inf, -np.inf], np.nan)
    
    # Separate up and down day amihud
    kline['amihud_up'] = np.where(kline['ret'] > 0, kline['amihud'], np.nan)
    kline['amihud_down'] = np.where(kline['ret'] < 0, kline['amihud'], np.nan)
    
    # 20d rolling mean (need at least 5 up days and 5 down days)
    def rolling_nanmean(series, window=20, min_count=5):
        """Rolling mean ignoring NaN, requiring min_count valid values."""
        result = []
        for sc, grp in kline.groupby('stock_code'):
            vals = grp[series].values
            out = np.full(len(vals), np.nan)
            for i in range(window - 1, len(vals)):
                window_vals = vals[max(0, i - window + 1):i + 1]
                valid = window_vals[~np.isnan(window_vals)]
                if len(valid) >= min_count:
                    out[i] = np.mean(valid)
            result.append(pd.Series(out, index=grp.index))
        return pd.concat(result)
    
    # This is slow with the loop above, let me use a faster approach
    print("Computing rolling Amihud components...")
    
    # Use expanding with shift for efficiency
    results_list = []
    for sc, grp in kline.groupby('stock_code'):
        grp = grp.copy()
        # Rolling sum and count for up/down
        up_sum = grp['amihud_up'].rolling(20, min_periods=5).sum()
        up_count = grp['amihud_up'].rolling(20, min_periods=5).count()
        down_sum = grp['amihud_down'].rolling(20, min_periods=5).sum()
        down_count = grp['amihud_down'].rolling(20, min_periods=5).count()
        
        grp['mean_amihud_up'] = up_sum / up_count
        grp['mean_amihud_down'] = down_sum / down_count
        
        # Require at least 5 observations each
        grp.loc[up_count < 5, 'mean_amihud_up'] = np.nan
        grp.loc[down_count < 5, 'mean_amihud_down'] = np.nan
        
        # Asymmetry: log ratio (down/up) — scale invariant
        ratio = grp['mean_amihud_down'] / grp['mean_amihud_up'].clip(lower=1e-10)
        grp['asym'] = np.log(ratio.clip(lower=0.01))
        
        results_list.append(grp[['date', 'stock_code', 'asym', 'amount']])
    
    df = pd.concat(results_list, ignore_index=True)
    df = df.dropna(subset=['asym'])
    print(f"Valid rows after asymmetry: {len(df)}")
    
    # 20d amount for neutralization
    kline_amt = kline.groupby('stock_code').apply(
        lambda g: g.assign(log_amount_20d=np.log(g['amount'].rolling(20, min_periods=10).mean().clip(lower=1)))
    ).reset_index(drop=True)
    
    df = df.merge(
        kline_amt[['date', 'stock_code', 'log_amount_20d']],
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
        
        raw = winsorize_mad(grp['asym'])
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
    print(f"Stats:\n{result['factor_value'].describe()}")


if __name__ == "__main__":
    main()
