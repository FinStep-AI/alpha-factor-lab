#!/usr/bin/env python3
"""
MAX5 Lottery Factor (彩票效应因子)
==================================

Barra Style: 尾部风险/反转
Factor ID: max5_lottery_v1

Logic:
------
MAX5 = average of the 5 highest daily returns over past 20 days
Factor = -MAX5 (negative: avoid lottery-like stocks)

From Bali, Cakici & Whitelaw (2011):
Stocks with high MAX (extreme positive returns) are overpriced due to 
lottery preferences, and subsequently underperform.

This is the "upside" complement to tail_risk_cvar_v1 (which looks at downside).
Together they capture both tails: avoid extreme positive AND extreme negative returns.

Key difference from CVaR: CVaR looks at worst returns (bottom), MAX5 looks at best returns (top).

Neutralize by log(amount_20d).
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
KLINE_PATH = BASE / "data" / "csi1000_kline_raw.csv"
OUTPUT_PATH = BASE / "data" / "factor_max5_lottery_v1.csv"


def main():
    print("=" * 60)
    print("MAX5 Lottery Factor (彩票效应因子)")
    print("=" * 60)
    
    kline = pd.read_csv(KLINE_PATH)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date'])
    print(f"Kline: {len(kline)} rows")
    
    # Daily returns
    kline['ret'] = kline.groupby('stock_code')['close'].pct_change()
    
    # For each stock on each day: average of top 5 returns in last 20 days
    results_list = []
    for sc, grp in kline.groupby('stock_code'):
        grp = grp.copy().reset_index(drop=True)
        rets = grp['ret'].values
        dates = grp['date'].values
        
        max5_vals = np.full(len(rets), np.nan)
        for i in range(19, len(rets)):
            window = rets[max(0, i-19):i+1]
            valid = window[~np.isnan(window)]
            if len(valid) >= 15:
                # Top 5 returns
                top5 = np.sort(valid)[-5:]
                max5_vals[i] = np.mean(top5)
        
        out = pd.DataFrame({
            'date': dates,
            'stock_code': sc,
            'max5': max5_vals
        })
        results_list.append(out)
    
    df = pd.concat(results_list, ignore_index=True)
    df = df.dropna(subset=['max5'])
    
    # Factor: NEGATIVE of MAX5 (avoid lottery stocks)
    df['raw_factor'] = -df['max5']
    
    print(f"Raw factor rows: {len(df)}")
    print(f"MAX5 stats:\n{df['max5'].describe()}")
    
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
        return s.clip(med - k * 1.4826 * mad, med + k * 1.4826 * mad)
    
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
