#!/usr/bin/env python3
"""
Salience Theory (ST) Factor — 突显理论因子
Based on: Cosemans & Frehen (2021) "Salience Theory and Stock Prices", JFE
         Song, Zhang & Tang (2026) "Salience Theory and Stock Return", CJE (A股验证)

Core idea (Bordalo, Gennaioli & Shleifer 2012):
- Investors overweight "salient" past returns (those that stand out vs market)
- Salience function: σ(r_i, r_m) = |r_i - r_m| / (|r_i| + |r_m| + θ)
  where θ > 0 avoids division by zero (Cosemans & Frehen use θ = 0.1)
- Upside salience (stock >> market): investors get excited → overweight
- Downside salience (stock << market): investors panic → underweight
- ST value: weighted sum where salient upside returns get higher weight
- HIGH ST stocks are OVERVALUED → lower future returns (negative predictability)

Construction (following Cosemans & Frehen 2021):
1. For each stock-day, compute salience weight: σ(r_i,t, r_m,t)
2. Classify as upside (r_i > r_m) or downside (r_i < r_m)
3. Sum salience-weighted upside returns - sum salience-weighted downside returns
   over trailing window (we use 20 days)
4. Higher ST = more salient upside = overvalued → expect lower future returns
5. Use NEGATIVE ST as factor (high factor value = low ST = undervalued = buy)

Neutralization: market-cap OLS neutralization + MAD winsorize + z-score
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')

def compute_salience_factor(kline_path, returns_path, output_path, window=20, theta=0.1):
    """Compute Salience Theory factor for CSI1000 universe."""
    
    print(f"Loading data...")
    kline = pd.read_csv(kline_path, parse_dates=['date'])
    returns = pd.read_csv(returns_path, parse_dates=['date'])
    
    # Merge to get amount for neutralization
    df = returns.merge(kline[['date','stock_code','amount','close']], on=['date','stock_code'], how='left')
    
    # Compute market return (equal-weighted average of all stocks per day)
    mkt_ret = df.groupby('date')['return'].mean().rename('mkt_return')
    df = df.merge(mkt_ret, on='date', how='left')
    
    print(f"Computing salience weights for {len(df)} stock-days...")
    
    # Salience function: σ(r_i, r_m) = |r_i - r_m| / (|r_i| + |r_m| + θ)
    r_i = df['return'].values
    r_m = df['mkt_return'].values
    
    salience = np.abs(r_i - r_m) / (np.abs(r_i) + np.abs(r_m) + theta)
    
    # Upside: r_i > r_m (stock beats market)
    upside_mask = r_i > r_m
    downside_mask = r_i < r_m
    
    # Salience-weighted upside return: σ * r_i * I(r_i > r_m)
    df['sw_up'] = salience * r_i * upside_mask
    # Salience-weighted downside return: σ * |r_i| * I(r_i < r_m)  [use absolute for symmetry]
    df['sw_down'] = salience * np.abs(r_i) * downside_mask
    
    # Sort for rolling computation
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    print(f"Rolling {window}-day ST computation...")
    
    # Rolling sum of salience-weighted upside and downside
    results = []
    for code, grp in df.groupby('stock_code'):
        grp = grp.sort_values('date')
        sw_up_roll = grp['sw_up'].rolling(window, min_periods=int(window*0.8)).sum()
        sw_down_roll = grp['sw_down'].rolling(window, min_periods=int(window*0.8)).sum()
        
        # ST = upside salience - downside salience
        # Higher ST = more salient upside events = overvalued
        st_raw = sw_up_roll - sw_down_roll
        
        tmp = grp[['date','stock_code']].copy()
        tmp['st_raw'] = st_raw.values
        tmp['log_amount_20d'] = grp['amount'].rolling(20, min_periods=10).mean().apply(lambda x: np.log(x+1) if pd.notna(x) else np.nan).values
        results.append(tmp)
    
    factor_df = pd.concat(results, ignore_index=True)
    factor_df = factor_df.dropna(subset=['st_raw', 'log_amount_20d'])
    
    print(f"Neutralizing and standardizing...")
    
    # Use NEGATIVE ST as factor (low ST = undervalued = buy signal)
    factor_df['neg_st'] = -factor_df['st_raw']
    
    # Market-cap (amount) OLS neutralization per day
    from numpy.linalg import lstsq
    
    def neutralize_and_standardize(group):
        y = group['neg_st'].values
        x = group['log_amount_20d'].values
        
        valid = np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 30:
            group['factor_value'] = np.nan
            return group
        
        y_v = y[valid]
        x_v = x[valid]
        X = np.column_stack([np.ones(len(x_v)), x_v])
        beta, _, _, _ = lstsq(X, y_v, rcond=None)
        residuals = np.full(len(y), np.nan)
        residuals[valid] = y_v - X @ beta
        
        # MAD winsorize
        med = np.nanmedian(residuals)
        mad = np.nanmedian(np.abs(residuals - med))
        if mad < 1e-10:
            group['factor_value'] = 0.0
            return group
        
        upper = med + 3 * 1.4826 * mad
        lower = med - 3 * 1.4826 * mad
        residuals = np.clip(residuals, lower, upper)
        
        # Z-score
        std = np.nanstd(residuals)
        if std < 1e-10:
            group['factor_value'] = 0.0
            return group
        
        group['factor_value'] = (residuals - np.nanmean(residuals)) / std
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(neutralize_and_standardize)
    
    # Clip extreme values
    factor_df['factor_value'] = factor_df['factor_value'].clip(-3, 3)
    
    # Output
    output = factor_df[['date','stock_code','factor_value']].dropna()
    output = output.sort_values(['date','stock_code']).reset_index(drop=True)
    
    print(f"Output: {len(output)} rows, date range: {output['date'].min()} to {output['date'].max()}")
    print(f"Stats: mean={output['factor_value'].mean():.4f}, std={output['factor_value'].std():.4f}")
    
    output.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    return output


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    kline_path = os.path.join(base, 'data', 'csi1000_kline_raw.csv')
    returns_path = os.path.join(base, 'data', 'csi1000_returns.csv')
    output_path = os.path.join(base, 'data', 'factor_salience_theory_v1.csv')
    
    compute_salience_factor(kline_path, returns_path, output_path, window=20, theta=0.1)
