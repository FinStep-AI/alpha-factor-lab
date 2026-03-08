#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realized Skewness Factor v2 - Multiple Variants

尝试多种变体:
1. 原始公式 (Amaya et al.): sqrt(N) * sum(r^3) / (sum(r^2))^(3/2)
2. 不同窗口: 10, 20, 40, 60
3. 正向/反向
4. 调频: 5d / 20d forward
"""

import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def compute_realized_skewness_amaya(kline_path, output_path, window=20):
    """
    Amaya et al. 原始公式:
    RSkew_t = sqrt(N) * sum_{i=1}^{N} r_i^3 / (sum_{i=1}^{N} r_i^2)^{3/2}
    
    其中 r_i 是日内(或日)收益率, N 是窗口内观测数
    """
    print(f"Computing Realized Skewness (Amaya formula, window={window})...")
    
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # Daily returns
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    
    # Amaya formula components
    def amaya_skew(group):
        ret = group['ret']
        r2 = (ret ** 2).rolling(window, min_periods=window//2).sum()
        r3 = (ret ** 3).rolling(window, min_periods=window//2).sum()
        n = ret.rolling(window, min_periods=window//2).count()
        
        rskew = np.sqrt(n) * r3 / (r2 ** 1.5)
        rskew = rskew.replace([np.inf, -np.inf], np.nan)
        return rskew
    
    df['raw_skew'] = df.groupby('stock_code', group_keys=False).apply(amaya_skew)
    
    # Market cap neutralization using log(20d avg amount)
    df['log_amount'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    
    def neutralize(group):
        y = group['raw_skew']
        x = group['log_amount']
        mask = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            return pd.Series(np.nan, index=group.index)
        y_c, x_c = y[mask].values, x[mask].values
        X = np.column_stack([np.ones(len(x_c)), x_c])
        try:
            beta = np.linalg.lstsq(X, y_c, rcond=None)[0]
            res = y_c - X @ beta
            out = pd.Series(np.nan, index=group.index)
            out[mask] = res
            return out
        except:
            return pd.Series(np.nan, index=group.index)
    
    df['factor'] = df.groupby('date', group_keys=False).apply(neutralize)
    
    # Winsorize MAD 3x
    def winsorize_mad(group):
        v = group['factor']
        m = v.notna()
        if m.sum() < 10: return v
        c = v[m]
        med = c.median()
        mad = np.median(np.abs(c - med))
        if mad < 1e-10: return v
        lo = med - 3 * 1.4826 * mad
        hi = med + 3 * 1.4826 * mad
        return v.clip(lo, hi)
    
    df['factor'] = df.groupby('date', group_keys=False).apply(winsorize_mad)
    
    # Z-score
    def zscore(group):
        v = group['factor']
        m = v.notna()
        if m.sum() < 10: return v
        mu, s = v[m].mean(), v[m].std()
        if s < 1e-10: return v * 0
        return (v - mu) / s
    
    df['factor'] = df.groupby('date', group_keys=False).apply(zscore)
    
    output = df[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    output = output.rename(columns={'factor': 'factor_value'})
    output.to_csv(output_path, index=False)
    
    print(f"  Saved {len(output):,} rows, {output['date'].nunique()} dates")
    return output


def compute_negative_skewness(kline_path, output_path, window=20):
    """负偏度版: -1 * realized_skewness (做多低偏度)"""
    print(f"Computing NEGATIVE Realized Skewness (window={window})...")
    
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    
    def amaya_skew(group):
        ret = group['ret']
        r2 = (ret ** 2).rolling(window, min_periods=window//2).sum()
        r3 = (ret ** 3).rolling(window, min_periods=window//2).sum()
        n = ret.rolling(window, min_periods=window//2).count()
        rskew = np.sqrt(n) * r3 / (r2 ** 1.5)
        return rskew.replace([np.inf, -np.inf], np.nan)
    
    df['raw_skew'] = df.groupby('stock_code', group_keys=False).apply(amaya_skew)
    # Take NEGATIVE
    df['raw_skew'] = -df['raw_skew']
    
    df['log_amount'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    
    def neutralize(group):
        y = group['raw_skew']
        x = group['log_amount']
        mask = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            return pd.Series(np.nan, index=group.index)
        y_c, x_c = y[mask].values, x[mask].values
        X = np.column_stack([np.ones(len(x_c)), x_c])
        try:
            beta = np.linalg.lstsq(X, y_c, rcond=None)[0]
            res = y_c - X @ beta
            out = pd.Series(np.nan, index=group.index)
            out[mask] = res
            return out
        except:
            return pd.Series(np.nan, index=group.index)
    
    df['factor'] = df.groupby('date', group_keys=False).apply(neutralize)
    
    def winsorize_mad(group):
        v = group['factor']
        m = v.notna()
        if m.sum() < 10: return v
        c = v[m]; med = c.median(); mad = np.median(np.abs(c - med))
        if mad < 1e-10: return v
        return v.clip(med - 3*1.4826*mad, med + 3*1.4826*mad)
    
    df['factor'] = df.groupby('date', group_keys=False).apply(winsorize_mad)
    
    def zscore(group):
        v = group['factor']; m = v.notna()
        if m.sum() < 10: return v
        mu, s = v[m].mean(), v[m].std()
        if s < 1e-10: return v * 0
        return (v - mu) / s
    
    df['factor'] = df.groupby('date', group_keys=False).apply(zscore)
    
    output = df[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    output = output.rename(columns={'factor': 'factor_value'})
    output.to_csv(output_path, index=False)
    print(f"  Saved {len(output):,} rows, {output['date'].nunique()} dates")
    return output


if __name__ == "__main__":
    kline = "data/csi1000_kline_raw.csv"
    
    # Variant 1: Amaya formula, positive (正偏度做多)
    compute_realized_skewness_amaya(kline, "data/factor_rskew_pos_20d.csv", window=20)
    
    # Variant 2: Amaya formula, negative (负偏度做多) 
    compute_negative_skewness(kline, "data/factor_rskew_neg_20d.csv", window=20)
    
    # Variant 3: Longer window 60d
    compute_realized_skewness_amaya(kline, "data/factor_rskew_pos_60d.csv", window=60)
    compute_negative_skewness(kline, "data/factor_rskew_neg_60d.csv", window=60)
    
    print("\n✅ All variants computed!")
