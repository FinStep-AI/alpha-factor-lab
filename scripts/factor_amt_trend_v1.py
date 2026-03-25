#!/usr/bin/env python3
"""
因子：成交额趋势斜率 (Amount Trend Slope, ATS)
公式：对每只股票过去20日log(amount)做OLS线性回归，取斜率
       → 成交额OLS中性化 + MAD winsorize + z-score
方向：正向（成交额加速放大 → 关注度上升 → 后续正超额）
Barra风格：Sentiment / Attention

逻辑：
- 成交额持续放大 = 市场关注度持续提升
- 与turnover_level不同：那个看绝对水平，这个看变化趋势/斜率
- 与vol_surge(成交量激增)不同：那个看点对点ratio，这个看线性趋势拟合
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys, warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

def compute_ols_slope(series):
    """对series做OLS线性回归，返回斜率"""
    n = len(series)
    if n < 10:
        return np.nan
    y = series.values
    mask = np.isfinite(y)
    if mask.sum() < 10:
        return np.nan
    x = np.arange(n, dtype=float)
    y_clean = np.where(mask, y, 0)
    # OLS: slope = cov(x,y) / var(x)
    x_m = x[mask]
    y_m = y_clean[mask]
    n_valid = len(x_m)
    x_mean = x_m.mean()
    y_mean = y_m.mean()
    slope = np.sum((x_m - x_mean) * (y_m - y_mean)) / (np.sum((x_m - x_mean)**2) + 1e-12)
    return slope

def main():
    print("Loading data...")
    kline = pd.read_csv(DATA_DIR / 'csi1000_kline_raw.csv')
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    print(f"Data: {kline.shape[0]} rows, {kline['stock_code'].nunique()} stocks")
    print(f"Date range: {kline['date'].min()} ~ {kline['date'].max()}")
    
    # 计算log(amount)
    kline['log_amount'] = np.log(kline['amount'].clip(lower=1))
    
    # 计算20日成交额趋势斜率
    print("Computing 20-day amount trend slope...")
    kline['amt_trend_slope'] = (
        kline.groupby('stock_code')['log_amount']
        .transform(lambda s: s.rolling(20, min_periods=15).apply(compute_ols_slope, raw=False))
    )
    
    # 计算20日成交额均值（用于中性化）
    kline['log_amount_20d'] = kline.groupby('stock_code')['log_amount'].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    
    # 截面处理：成交额OLS中性化 + MAD winsorize + z-score
    print("Cross-sectional neutralization...")
    results = []
    for date, group in kline.groupby('date'):
        df = group[['stock_code', 'amt_trend_slope', 'log_amount_20d']].dropna()
        if len(df) < 50:
            continue
        
        raw = df['amt_trend_slope'].values
        x = df['log_amount_20d'].values
        
        # OLS中性化 (去成交额)
        X = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(X, raw, rcond=None)[0]
            residuals = raw - X @ beta
        except:
            continue
        
        # MAD winsorize
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))
        if mad < 1e-12:
            continue
        clipped = np.clip(residuals, med - 5 * 1.4826 * mad, med + 5 * 1.4826 * mad)
        
        # z-score
        std = clipped.std()
        if std < 1e-12:
            continue
        z = (clipped - clipped.mean()) / std
        
        for i, (_, row) in enumerate(df.iterrows()):
            results.append({
                'date': date,
                'stock_code': row['stock_code'],
                'factor_value': z[i]
            })
    
    result_df = pd.DataFrame(results)
    print(f"Factor computed: {result_df.shape[0]} rows, {result_df['date'].nunique()} dates")
    
    # 统计
    print(f"\nFactor stats:")
    print(f"  Mean: {result_df['factor_value'].mean():.4f}")
    print(f"  Std:  {result_df['factor_value'].std():.4f}")
    print(f"  Min:  {result_df['factor_value'].min():.4f}")
    print(f"  Max:  {result_df['factor_value'].max():.4f}")
    
    output_path = DATA_DIR / 'factor_amt_trend_v1.csv'
    result_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    return result_df

if __name__ == '__main__':
    main()
