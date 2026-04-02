#!/usr/bin/env python3
"""
Trend Factor — Han, Zhou, Zhu (2016) "A Trend Factor: Any Economic Gains from Using Information over Investment Horizons?"
Journal of Financial Economics, 2016.

公式:
  TREND_i,t = Σ_{L∈windows} w_L × (MA_L(close_i) / close_i,t - 1)

原始论文用 {3,5,10,20,50,100,200,...} 日均线。
考虑到我们数据从2022-10开始(~850个交易日)，用 {3,5,10,20,50} 日窗口。
权重：等权简化版 (论文的最优权重基于截面回归，需要训练期)。

关键创新点：
1. 多期限信号融合 — 同时捕获短/中/长期趋势
2. 标准化 — MA/P - 1 使信号可比
3. 市值中性化 — 截面回归去除市值影响

Source: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2600506

输出: data/factor_trend.csv (date, stock_code, factor_value)
"""

import pandas as pd
import numpy as np
import sys
import os

def compute_trend_factor(
    kline_path='data/csi1000_kline_raw.csv',
    output_path='data/factor_trend.csv',
    windows=[3, 5, 10, 20, 50],
    min_periods_frac=0.8,
    neutralize_by_cap=True
):
    print("=" * 60)
    print("Trend Factor Construction")
    print("=" * 60)
    
    # 加载数据
    df = pd.read_csv(kline_path, parse_dates=['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    print(f"Loaded {len(df)} rows, {df['stock_code'].nunique()} stocks")
    
    # 计算每个窗口的 MA_L / P - 1
    print(f"\nComputing MA signals for windows: {windows}")
    
    signals = []
    for w in windows:
        min_p = max(1, int(w * min_periods_frac))
        col_name = f'ma_signal_{w}'
        df[col_name] = df.groupby('stock_code')['close'].transform(
            lambda x: x.rolling(window=w, min_periods=min_p).mean()
        )
        df[col_name] = df[col_name] / df['close'] - 1
        signals.append(col_name)
        valid = df[col_name].notna().sum()
        print(f"  MA{w}: {valid} valid values ({valid/len(df)*100:.1f}%)")
    
    # 等权组合
    print("\nCombining signals (equal weight)...")
    df['trend_raw'] = df[signals].mean(axis=1)
    valid_raw = df['trend_raw'].notna().sum()
    print(f"  Raw trend: {valid_raw} valid values")
    
    # 市值中性化
    if neutralize_by_cap:
        print("\nNeutralizing by market cap proxy (log_amount)...")
        # 用 ln(amount) 作为市值代理
        df['log_amount'] = np.log(df['amount'].clip(lower=1))
        
        results = []
        for date, group in df.groupby('date'):
            g = group[['stock_code', 'trend_raw', 'log_amount']].dropna()
            if len(g) < 50:
                continue
            
            # 截面回归 trend_raw ~ log_amount，取残差
            x = g['log_amount'].values
            y = g['trend_raw'].values
            
            # 标准化x
            x_mean = x.mean()
            x_std = x.std()
            if x_std < 1e-10:
                continue
            x_norm = (x - x_mean) / x_std
            
            # OLS
            X = np.column_stack([np.ones(len(x_norm)), x_norm])
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                residuals = y - X @ beta
            except:
                continue
            
            g = g.copy()
            g['factor_value'] = residuals
            g['date'] = date
            results.append(g[['date', 'stock_code', 'factor_value']])
        
        result_df = pd.concat(results, ignore_index=True)
    else:
        df['factor_value'] = df['trend_raw']
        result_df = df[['date', 'stock_code', 'factor_value']].dropna()
    
    # 截面标准化 (每天z-score)
    print("Cross-sectional standardization...")
    result_df['factor_value'] = result_df.groupby('date')['factor_value'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 1e-10 else 0
    )
    
    # 输出
    result_df = result_df.sort_values(['date', 'stock_code']).reset_index(drop=True)
    result_df.to_csv(output_path, index=False)
    
    print(f"\nOutput: {output_path}")
    print(f"  Rows: {len(result_df)}")
    print(f"  Date range: {result_df['date'].min()} to {result_df['date'].max()}")
    print(f"  Trading days: {result_df['date'].nunique()}")
    print(f"  Stocks per day: {result_df.groupby('date')['stock_code'].count().mean():.0f}")
    
    # 统计描述
    print(f"\n  Factor stats:")
    print(f"    Mean:   {result_df['factor_value'].mean():.4f}")
    print(f"    Std:    {result_df['factor_value'].std():.4f}")
    print(f"    Skew:   {result_df['factor_value'].skew():.4f}")
    print(f"    Kurt:   {result_df['factor_value'].kurtosis():.4f}")
    
    return result_df

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    compute_trend_factor()
