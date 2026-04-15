"""
因子: One-Sided Range Indicator (OSRI)
ID: osri_v1

逻辑:
  将日波动分解为向上和向下的两个分量:
  up_extreme = (high - prev_close) / prev_close   (向上跳空幅度)
  down_extreme = (low - prev_close) / prev_close  (向下跳空幅度)
  
  OSRI = down_extreme / (up_extreme - down_extreme)  当 up_extreme ≠ down_extreme
      = up_extreme 为正值时表示当日下跌压力更大
      = 负值时表示当日上涨压力更大
  
  20日均值: 持续向上(负OSRI)或向下(正OSRI)的倾向。
  
  方向分析:
  - 持续正OSRI (上涨压力 > 下跌压力) = 日内上涨更猛 = positive alpha?
  - 持续负OSRI = 日内下跌更猛 = negative alpha?
  
  假设中证1000: 日内持续上涨(负OSRI)的股票有更强的上升趋势力 = 正向信号

方向: 正向（低/负 OSRI = 高预期收益）
Barra风格: 趋势/动量
"""

import pandas as pd
import numpy as np
import sys

def calc_osri_factor(kline_path, output_path, window=20):
    """计算 One-Sided Range Indicator 因子"""
    print("Loading kline data...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # Prev close
    df['prev_close'] = df.groupby('stock_code')['close'].shift(1)
    
    # Upward and downward extremes relative to prev close
    df['up_extreme'] = (df['high'] - df['prev_close']) / df['prev_close'].clip(lower=0.01)
    df['down_extreme'] = (df['low'] - df['prev_close']) / df['prev_close'].clip(lower=0.01)
    
    # OSRI = ratio of downward to total deviation from prev_close
    # When down_extreme >>> up_extreme, OSRI ~ 1 (strong downward intraday force)
    # When up_extreme >>> down_extreme, OSRI ~ -1 (strong upward intraday force)
    total_dev = df['up_extreme'] - df['down_extreme']
    # Avoid division
    total_dev = total_dev.where(np.abs(total_dev) > 1e-8, np.sign(total_dev) * 1e-8)
    
    df['osri'] = df['down_extreme'] / total_dev
    df['osri'] = df['osri'].clip(-5, 5)  # Clip extreme values
    
    # 20-day rolling mean
    print("Calculating rolling OSRI...")
    df['osri_raw'] = df.groupby('stock_code')['osri'].transform(
        lambda x: x.rolling(window, min_periods=int(window * 0.7)).mean()
    )
    
    factor_df = df[['date', 'stock_code', 'osri_raw']].dropna()
    factor_df = factor_df.rename(columns={'osri_raw': 'raw_factor'})
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    
    print(f"Raw factor rows: {len(factor_df)}")
    print(f"Raw factor stats: mean={factor_df['raw_factor'].mean():.4f}, std={factor_df['raw_factor'].std():.4f}")
    
    # Neutralize by log(amount)
    print("Neutralizing by log(amount)...")
    amt_df = df[['date', 'stock_code', 'amount']].copy()
    amt_df['log_amount_20d'] = amt_df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean().clip(lower=1))
    )
    
    factor_df = factor_df.merge(
        amt_df[['date', 'stock_code', 'log_amount_20d']],
        on=['date', 'stock_code'], how='left'
    )
    
    def neutralize_cross_section(group):
        y = group['raw_factor'].values.astype(float)
        x = group['log_amount_20d'].values.astype(float)
        
        valid = np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 30:
            group['factor'] = np.nan
            return group
        
        y_valid = y[valid]
        x_valid = x[valid]
        
        X = np.column_stack([np.ones(len(x_valid)), x_valid])
        try:
            beta = np.linalg.lstsq(X, y_valid, rcond=None)[0]
            residuals = np.full(len(y), np.nan)
            X_full = np.column_stack([np.ones(valid.sum()), x_valid])
            residuals[valid] = y_valid - X_full @ beta
        except:
            group['factor'] = np.nan
            return group
        
        group['factor'] = residuals
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(neutralize_cross_section)
    
    # MAD winsorize + z-score
    def mad_zscore(group):
        vals = group['factor'].values
        valid = np.isfinite(vals)
        if valid.sum() < 30:
            group['factor'] = np.nan
            return group
        
        med = np.nanmedian(vals[valid])
        mad = np.nanmedian(np.abs(vals[valid] - med))
        if mad < 1e-8:
            group['factor'] = 0.0
            return group
        
        upper = med + 5 * 1.4826 * mad
        lower = med - 5 * 1.4826 * mad
        vals_clipped = np.clip(vals, lower, upper)
        
        mean = np.nanmean(vals_clipped[valid])
        std = np.nanstd(vals_clipped[valid])
        if std < 1e-8:
            group['factor'] = 0.0
            return group
        
        group['factor'] = (vals_clipped - mean) / std
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(mad_zscore)
    
    # Output
    output = factor_df[['date', 'stock_code', 'factor']].dropna()
    output = output.rename(columns={'factor': 'factor_value'})
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')
    output.to_csv(output_path, index=False)
    
    print(f"\nFactor saved to {output_path}")
    print(f"  Rows: {len(output)}")
    print(f"  Date range: {output['date'].min()} ~ {output['date'].max()}")
    print(f"  Stocks per date (mean): {output.groupby('date')['stock_code'].count().mean():.0f}")
    print(f"\nFactor distribution:")
    print(f"  Mean: {output['factor_value'].mean():.4f}")
    print(f"  Std:  {output['factor_value'].std():.4f}")
    
    return output

if __name__ == '__main__':
    calc_osri_factor('data/csi1000_kline_raw.csv', 'data/factor_osri_v1.csv')
