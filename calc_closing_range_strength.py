"""
因子：Closing Range Strength (CRS)
ID: crs_v1

逻辑：
  Daily Range Position = (close - low) / (high - low)  ∈ [0,1]
  值接近1 = 收盘在当日高位 = 尾盘强势买入
  值接近0 = 收盘在当日低位 = 尾盘被抛售
  
  20日均值高 = 持续尾盘强势 = 机构/知情者认可 → 正 alpha
  
  经市值中性化。

方向：正向（高因子值 = 高预期收益）
Barra风格：Momentum / 微观结构
"""

import pandas as pd
import numpy as np

def calc_crs_factor(kline_path, output_path, window=20):
    """计算 Closing Range Strength 因子"""
    print("Loading kline data...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # Daily range position
    daily_range = df['high'] - df['low']
    # Avoid division by zero
    daily_range = daily_range.clip(lower=(df['close'] * 0.001))  # minimum range = 0.1% of close
    df['range_position'] = (df['close'] - df['low']) / daily_range
    df['range_position'] = df['range_position'].clip(0, 1)
    
    # 20-day rolling mean
    print("Calculating rolling CRS...")
    df['crs_raw'] = df.groupby('stock_code')['range_position'].transform(
        lambda x: x.rolling(window, min_periods=int(window * 0.7)).mean()
    )
    
    factor_df = df[['date', 'stock_code', 'crs_raw']].dropna()
    factor_df = factor_df.rename(columns={'crs_raw': 'raw_factor'})
    
    print(f"Raw factor rows: {len(factor_df)}")
    
    # Neutralize by log(amount)
    print("Neutralizing by log(amount)...")
    amt_df = df[['date', 'stock_code', 'amount']].copy()
    amt_df['log_amount_20d'] = amt_df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean().clip(lower=1))
    )
    amt_df['date'] = pd.to_datetime(amt_df['date'])
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    
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
    calc_crs_factor('data/csi1000_kline_raw.csv', 'data/factor_crs_v1.csv')
