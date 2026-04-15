"""
因子： conviction_weighted_volume (CWV)
ID: cwv_v1

逻辑：
  每日 conviction = volume × sign(pct_change)
  累计20日 conviction 差值 = sum(volume × sign(ret), 20d)
  高正值 = 持续有大量资金在上涨日买入(知情/乐观) > 下跌日卖出
  低/负值 = 下跌日资金多 = 崩盘前兆/无 conviction
  
  经市值中性化处理。

方向：正向（高因子值 = 高预期收益）
Barra风格：Momentum（作为 Volume-Price Divergence 动量子类）
"""

import pandas as pd
import numpy as np
import sys

def calc_cwv_factor(kline_path, output_path, window=20):
    """计算 Conviction-Weighted Volume 因子"""
    print("Loading kline data...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df['pct_change'] = pd.to_numeric(df['pct_change'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # Daily conviction: volume * sign(return)
    df['sign_ret'] = np.sign(df['pct_change'].fillna(0))
    df['conviction'] = df['volume'] * df['sign_ret']
    
    # Rolling sum over window days
    print("Calculating rolling conviction sum...")
    df['cwv_raw'] = df.groupby('stock_code')['conviction'].transform(
        lambda x: x.rolling(window, min_periods=int(window * 0.7)).sum()
    )
    
    # Normalize by average volume over the same window (percentile version)
    df['avg_vol'] = df.groupby('stock_code')['volume'].transform(
        lambda x: x.rolling(window, min_periods=int(window * 0.7)).mean()
    )
    df['cwv_norm'] = df['cwv_raw'] / df['avg_vol'].clip(lower=1)
    
    # Keep only valid rows
    factor_df = df[['date', 'stock_code', 'cwv_norm']].dropna()
    factor_df = factor_df.rename(columns={'cwv_norm': 'raw_factor'})
    # Keep date as datetime for merge
    if factor_df['date'].dtype == object:
        factor_df['date'] = pd.to_datetime(factor_df['date'])
    
    print(f"Raw factor rows: {len(factor_df)}")
    
    # === 中性化：成交额 OLS ===
    print("Neutralizing by log(amount)...")
    # Merge amount for neutralization
    amt_df = df[['date', 'stock_code', 'amount']].copy()
    amt_df['log_amount_20d'] = amt_df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean().clip(lower=1))
    )
    # Ensure date types match
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
            # Compute residuals for all valid points
            residuals = np.full(len(y), np.nan)
            X_full = np.column_stack([np.ones(valid.sum()), x_valid])
            residuals[valid] = y_valid - X_full @ beta
        except Exception as e:
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
    kline_path = 'data/csi1000_kline_raw.csv'
    output_path = 'data/factor_cwv_v1.csv'
    calc_cwv_factor(kline_path, output_path)
