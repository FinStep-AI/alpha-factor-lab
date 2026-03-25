"""
因子：换手率偏度 (Turnover Skewness)
ID: turnover_skew_v1

逻辑：
  过去20日换手率分布的偏度。
  低偏度(取负) = 换手率分布均匀 = 稳定持续关注 = Quality
  高偏度 = 脉冲式关注（偶尔爆量）= Sentiment/投机

方向：反向（低偏度=高因子值=高预期收益）
中性化：成交额OLS中性化
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
import sys

def calc_turnover_skew(kline_path, output_path, window=20):
    print("Loading kline data...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    print("Calculating rolling turnover skewness...")
    
    # 计算换手率偏度
    def rolling_skew(x):
        return x.rolling(window, min_periods=int(window * 0.7)).skew()
    
    df['turnover_skew'] = df.groupby('stock_code')['turnover'].transform(rolling_skew)
    
    # 取负：低偏度=高因子值
    df['raw_factor'] = -df['turnover_skew']
    
    # 准备成交额中性化
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean().clip(lower=1))
    )
    
    print("Cross-section neutralization...")
    factor_df = df[['date', 'stock_code', 'raw_factor', 'log_amount_20d']].dropna().copy()
    
    def process_date(group):
        y = group['raw_factor'].values
        x = group['log_amount_20d'].values
        
        valid = np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 30:
            group['factor'] = np.nan
            return group
        
        X = np.column_stack([np.ones(valid.sum()), x[valid]])
        try:
            beta = np.linalg.lstsq(X, y[valid], rcond=None)[0]
            residuals = np.full(len(y), np.nan)
            residuals[valid] = y[valid] - X @ beta
        except:
            group['factor'] = np.nan
            return group
        
        vals = residuals[valid]
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        if mad < 1e-8:
            group['factor'] = 0.0
            return group
        
        bound = 5 * 1.4826 * mad
        residuals = np.clip(residuals, med - bound, med + bound)
        
        vals_valid = residuals[valid]
        mean = np.mean(vals_valid)
        std = np.std(vals_valid)
        if std < 1e-8:
            group['factor'] = 0.0
            return group
        
        group['factor'] = (residuals - mean) / std
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(process_date)
    
    output = factor_df[['date', 'stock_code', 'factor']].dropna().copy()
    output = output.rename(columns={'factor': 'factor_value'})
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')
    output.to_csv(output_path, index=False)
    
    print(f"Factor saved to {output_path}")
    print(f"  Rows: {len(output)}")
    print(f"  Date range: {output['date'].min()} ~ {output['date'].max()}")
    print(f"  Stocks per date: {output.groupby('date')['stock_code'].count().mean():.0f}")
    print(f"  Mean: {output['factor_value'].mean():.4f}, Std: {output['factor_value'].std():.4f}")

if __name__ == '__main__':
    calc_turnover_skew('data/csi1000_kline_raw.csv', 'data/factor_turnover_skew_v1.csv')
