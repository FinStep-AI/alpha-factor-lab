"""
因子：成交额-振幅弹性 (Amount-Amplitude Elasticity)
ID: amt_amp_elasticity_v1

逻辑：
  对每只股票20日窗口内做OLS回归：log(amplitude) = α + β × log(amount/median_amount) + ε
  β = 成交额变化对振幅的敏感度（弹性）
  高β = 市场脆弱，少量资金变化就引起大幅波动 = 流动性差/价格发现弱
  低β = 市场稳健 = Quality/Liquidity代理
  
  反向使用（取负）：factor = -β → 高factor值 = 低弹性 = 更好

方向：正向（做多低弹性/高稳健性的股票）
中性化：成交额OLS中性化
Barra风格：Quality代理
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys
import os

def calc_elasticity_factor(kline_path, output_path, window=20):
    """计算成交额-振幅弹性因子"""
    print("Loading kline data...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # 计算log振幅和log相对成交额
    df['log_amp'] = np.log(df['amplitude'].clip(lower=0.01))  # 振幅最小0.01%
    
    # 每只股票的成交额标准化（相对于自身20日中位数）
    df['log_amount'] = np.log(df['amount'].clip(lower=1))
    
    print("Calculating rolling elasticity (this may take a while)...")
    
    results = []
    stocks = df['stock_code'].unique()
    total = len(stocks)
    
    for i, stock in enumerate(stocks):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i+1}/{total}...")
        
        sdf = df[df['stock_code'] == stock].copy()
        sdf = sdf.sort_values('date').reset_index(drop=True)
        
        if len(sdf) < window + 5:
            continue
        
        log_amp = sdf['log_amp'].values
        log_amt = sdf['log_amount'].values
        dates = sdf['date'].values
        
        for j in range(window - 1, len(sdf)):
            # 20日窗口
            y = log_amp[j - window + 1: j + 1]
            x = log_amt[j - window + 1: j + 1]
            
            # 去均值化x（相对成交额）
            x_centered = x - np.median(x)
            
            # 检查是否有足够变异
            if np.std(x_centered) < 1e-8 or np.std(y) < 1e-8:
                continue
            
            # OLS回归：y = a + b * x_centered
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_centered, y)
            
            results.append({
                'date': dates[j],
                'stock_code': stock,
                'elasticity': slope  # β
            })
    
    print(f"  Computed {len(results)} factor values")
    
    factor_df = pd.DataFrame(results)
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    
    # 取负：高因子值 = 低弹性 = 更稳健
    factor_df['raw_factor'] = -factor_df['elasticity']
    
    print("Neutralizing by log(amount)...")
    # 成交额中性化
    # 先merge成交额
    amt_df = df[['date', 'stock_code', 'amount']].copy()
    amt_df['log_amount_20d'] = amt_df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean().clip(lower=1))
    )
    factor_df = factor_df.merge(
        amt_df[['date', 'stock_code', 'log_amount_20d']], 
        on=['date', 'stock_code'], how='left'
    )
    
    # OLS中性化：逐截面
    def neutralize_cross_section(group):
        y = group['raw_factor'].values
        x = group['log_amount_20d'].values
        
        valid = np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 30:
            group['factor'] = np.nan
            return group
        
        y_valid = y[valid]
        x_valid = x[valid]
        
        # OLS
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
        
        # MAD winsorize at 5*MAD
        upper = med + 5 * 1.4826 * mad
        lower = med - 5 * 1.4826 * mad
        vals_clipped = np.clip(vals, lower, upper)
        
        # z-score
        mean = np.nanmean(vals_clipped[valid])
        std = np.nanstd(vals_clipped[valid])
        if std < 1e-8:
            group['factor'] = 0.0
            return group
        
        group['factor'] = (vals_clipped - mean) / std
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(mad_zscore)
    
    # 输出
    output = factor_df[['date', 'stock_code', 'factor']].dropna()
    output = output.rename(columns={'factor': 'factor_value'})
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')
    output.to_csv(output_path, index=False)
    
    print(f"Factor saved to {output_path}")
    print(f"  Rows: {len(output)}")
    print(f"  Date range: {output['date'].min()} ~ {output['date'].max()}")
    print(f"  Stocks per date (mean): {output.groupby('date')['stock_code'].count().mean():.0f}")
    
    # 检查因子分布
    print(f"\nFactor distribution:")
    print(f"  Mean: {output['factor_value'].mean():.4f}")
    print(f"  Std:  {output['factor_value'].std():.4f}")
    print(f"  Min:  {output['factor_value'].min():.4f}")
    print(f"  Max:  {output['factor_value'].max():.4f}")
    
    return output

if __name__ == '__main__':
    kline_path = 'data/csi1000_kline_raw.csv'
    output_path = 'data/factor_amt_amp_elasticity_v1.csv'
    
    calc_elasticity_factor(kline_path, output_path)
