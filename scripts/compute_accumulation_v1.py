"""
吸筹因子 (Accumulation Signal) v1
=================================
逻辑：
- 在"高成交量+低价格变动"的日子，大资金在吸筹/派发，但不想暴露方向
- 这些日子的收益方向暗示了大资金的真实意图
- 高量低波日收益为正 → 吸筹(做多信号)
- 高量低波日收益为负 → 派发(做空信号)

构造方法：
1. 计算每日"吸筹指标" = turnover / amplitude (量能/振幅比)
   - 高比值 = 放量但价格波动小 = 吸筹/派发行为
2. 识别过去20天中吸筹指标最高的5天(top 25%)
3. 计算这些高量低波日的平均收益率方向
4. 做市值中性化

Barra: 微观结构 / 资金流
"""

import pandas as pd
import numpy as np
import sys
import os

def compute_factor():
    print("Loading data...")
    df = pd.read_csv('data/csi1000_kline_raw.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")
    
    # Calculate daily return
    df['daily_ret'] = df.groupby('stock_code')['close'].pct_change()
    
    # amplitude is already in the data (high-low)/prev_close * 100
    # turnover is already in the data (as percentage)
    
    # Calculate volume-amplitude ratio (吸筹指标)
    # High ratio = high turnover with low amplitude = stealth accumulation
    # Use turnover / (amplitude + 0.01) to avoid division by zero
    df['vol_amp_ratio'] = df['turnover'] / (df['amplitude'].clip(lower=0.1))
    
    print(f"vol_amp_ratio stats: mean={df['vol_amp_ratio'].mean():.3f}, median={df['vol_amp_ratio'].median():.3f}")
    
    # For each stock, rolling 20-day window:
    # 1. Find top-5 highest vol_amp_ratio days (stealth days)
    # 2. Calculate their average return
    window = 20
    top_k = 5  # top 25% of window
    
    results = []
    
    for stock_code, gdf in df.groupby('stock_code'):
        gdf = gdf.sort_values('date').reset_index(drop=True)
        n = len(gdf)
        
        factor_vals = []
        dates = []
        
        for i in range(window - 1, n):
            w = gdf.iloc[i - window + 1: i + 1]
            
            # Get vol_amp_ratio and daily_ret for this window
            var = w['vol_amp_ratio'].values
            rets = w['daily_ret'].values
            
            # Find indices of top-k highest vol_amp_ratio days
            valid_mask = ~(np.isnan(var) | np.isnan(rets))
            if valid_mask.sum() < top_k:
                factor_vals.append(np.nan)
                dates.append(gdf.iloc[i]['date'])
                continue
            
            # Get top-k indices by vol_amp_ratio
            valid_indices = np.where(valid_mask)[0]
            valid_var = var[valid_indices]
            top_indices = valid_indices[np.argsort(valid_var)[-top_k:]]
            
            # Average return of stealth days
            stealth_ret = np.mean(rets[top_indices])
            
            factor_vals.append(stealth_ret)
            dates.append(gdf.iloc[i]['date'])
        
        stock_result = pd.DataFrame({
            'date': dates,
            'stock_code': stock_code,
            'factor_raw': factor_vals
        })
        results.append(stock_result)
    
    factor_df = pd.concat(results, ignore_index=True)
    print(f"Factor computed: {factor_df.shape}")
    print(f"Factor raw stats: mean={factor_df['factor_raw'].mean():.6f}, std={factor_df['factor_raw'].std():.6f}")
    
    # Market cap neutralization using amount as proxy
    # Load amount for neutralization
    amt_df = df[['date', 'stock_code', 'amount']].copy()
    amt_df['date'] = pd.to_datetime(amt_df['date'])
    
    # 20-day average amount
    amt_df = amt_df.sort_values(['stock_code', 'date'])
    amt_df['avg_amount_20d'] = amt_df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    amt_df['log_amount'] = np.log(amt_df['avg_amount_20d'].clip(lower=1))
    
    factor_df = factor_df.merge(amt_df[['date', 'stock_code', 'log_amount']], 
                                 on=['date', 'stock_code'], how='left')
    
    # Cross-sectional neutralization
    def neutralize(group):
        y = group['factor_raw'].values
        x = group['log_amount'].values
        
        valid = ~(np.isnan(y) | np.isnan(x))
        if valid.sum() < 30:
            group['factor'] = np.nan
            return group
        
        y_v = y[valid]
        x_v = x[valid]
        
        # OLS: y = a + b*x + residual
        X = np.column_stack([np.ones(len(x_v)), x_v])
        try:
            beta = np.linalg.lstsq(X, y_v, rcond=None)[0]
            residuals = np.full(len(y), np.nan)
            residuals[valid] = y_v - X @ beta
            group['factor'] = residuals
        except:
            group['factor'] = np.nan
        
        return group
    
    print("Neutralizing...")
    factor_df = factor_df.groupby('date', group_keys=False).apply(neutralize)
    
    # Winsorize (MAD 3x)
    def winsorize_mad(group):
        vals = group['factor'].values
        valid = ~np.isnan(vals)
        if valid.sum() < 10:
            return group
        median = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals[valid] - median))
        if mad < 1e-10:
            return group
        upper = median + 3 * 1.4826 * mad
        lower = median - 3 * 1.4826 * mad
        group['factor'] = np.clip(vals, lower, upper)
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_mad)
    
    # Z-score standardization
    def zscore(group):
        vals = group['factor'].values
        valid = ~np.isnan(vals)
        if valid.sum() < 10:
            return group
        mean = np.nanmean(vals[valid])
        std = np.nanstd(vals[valid])
        if std < 1e-10:
            group['factor'] = 0.0
        else:
            group['factor'] = (vals - mean) / std
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(zscore)
    
    # Output
    output = factor_df[['date', 'stock_code', 'factor']].copy()
    output = output.dropna(subset=['factor'])
    output.to_csv('data/factor_accumulation_v1.csv', index=False)
    
    print(f"\nOutput saved: {output.shape}")
    print(f"Date range: {output['date'].min()} ~ {output['date'].max()}")
    print(f"Factor stats after processing:")
    print(output['factor'].describe())
    
    return output

if __name__ == '__main__':
    compute_factor()
