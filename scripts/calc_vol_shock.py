#!/usr/bin/env python3
"""
Volume Shock Moderation Factor (日度近似版)
源自方正金工"适度冒险因子"的日线降维

核心逻辑：
- 检测成交量异常日（volume > MA20 + 1σ）
- 计算这些日的收益率绝对值
- 计算其近20日均值和标准差
- 合成因子值：|均值-截面均值| 的低值 = 适度反应 = 正alpha

用法：
  python3 scripts/calc_vol_shock.py
"""
import numpy as np
import pandas as pd
import sys, os

def calc_factor(kline_path='data/csi1000_kline_raw.csv', output_path=None):
    df = pd.read_csv(kline_path, parse_dates=['date'])
    # Use 'stock_code' as the stock identifier
    df['code'] = df['stock_code']
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    
    # Calculate daily log return
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Use stock_code directly as groupby key
    grp = df.groupby('stock_code')
    # Rolling stats: MA20, std20 for volume
    df['vol_ma20'] = grp['volume'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df['vol_std20'] = grp['volume'].transform(lambda x: x.rolling(20, min_periods=10).std())
    
    # Volume shock flag: volume > MA20 + 1*std
    df['vol_shock'] = (df['volume'] > df['vol_ma20'] + df['vol_std20']).astype(int)
    
    # Abs return on shock days, else NaN
    df['abs_ret'] = df['log_ret'].abs()
    df['abs_ret_shock'] = df['abs_ret'].where(df['vol_shock'] == 1)
    
    # rolling mean/std (min 3 shocks in window = realistic threshold)
    df['shock_ret_mean'] = grp['abs_ret_shock'].transform(
        lambda x: x.rolling(20, min_periods=3).mean()
    )
    # Rolling std of abs_return on shock days (20d)
    df['shock_ret_std'] = grp['abs_ret_shock'].transform(
        lambda x: x.rolling(20, min_periods=15).std()
    )
    
    # Factor: negative of standardized shock return mean (low shock_ret_mean = better)
    factor_name = 'vol_shock_mod_v1'
    df[factor_name] = -df['shock_ret_mean']  # negative: lower avg shock return = better
    
    # Market-cap neutralization
    # Use log(amount) as proxy for size
    df['log_amount'] = np.log(df['amount'].replace(0, np.nan))
    df['log_amount_20'] = grp['log_amount'].transform(lambda x: x.rolling(20).mean())
    
    result = df[['date', 'code', factor_name, 'log_amount_20']].dropna(subset=[factor_name, 'log_amount_20']).copy()
    
    # Cross-section z-score per date
    def zscore_group(g):
        mu = g[factor_name].mean()
        std = g[factor_name].std()
        g[factor_name] = (g[factor_name] - mu) / std if std > 0 else 0
        return g
    
    result = result.groupby('date', group_keys=False).apply(zscore_group)
    
    if output_path is None:
        output_path = f'data/factor_{factor_name}.csv'
    result[['date', 'code', factor_name]].rename(columns={'code':'stock_code'}).to_csv(output_path, index=False)
    print(f"✅ Factor saved to {output_path}: {len(result)} rows, {result['code'].nunique()} stocks")
    
    # Quick stats
    latest = result.groupby('code').tail(1)[factor_name]
    print(f"   Latest factor mean: {latest.mean():.4f}, std: {latest.std():.4f}")
    
    return output_path

if __name__ == '__main__':
    calc_factor()
