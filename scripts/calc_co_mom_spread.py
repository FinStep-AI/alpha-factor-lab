#!/usr/bin/env python3
"""
因子: Close-Open Momentum Spread (收盘-开盘动量差) v1

理论基础: 
- Lou, Polk & Skouras (2019) "A Tug of War: Overnight vs Intraday Returns"
- A股集合竞价由散户主导(开盘价噪声更多)，收盘连续竞价机构比例更高
- 收盘价动量 vs 开盘价动量的差值 = 知情交易者vs非知情交易者的信念差异

公式: log(close_t / close_{t-N}) - log(open_t / open_{t-N})
     = sum(log(close_t/close_{t-1}) - log(open_t/open_{t-1}), N日)
     = sum(日内收益 - 隔夜收益的差异变化, N日)

正值 = 日内持续走强(机构买入) > 隔夜表现 → 信息确认
负值 = 日内持续走弱(机构卖出) > 隔夜表现 → 信息恶化

假设: 做多正值(机构信念一致强化), 即正向因子
"""

import numpy as np
import pandas as pd
import sys

def calc_close_open_mom_spread(kline_path, output_path, window=20):
    """计算收盘-开盘动量差因子"""
    
    print(f"[1/5] 加载数据...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    print(f"  数据量: {len(df):,} 行, {df['stock_code'].nunique()} 只股票")
    
    print(f"[2/5] 计算收盘-开盘动量差 (window={window})...")
    
    results = []
    for code, grp in df.groupby('stock_code'):
        grp = grp.sort_values('date').copy()
        
        # 日收盘价对数收益率
        log_close_ret = np.log(grp['close'] / grp['close'].shift(1))
        # 日开盘价对数收益率  
        log_open_ret = np.log(grp['open'] / grp['open'].shift(1))
        
        # 差值 = 日内变化的累积信号
        # close_ret - open_ret = (close_t/close_{t-1}) / (open_t/open_{t-1})
        # 累积N日 = close_momentum - open_momentum
        daily_diff = log_close_ret - log_open_ret
        
        # N日累积
        factor_raw = daily_diff.rolling(window, min_periods=window//2).sum()
        
        # 用于中性化的成交额
        log_amount_20d = np.log(grp['amount'].rolling(20, min_periods=10).mean() + 1)
        
        grp_result = grp[['date', 'stock_code']].copy()
        grp_result['raw_factor'] = factor_raw.values
        grp_result['log_amount_20d'] = log_amount_20d.values
        results.append(grp_result)
    
    df_factor = pd.concat(results, ignore_index=True)
    df_factor = df_factor.dropna(subset=['raw_factor', 'log_amount_20d'])
    
    print(f"  有效因子值: {df_factor['raw_factor'].notna().sum():,}")
    print(f"  因子统计: mean={df_factor['raw_factor'].mean():.6f}, std={df_factor['raw_factor'].std():.6f}")
    
    print(f"[3/5] MAD Winsorize + 成交额OLS中性化 + Z-score...")
    
    factor_values = []
    for date, day_df in df_factor.groupby('date'):
        if len(day_df) < 50:
            continue
        
        vals = day_df['raw_factor'].copy()
        
        # MAD winsorize
        median = vals.median()
        mad = (vals - median).abs().median()
        if mad > 0:
            upper = median + 5 * 1.4826 * mad
            lower = median - 5 * 1.4826 * mad
            vals = vals.clip(lower, upper)
        
        # OLS中性化
        X = day_df['log_amount_20d'].values
        mask = np.isfinite(X) & np.isfinite(vals.values)
        if mask.sum() < 50:
            continue
        
        X_clean = X[mask]
        y_clean = vals.values[mask]
        
        X_mat = np.column_stack([np.ones(len(X_clean)), X_clean])
        try:
            beta = np.linalg.lstsq(X_mat, y_clean, rcond=None)[0]
            residuals = y_clean - X_mat @ beta
        except:
            continue
        
        # Z-score
        std = residuals.std()
        if std > 0:
            z = (residuals - residuals.mean()) / std
        else:
            continue
        
        day_result = day_df.iloc[np.where(mask)[0]][['date', 'stock_code']].copy()
        day_result['factor_value'] = z
        factor_values.append(day_result)
    
    df_out = pd.concat(factor_values, ignore_index=True)
    
    print(f"[4/5] 输出因子...")
    print(f"  总行数: {len(df_out):,}")
    print(f"  日期范围: {df_out['date'].min()} ~ {df_out['date'].max()}")
    print(f"  因子统计: mean={df_out['factor_value'].mean():.4f}, std={df_out['factor_value'].std():.4f}")
    
    df_out.to_csv(output_path, index=False)
    print(f"  保存到: {output_path}")
    
    # 同时计算反向版本
    df_rev = df_out.copy()
    df_rev['factor_value'] = -df_rev['factor_value']
    rev_path = output_path.replace('.csv', '_rev.csv')
    df_rev.to_csv(rev_path, index=False)
    print(f"  反向版本: {rev_path}")
    
    print(f"[5/5] 完成!")
    return df_out

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    parser.add_argument('--output', default='data/factor_co_mom_spread_v1.csv')
    parser.add_argument('--window', type=int, default=20)
    args = parser.parse_args()
    
    calc_close_open_mom_spread(args.kline, args.output, args.window)
