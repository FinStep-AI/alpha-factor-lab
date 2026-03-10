#!/usr/bin/env python3
"""
因子：成交额加权动量 (Amount-Weighted Momentum)
ID: amt_weighted_mom_v1
Barra风格: Momentum

逻辑：
  成交额加权平均收益 vs 等权平均收益 的差值
  差值>0 = 高量日倾向上涨 = 资金推升
  差值<0 = 高量日倾向下跌 = 资金出逃
  
向量化实现，避免Python循环
"""

import pandas as pd
import numpy as np
import os

def compute_factor(kline_path, output_path):
    print("Loading kline data...")
    df = pd.read_csv(kline_path, dtype={'stock_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # 日收益率
    df['daily_ret'] = df.groupby('stock_code')['close'].pct_change()
    
    window = 20
    
    print("Computing rolling weighted momentum (vectorized)...")
    
    # 对每只股票计算滚动窗口
    # ret * amount 的20日滚动和 / amount的20日滚动和 - ret的20日滚动均值
    df['ret_x_amt'] = df['daily_ret'] * df['amount']
    
    g = df.groupby('stock_code')
    
    # 滚动和
    df['sum_ret_x_amt'] = g['ret_x_amt'].transform(lambda x: x.rolling(window, min_periods=int(window*0.8)).sum())
    df['sum_amt'] = g['amount'].transform(lambda x: x.rolling(window, min_periods=int(window*0.8)).sum())
    df['mean_ret'] = g['daily_ret'].transform(lambda x: x.rolling(window, min_periods=int(window*0.8)).mean())
    df['mean_amt'] = g['amount'].transform(lambda x: x.rolling(window, min_periods=int(window*0.8)).mean())
    
    # 成交额加权平均收益 - 等权平均收益
    df['factor_raw'] = df['sum_ret_x_amt'] / df['sum_amt'] - df['mean_ret']
    
    # log(20日平均成交额) 用于中性化
    df['log_amount_20d'] = np.log(df['mean_amt'].clip(lower=1))
    
    # 去掉无效行
    factor_df = df[['date', 'stock_code', 'factor_raw', 'log_amount_20d']].dropna().copy()
    
    print(f"Raw factor: {len(factor_df)} rows, {factor_df['date'].nunique()} dates")
    
    # 截面中性化
    print("Cross-section neutralization...")
    def neutralize_cs(group):
        y = group['factor_raw'].values.copy()
        x = group['log_amount_20d'].values.copy()
        
        valid = np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 30:
            return pd.Series(np.nan, index=group.index, name='factor')
        
        y_v, x_v = y[valid], x[valid]
        
        # MAD winsorize
        med = np.median(y_v)
        mad = np.median(np.abs(y_v - med)) * 1.4826
        if mad > 0:
            y_v = np.clip(y_v, med - 3*mad, med + 3*mad)
        
        # OLS
        x_mat = np.column_stack([np.ones(len(x_v)), x_v])
        beta, _, _, _ = np.linalg.lstsq(x_mat, y_v, rcond=None)
        resid = y_v - x_mat @ beta
        
        std = np.std(resid)
        if std > 0:
            resid = (resid - np.mean(resid)) / std
        
        out = np.full(len(y), np.nan)
        out[valid] = resid
        return pd.Series(out, index=group.index, name='factor')
    
    factor_df['factor'] = factor_df.groupby('date', group_keys=False).apply(
        lambda g: neutralize_cs(g)
    ).values
    
    # 输出
    output = factor_df[['date', 'stock_code', 'factor']].dropna()
    output = output.copy()
    output['stock_code'] = output['stock_code'].astype(str).str.zfill(6)
    output = output.sort_values(['date', 'stock_code'])
    output.to_csv(output_path, index=False)
    print(f"Factor saved: {output_path} ({len(output)} rows)")
    
    # Diagnostics
    print(f"\nDate range: {output['date'].min()} ~ {output['date'].max()}")
    print(f"Stocks/date (mean): {output.groupby('date')['stock_code'].count().mean():.0f}")
    print(f"Factor stats:\n{output['factor'].describe()}")

if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kline_path = os.path.join(base, 'data', 'csi1000_kline_raw.csv')
    output_path = os.path.join(base, 'data', 'factor_amt_weighted_mom_v1.csv')
    compute_factor(kline_path, output_path)
