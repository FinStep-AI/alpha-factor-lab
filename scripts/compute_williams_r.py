#!/usr/bin/env python3
"""
因子：威廉姆斯%R反转 (Williams %R Reversal)
ID: williams_r_rev_v1

逻辑：
  %R = (highest_high_20 - close) / (highest_high_20 - lowest_low_20)
  范围 [0, 1]
  
  %R ≈ 1: 股价接近20日最低 → 超卖
  %R ≈ 0: 股价接近20日最高 → 超买
  
  反转假设: 做多超卖(%R高)，做空超买(%R低)
  正向使用：高%R → 高预期收益
  
  与close_location不同: 
  - CLV看单日日内位置(close在high-low的哪里)
  - %R看跨日位置(close在20日range的哪里)
  两者完全不同的信息

中性化: 成交额 OLS残差
"""

import pandas as pd
import numpy as np
import os

def compute_factor(kline_path, output_path):
    print("Loading kline data...")
    df = pd.read_csv(kline_path, dtype={'stock_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    window = 20
    g = df.groupby('stock_code')
    
    # 20日最高价和最低价
    df['highest_20'] = g['high'].transform(
        lambda x: x.rolling(window, min_periods=16).max()
    )
    df['lowest_20'] = g['low'].transform(
        lambda x: x.rolling(window, min_periods=16).min()
    )
    
    # %R = (highest - close) / (highest - lowest)
    range_20 = df['highest_20'] - df['lowest_20']
    df['factor_raw'] = np.where(
        range_20 > 0,
        (df['highest_20'] - df['close']) / range_20,
        0.5
    )
    
    # log(20日平均成交额)
    df['mean_amt'] = g['amount'].transform(
        lambda x: x.rolling(window, min_periods=16).mean()
    )
    df['log_amount_20d'] = np.log(df['mean_amt'].clip(lower=1))
    
    factor_df = df[['date', 'stock_code', 'factor_raw', 'log_amount_20d']].dropna().copy()
    print(f"Raw factor: {len(factor_df)} rows, {factor_df['date'].nunique()} dates")
    
    # 截面中性化
    print("Neutralizing...")
    def neutralize_cs(group):
        y = group['factor_raw'].values.copy()
        x = group['log_amount_20d'].values.copy()
        valid = np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 30:
            return pd.Series(np.nan, index=group.index, name='factor')
        y_v, x_v = y[valid], x[valid]
        med = np.median(y_v)
        mad = np.median(np.abs(y_v - med)) * 1.4826
        if mad > 0:
            y_v = np.clip(y_v, med - 3*mad, med + 3*mad)
        x_mat = np.column_stack([np.ones(len(x_v)), x_v])
        beta = np.linalg.lstsq(x_mat, y_v, rcond=None)[0]
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
    
    output = factor_df[['date', 'stock_code', 'factor']].dropna().copy()
    output['stock_code'] = output['stock_code'].astype(str).str.zfill(6)
    output = output.sort_values(['date', 'stock_code'])
    output.to_csv(output_path, index=False)
    print(f"Factor saved: {output_path} ({len(output)} rows)")
    print(f"Date range: {output['date'].min()} ~ {output['date'].max()}")
    print(f"Stocks/date: {output.groupby('date')['stock_code'].count().mean():.0f}")

if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    compute_factor(
        os.path.join(base, 'data', 'csi1000_kline_raw.csv'),
        os.path.join(base, 'data', 'factor_williams_r_v1.csv')
    )
