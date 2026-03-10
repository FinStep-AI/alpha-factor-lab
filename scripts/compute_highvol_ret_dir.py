#!/usr/bin/env python3
"""
因子：高量日收益方向 (High-Volume Day Return Direction)  
ID: highvol_ret_dir_v1

逻辑：
- 识别过去20天中量能突增的日子（当日成交额 > 20日滚动中位数 × 1.5）
- 计算高量日的平均收益
- 再减去全部日的平均收益（去趋势）
- 正值 = 主力在推升; 负值 = 主力在出货

向量化实现
"""

import pandas as pd
import numpy as np
import os

def compute_factor(kline_path, output_path):
    print("Loading kline data...")
    df = pd.read_csv(kline_path, dtype={'stock_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    df['daily_ret'] = df.groupby('stock_code')['close'].pct_change()
    
    window = 20
    print("Computing factor (vectorized)...")
    
    g = df.groupby('stock_code')
    
    # 滚动中位数 of amount (20d)
    df['amt_median_20d'] = g['amount'].transform(
        lambda x: x.rolling(window, min_periods=16).median()
    )
    
    # 高量标志: 当日amount > 1.5 × 滚动中位数
    df['is_high_vol'] = (df['amount'] > df['amt_median_20d'] * 1.5).astype(float)
    
    # 高量日的收益 (非高量日设为NaN)
    df['high_vol_ret'] = np.where(df['is_high_vol'] == 1, df['daily_ret'], np.nan)
    # 非高量日收益
    df['low_vol_ret'] = np.where(df['is_high_vol'] == 0, df['daily_ret'], np.nan)
    
    # 20日滚动: 高量日平均收益, 低量日平均收益
    df['avg_hv_ret'] = g['high_vol_ret'].transform(
        lambda x: x.rolling(window, min_periods=3).mean()
    )
    df['avg_lv_ret'] = g['low_vol_ret'].transform(
        lambda x: x.rolling(window, min_periods=3).mean()
    )
    df['avg_all_ret'] = g['daily_ret'].transform(
        lambda x: x.rolling(window, min_periods=16).mean()
    )
    
    # 因子 = 高量日平均收益 - 全部日平均收益
    df['factor_raw'] = df['avg_hv_ret'] - df['avg_all_ret']
    
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
    print(f"Stats:\n{output['factor'].describe()}")

if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    compute_factor(
        os.path.join(base, 'data', 'csi1000_kline_raw.csv'),
        os.path.join(base, 'data', 'factor_highvol_ret_dir_v1.csv')
    )
