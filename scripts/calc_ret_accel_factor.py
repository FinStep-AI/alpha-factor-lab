#!/usr/bin/env python3
"""
收益率时序加速因子 (Return Acceleration Factor)
=================================================
逻辑: 短期收益趋势变化相对于长期趋势
  accel = MA20(ret) - MA60(ret)
  = 近20日收益率均值 - 近60日收益率均值

正 = 加速上升（短期比长期好→动能量在积累，考虑反转/延续要看截面）
负 = 加速下降（短期比长期差→动能衰竭）

用交叉验证在不同市场环境下的效果不确定，
先做全量版本试一下。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import lstsq


def compute_ret_accel_factor(kline_path: str, output_path: str,
                              short_window: int = 20,
                              long_window: int = 60,
                              mad_threshold: float = 5.0):
    print(f"[Ret Acceleration] short={short_window}, long={long_window}")
    
    df = pd.read_csv(kline_path, dtype={'stock_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # 收益率
    df['ret'] = df['pct_change'] / 100.0
    df.loc[df['ret'].abs() > 0.2, 'ret'] = np.nan  # 极端收益截断
    
    # 短/长周期均值
    df['ma_short'] = df.groupby('stock_code')['ret'].transform(
        lambda x: x.rolling(short_window, min_periods=int(short_window*0.6)).mean()
    )
    df['ma_long'] = df.groupby('stock_code')['ret'].transform(
        lambda x: x.rolling(long_window, min_periods=int(long_window*0.6)).mean()
    )
    
    # 加速因子
    df['accel_raw'] = df['ma_short'] - df['ma_long']
    
    # 成交额OLS中性化 + MAD + z-score
    df['amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    
    def neutralize_group(group):
        mask = (group['accel_raw'].notna() &
                group['amount_20d'].notna() &
                (group['amount_20d'] > 0))
        if mask.sum() < 30:
            group['factor'] = np.nan
            return group
        
        y = group.loc[mask, 'accel_raw'].values.astype(float)
        x = np.log(group.loc[mask, 'amount_20d'].values.astype(float) + 1)
        x_mat = np.column_stack([np.ones(len(x)), x])
        
        try:
            b, _, _, _ = lstsq(x_mat, y, rcond=None)
            resid = y - x_mat @ b
            group.loc[mask, 'factor'] = resid
        except Exception:
            group['factor'] = np.nan
        
        group.loc[~mask, 'factor'] = np.nan
        return group
    
    print("截面OLS中性化中...")
    df = df.groupby('date', group_keys=False).apply(neutralize_group)
    
    def mad_zscore(group):
        vals = group['factor'].values.astype(float)
        valid = ~np.isnan(vals)
        if valid.sum() < 30:
            group['factor'] = np.nan
            return group
        
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals[valid] - med))
        if mad < 1e-10:
            group.loc[valid, 'factor'] = 0.0
            group.loc[~valid, 'factor'] = np.nan
            return group
        
        upper = med + mad_threshold * 1.4826 * mad
        lower = med - mad_threshold * 1.4826 * mad
        clipped = np.clip(vals, lower, upper)
        
        m, s = np.nanmean(clipped[valid]), np.nanstd(clipped[valid])
        if s < 1e-10:
            group.loc[valid, 'factor'] = 0.0
        else:
            group.loc[valid, 'factor'] = (clipped[valid] - m) / s
        
        group.loc[~valid, 'factor'] = np.nan
        return group
    
    print("MAD + z-score中...")
    df = df.groupby('date', group_keys=False).apply(mad_zscore)
    
    out = df[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    out = out.rename(columns={'factor': 'factor_value'})
    out['stock_code'] = out['stock_code'].astype(str).str.zfill(6)
    out.to_csv(output_path, index=False)
    
    n_dates = out['date'].nunique()
    n_avg = out.groupby('date')['stock_code'].count().mean()
    print(f"\n✓ 收益率加速度因子完成")
    print(f"  输出: {output_path}")
    print(f"  日数: {n_dates}  平均股票: {n_avg:.0f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    parser.add_argument('--output', default='data/factor_ret_accel_v1.csv')
    parser.add_argument('--short-window', type=int, default=20)
    parser.add_argument('--long-window', type=int, default=60)
    parser.add_argument('--mad-threshold', type=float, default=5.0)
    args = parser.parse_args()
    
    compute_ret_accel_factor(
        kline_path=args.kline,
        output_path=args.output,
        short_window=args.short_window,
        long_window=args.long_window,
        mad_threshold=args.mad_threshold
    )
