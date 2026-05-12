#!/usr/bin/env python3
"""
累积下行/上行比率因子 (Cumulative Down/Up Ratio)
===============================================
逻辑:
  过去20日内, 所有上涨日的 |ret| 之和 vs 所有下跌日的 |ret| 之和
  ratio = Σ(down_abs) / Σ(up_abs)
  
高比率 = 下跌动量远大于上涨动量 = 对称失衡
方向 : 待验证 (如果高ratio=过度下跌 → 反转做多; 
        如果高ratio=持续下跌动量 → 继续做空)

先做对称失衡版本: ratio = Σ(下跌日 |ret|) - Σ(上涨日 |ret|)
"""

import numpy as np
import pandas as pd
from numpy.linalg import lstsq


def compute_down_up_ratio(kline_path: str, output_path: str,
                          window: int = 20,
                          mad_threshold: float = 5.0,
                          direction: str = 'asym'):
    """
    direction: 'asym'（下减上，反向-低值做多）| 'ratio'（比率，反向-高值做多）
    """
    print(f"[Down/Up Ratio] window={window} direction={direction}")
    
    df = pd.read_csv(kline_path, dtype={'stock_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    df['ret'] = df['pct_change'] / 100.0
    df.loc[df['ret'].abs() > 0.2, 'ret'] = np.nan
    
    df['up_abs'] = np.where(df['ret'] > 0, df['ret'], 0)
    df['down_abs'] = np.where(df['ret'] < 0, -df['ret'], 0)
    
    df['sum_up_20d'] = df.groupby('stock_code')['up_abs'].transform(
        lambda x: x.rolling(window, min_periods=int(window*0.5)).sum()
    )
    df['sum_down_20d'] = df.groupby('stock_code')['down_abs'].transform(
        lambda x: x.rolling(window, min_periods=int(window*0.5)).sum()
    )
    
    if direction == 'asym':
        # 下减上的不对称性
        df['raw'] = df['sum_down_20d'] - df['sum_up_20d']
        desc = 'asym_ret_20d'
    else:
        # 比率 (加eps防零除)
        df['raw'] = df['sum_down_20d'] / (df['sum_up_20d'] + 1e-8)
        desc = 'down_up_ratio_20d'
    
    df['amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    
    def neutralize(group):
        mask = (group['raw'].notna() &
                group['amount_20d'].notna() &
                (group['amount_20d'] > 0))
        if mask.sum() < 30:
            group['factor'] = np.nan
            return group
        y = group.loc[mask, 'raw'].values.astype(float)
        x = np.log(group.loc[mask, 'amount_20d'].values.astype(float) + 1)
        x_mat = np.column_stack([np.ones(len(x)), x])
        try:
            b, _, _, _ = lstsq(x_mat, y, rcond=None)
            group.loc[mask, 'factor'] = y - x_mat @ b
        except Exception:
            group['factor'] = np.nan
        group.loc[~mask, 'factor'] = np.nan
        return group
    
    print("截面OLS中性化...")
    df = df.groupby('date', group_keys=False).apply(neutralize)
    
    def mad_std(group):
        vals = group['factor'].values.astype(float)
        valid = ~np.isnan(vals)
        if valid.sum() < 30:
            group['factor'] = np.nan
            return group
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals[valid] - med))
        if mad < 1e-10:
            group.loc[valid, 'factor'] = 0.0
            return group
        lim = mad_threshold * 1.4826 * mad
        clipped = np.clip(vals, med - lim, med + lim)
        m, s = np.nanmean(clipped[valid]), np.nanstd(clipped[valid])
        if s < 1e-10:
            group.loc[valid, 'factor'] = 0.0
        else:
            group.loc[valid, 'factor'] = (clipped[valid] - m) / s
        group.loc[~valid, 'factor'] = np.nan
        return group
    
    print("MAD + z-score...")
    df = df.groupby('date', group_keys=False).apply(mad_std)
    
    out = df[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    out = out.rename(columns={'factor': 'factor_value'})
    out['stock_code'] = out['stock_code'].astype(str).str.zfill(6)
    out.to_csv(output_path, index=False)
    
    print(f"✓ 完成 → {output_path} ({out['date'].nunique()}天)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    parser.add_argument('--output', default='data/factor_down_up_asym_v1.csv')
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--direction', default='asym', choices=['asym', 'ratio'])
    parser.add_argument('--mad-threshold', type=float, default=5.0)
    args = parser.parse_args()
    
    compute_down_up_ratio(
        kline_path=args.kline,
        output_path=args.output,
        window=args.window,
        direction=args.direction,
        mad_threshold=args.mad_threshold
    )
