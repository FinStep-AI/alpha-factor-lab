#!/usr/bin/env python3
"""
开盘跳空形态因子 (Gap Posture Factor)
======================================
逻辑:
  gap_ret = open / prev_close - 1            # 开盘跳空幅度
  posture = (close - low) / (high - low + eps)  # 收盘在日内区间的位置
  gap_dir = sign(gap_ret)                    # 跳空方向（+/-1）

两种信号，分别做市值/成交额中性化：

1. gap_fade: 正跳空 + 收盘贴近低点 → 跳空衰竭 → 次日反转
   = -sign(gap_ret) * posture   （负相关：正跳空+低收盘=高衰竭=负向体质）

2. gap_continue: 正跳空 + 收盘贴近高点 → 跳空持续 → 动量延续
   = sign(gap_ret) * posture

20日ROLLING平滑后截面中性化。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import lstsq


def compute_gap_posture_factor(kline_path: str, output_path: str,
                               window: int = 20,
                               mode: str = 'fade',   # 'fade' or 'continue'
                               mad_threshold: float = 5.0):
    """
    Parameters
    ----------
    kline_path : str
    output_path : str
    window : 平滑窗口
    mode : 'fade'(反向) 或 'continue'(正向)
    """
    print(f"[Gap Posture] 模式={mode}, window={window}")
    
    # ─── 读取数据 ───
    df = pd.read_csv(kline_path, dtype={'stock_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # ─── 计算跳空和收盘位置 ───
    df['prev_close'] = df.groupby('stock_code')['close'].shift(1)
    df['gap_ret'] = df['open'] / (df['prev_close'] + 1e-8) - 1
    
    hl = df['high'] - df['low']
    hl = hl.replace(0, np.nan)
    df['close_pos'] = (df['close'] - df['low']) / (hl + 1e-8)  # 0=收在低点, 1=收在最高
    df['close_pos'] = df['close_pos'].clip(0, 1)
    
    # ─── 跳空形态信号 ───
    if mode == 'fade':
        # 正跳空+低收=衰竭 → 负信号（高衰竭→做空，次日反弹）
        # 反转使用：高衰竭→次日反弹→做多
        df['signal_raw'] = np.where(df['gap_ret'] > 0,
                                    -df['close_pos'],   # 正跳空+低收=高衰竭(负值)→反转做多
                                     df['close_pos'])   # 负跳空+低收=加剧→正向做空
    else:  # 'continue'
        df['signal_raw'] = np.sign(df['gap_ret']) * df['close_pos']
    
    # ─── 20日平滑 ───
    df['signal_ma'] = df.groupby('stock_code')['signal_raw'].transform(
        lambda x: x.rolling(window, min_periods=int(window*0.6)).mean()
    )
    
    # ─── 成交额中性化 + MAD + z-score ───
    df['amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    
    def neutralize_and_std(group):
        mask = (group['signal_ma'].notna() &
                group['amount_20d'].notna() &
                (group['amount_20d'] > 0))
        if mask.sum() < 30:
            group['factor'] = np.nan
            return group
        
        y = group.loc[mask, 'signal_ma'].values.astype(float)
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
    df = df.groupby('date', group_keys=False).apply(neutralize_and_std)
    
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
        
        m = np.nanmean(clipped[valid])
        s = np.nanstd(clipped[valid])
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
    print(f"\n✓ Gap Posture ({mode}) 因子完成")
    print(f"  输出: {output_path}")
    print(f"  日数: {n_dates}  平均股票: {n_avg:.0f}")
    print(f"  因子均值: {out['factor_value'].mean():.4f}  std: {out['factor_value'].std():.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    parser.add_argument('--output', default='data/factor_gap_posture_v1.csv')
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--mode', default='fade', choices=['fade', 'continue'])
    parser.add_argument('--mad-threshold', type=float, default=5.0)
    args = parser.parse_args()
    
    compute_gap_posture_factor(
        kline_path=args.kline,
        output_path=args.output,
        window=args.window,
        mode=args.mode,
        mad_threshold=args.mad_threshold
    )
