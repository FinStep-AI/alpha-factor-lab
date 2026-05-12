#!/usr/bin/python3
"""
波动率幂律斜率因子 (Volatility Power-Law Slope)
=================================================
逻辑：多尺度(std5/std20/std60)横截面的幂律拟合斜率近似Hurst

  log_std = a + H * log(window) + ε
  
  H > 0.5: 波动率强聚集(趋势持续)
  H < 0.5: 波动率反聚集(均值回复)

截面: 每只股票在截面T+1日，横跨个股，做OLS: 各档窗深回归标准差再算斜率(H_approx = slope / slope_ref？)

技术方案：
  直接取 (log_std5 - log_std20) / (log(5)-log(20))
  这是两尺度的Hurst近似，不考虑主观加权。

方向: 待验证
"""

import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import lstsq


def compute_vol_slope_factor(kline_path: str, output_path: str,
                              short_win: int = 5,
                              mid_win: int = 20,
                              long_win: int = 60,
                              mad_threshold: float = 5.0):
    print(f"[Vol Slope] short={short_win}, mid={mid_win}, long={long_win}")
    
    df = pd.read_csv(kline_path, dtype={'stock_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    df['ret'] = df['pct_change'] / 100.0
    df.loc[df['ret'].abs() > 0.2, 'ret'] = np.nan
    
    # log(return) - 用于幂律估计
    df['log_ret'] = np.log1p(df['ret'].fillna(0))
    
    for w in [short_win, mid_win, long_win]:
        df[f'std_{w}d'] = df.groupby('stock_code')['log_ret'].transform(
            lambda x: x.rolling(w, min_periods=max(2, int(w*0.5))).std()
        )
    
    # Log of variances
    df['log_std_short'] = np.log(df[f'std_{short_win}d'].clip(1e-10))
    df['log_std_long']  = np.log(df[f'std_{long_win}d'].clip(1e-10))
    df['log_std_mid']   = np.log(df[f'std_{mid_win}d'].clip(1e-10))
    
    log_w_ratio_sl = np.log(short_win / long_win)  # ≈ log(5/60)
    
    # 两尺度Hurst近似 (忽略mid): H_approx ≈ (log(std5)-log(std60)) / log(5/60)
    # log_correction   #  哦 这等价于 obj_val_w = log_std / log(ts)
    
    # 不用乘方来直接取 ratio:
    # H_2scale = (log_std_short - log_std_long) / log(short_win/long_win)
    df['h_2scale'] = (df['log_std_short'] - df['log_std_long']) / (log_w_ratio_sl + 1e-10)
    
    # 也可以用三尺度OLS, 但两尺度最简单
    df['raw'] = df['h_2scale']
    
    # 成交额中性化
    df['amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    
    def neutralize(group):
        mask = (group['raw'].notna() &
                group['amount_20d'].notna() &
                (group['amount_20d'] > 0) &
                np.isfinite(group['raw']))
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
    
    def mad_zscore(group):
        vals = group['factor'].values.astype(float)
        valid = ~np.isnan(vals) & np.isfinite(vals)
        if valid.sum() < 30:
            group['factor'] = np.nan
            return group
        med = np.nanmedian(vals[valid])
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
    df = df.groupby('date', group_keys=False).apply(mad_zscore)
    
    out = df[['date', 'stock_code', 'factor', 'h_2scale']].dropna(subset=['factor'])
    out = out.rename(columns={'factor': 'factor_value'})
    out['stock_code'] = out['stock_code'].astype(str).str.zfill(6)
    out.to_csv(output_path, index=False)
    
    print(f"✓ 波动率斜率因子 → {output_path} ({out['date'].nunique()}天)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    parser.add_argument('--output', default='data/factor_vol_slope_v1.csv')
    parser.add_argument('--short-win', type=int, default=5)
    parser.add_argument('--mid-win', type=int, default=20)
    parser.add_argument('--long-win', type=int, default=60)
    parser.add_argument('--mad-threshold', type=float, default=5.0)
    args = parser.parse_args()
    
    compute_vol_slope_factor(
        kline_path=args.kline,
        output_path=args.output,
        short_win=args.short_win,
        mid_win=args.mid_win,
        long_win=args.long_win,
        mad_threshold=args.mad_threshold
    )
