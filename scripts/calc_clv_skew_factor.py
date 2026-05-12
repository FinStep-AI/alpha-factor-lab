#!/usr/bin/env python3
"""
日内价格位置幂律因子 (Intraday Close Position Skew)
======================================================
衡量过去20日内每日收盘位置的偏度(而非均值)。

  clv_day = (high + low + close) / (2*(high+low)/2) = close / (high+low) * 2
  简化: clv = (high + low - 2*close?) — 不对

标准CLV (Close Location Value):
  clv = (close-low) / (high-low+eps) ∈ [0,1]
  0=最低, 1=最高

日内偏度：衡量"收盘位置分布"的偏斜方向：
- skew > 0: 均值 > 0.5 → 股票偏好在区间上半部收盘 → 日内买盘较强
- skew < 0: 均值 < 0.5 → 股票偏好在区间下半部收盘 → 日内卖压较强

采用 CLV (close-low)/(high-low+eps)，20日截面偏度中性化后排序。

待验证方向: skew < 0 是否意味着次日反转回到中间值？
"""

import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import lstsq


def safe_skew(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 5:
        return np.nan
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-10:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def compute_clv_skew_factor(kline_path: str, output_path: str,
                              window: int = 20,
                              mad_threshold: float = 5.0,
                              forward_days: int = 20):
    print(f"[CLV Skew] window={window}")
    
    df = pd.read_csv(kline_path, dtype={'stock_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    hl = df['high'] - df['low']
    hl = hl.replace(0, np.nan)
    df['clv'] = (df['close'] - df['low']) / (hl + 1e-8)
    df['clv'] = df['clv'].clip(0, 1)
    
    # 过去20日CLV滚动偏度
    print("计算20日CLV偏度...")
    
    # 先做20日滚动偏度（慢但准确）
    def rolling_skew(group):
        vals = group['clv'].values.astype(float)
        out = np.full(len(vals), np.nan)
        for i in range(window - 1, len(vals)):
            out[i] = safe_skew(vals[i - window + 1:i + 1])
        return pd.Series(out, index=group.index)
    
    df['clv_skew'] = df.groupby('stock_code', group_keys=False).apply(rolling_skew)
    
    # 成交额中性化 + MAD + z-score
    df['amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    
    def neutralize(group):
        mask = (group['clv_skew'].notna() &
                group['amount_20d'].notna() &
                (group['amount_20d'] > 0) &
                np.isfinite(group['clv_skew']))
        if mask.sum() < 30:
            group['factor'] = np.nan
            return group
        y = group.loc[mask, 'clv_skew'].values.astype(float)
        x = np.log(group.loc[mask, 'amount_20d'].values.astype(float) + 1)
        x_mat = np.column_stack([np.ones(len(x)), x])
        try:
            b, _, _, _ = lstsq(x_mat, y, rcond=None)
            group.loc[mask, 'factor'] = y - x_mat @ b
        except Exception:
            group['factor'] = np.nan
        group.loc[~mask, 'factor'] = np.nan
        return group
    
    print("截面中性化...")
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
    
    out = df[['date', 'stock_code', 'factor', 'clv_skew']].dropna(subset=['factor'])
    out = out.rename(columns={'factor': 'factor_value'})
    out['stock_code'] = out['stock_code'].astype(str).str.zfill(6)
    out.to_csv(output_path, index=False)
    
    print(f"✓ CLV偏度因子 → {output_path} ({out['date'].nunique()}天)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    parser.add_argument('--output', default='data/factor_clv_skew_v1.csv')
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--mad-threshold', type=float, default=5.0)
    args = parser.parse_args()
    
    compute_clv_skew_factor(
        kline_path=args.kline,
        output_path=args.output,
        window=args.window,
        mad_threshold=args.mad_threshold
    )
