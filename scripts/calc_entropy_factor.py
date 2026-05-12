#!/usr/bin/env python3
"""
收益率分布熵因子 (Return Distribution Entropy)
===============================================
计算过去20日收益率分布的 Shannon 熵近似：
  H = - Σ p_i * log(p_i)  其中 p_i = |ret_i| / Σ|ret_j|

高熵 → 收益分布均匀分散 → 更多"噪音型"随机波动
低熵 → 收益集中在少数大日 → 信息驱动型（信号明确）

方向: 低熵 → 信息明确 → 后续有更明确的方向性 → 做多低熵
"""

import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import lstsq


def compute_entropy_factor(kline_path: str, output_path: str,
                            window: int = 20,
                            mad_threshold: float = 5.0):
    print(f"[Return Entropy] window={window}")
    
    df = pd.read_csv(kline_path, dtype={'stock_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    df['ret'] = df['pct_change'] / 100.0
    df.loc[df['ret'].abs() > 0.2, 'ret'] = np.nan
    
    # 计算熵（用shannon熵）
    def entropy_20d(series):
        vals = series.values.astype(float)
        vals = np.abs(vals[~np.isnan(vals)])
        total = vals.sum()
        if total < 1e-10 or len(vals) < 5:
            return np.nan
        probs = vals / total
        probs = probs[probs > 0]  # 排除0
        return -np.sum(probs * np.log(probs))
    
    print("计算20日Shannon熵...")
    df['entropy'] = df.groupby('stock_code')['ret'].transform(
        lambda x: x.rolling(window, min_periods=int(window*0.6)).apply(entropy_20d, raw=False)
    )
    
    # 成交额中性化
    df['amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    
    def neutralize(group):
        mask = (group['entropy'].notna() &
                group['amount_20d'].notna() &
                (group['amount_20d'] > 0))
        if mask.sum() < 30:
            group['factor'] = np.nan
            return group
        y = group.loc[mask, 'entropy'].values.astype(float)
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
    df = df.groupby('date', group_keys=False).apply(mad_zscore)
    
    out = df[['date', 'stock_code', 'factor', 'entropy']].dropna(subset=['factor'])
    out = out.rename(columns={'factor': 'factor_value'})
    out['stock_code'] = out['stock_code'].astype(str).str.zfill(6)
    out.to_csv(output_path, index=False)
    
    print(f"✓ 熵因子完成 → {output_path} ({out['date'].nunique()}天)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    parser.add_argument('--output', default='data/factor_entropy_v1.csv')
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--mad-threshold', type=float, default=5.0)
    args = parser.parse_args()
    
    compute_entropy_factor(
        kline_path=args.kline,
        output_path=args.output,
        window=args.window,
        mad_threshold=args.mad_threshold
    )
