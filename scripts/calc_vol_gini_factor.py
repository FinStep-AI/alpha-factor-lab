#!/usr/bin/env python3
"""因子计算: 成交量Gini集中度因子 (Volume Concentration Factor)

- 20日滚动Gini系数衡量成交量在个股时间序列上的集中程度
- gini = 1 - 2*积分(Lorenz曲线)，值越高=越集中
- 成交额OLS中性化 + MAD缩尾 + z-score
- 方向: 待验证
"""

import numpy as np
import pandas as pd
from pathlib import Path


def gini_coefficient(arr: np.ndarray) -> float:
    """计算Gini系数"""
    arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        return np.nan
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    if sorted_arr.sum() == 0:
        return 0.0
    # Lorenz曲线积分
    cum = np.cumsum(sorted_arr)
    lorenz = np.concatenate([[0], cum / cum[-1]])
    indices = np.arange(0, n + 1) / n
    # 梯形积分
    integral = np.trapz(lorenz, indices)
    gini = 1 - 2 * integral
    return gini


def compute_gini_volume_factor(kline_path: str, output_path: str,
                                window: int = 20,
                                mad_threshold: float = 5.0):
    """计算成交量Gini集中度因子"""
    
    # ─── 读取数据 ───
    df = pd.read_csv(kline_path, dtype={'stock_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # ─── 20日滚动Gini系数 ───
    print("计算20日Gini系数中...")
    gini_list = []
    for sc, grp in df.groupby('stock_code', sort=False):
        vol = grp['volume'].values.astype(float)
        gini_vals = np.full(len(vol), np.nan)
        for i in range(window - 1, len(vol)):
            window_vol = vol[i - window + 1:i + 1]
            gini_vals[i] = gini_coefficient(window_vol)
        gini_list.append(pd.DataFrame({'stock_code': sc, 'date': grp['date'].values,
                                        'gini_raw': gini_vals}))
    
    gini_df = pd.concat(gini_list, ignore_index=True)
    df = df.merge(gini_df, on=['stock_code', 'date'], how='left')
    
    # ─── 成交额20日均值 ───
    df['amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    
    # ─── 截面中性化 + 标准化 ───
    from numpy.linalg import lstsq
    
    def process_group(group):
        mask = group['gini_raw'].notna() & group['amount_20d'].notna() & (group['amount_20d'] > 0)
        if mask.sum() < 30:
            group['factor'] = np.nan
            return group
        
        y = group.loc[mask, 'gini_raw'].values.astype(float)
        x = np.log(group.loc[mask, 'amount_20d'].values.astype(float) + 1)
        x_mat = np.column_stack([np.ones(len(x)), x])
        
        try:
            b, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
            resid = y - x_mat @ b
            group.loc[mask, 'factor'] = resid
        except Exception:
            group['factor'] = np.nan
        
        group.loc[~mask, 'factor'] = np.nan
        return group
    
    print("截面OLS中性化中...")
    df = df.groupby('date', group_keys=False).apply(process_group)
    
    # MAD缩尾 + z-score
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
    
    print("MAD缩尾 + z-score中...")
    df = df.groupby('date', group_keys=False).apply(mad_zscore)
    
    # ─── 输出 ───
    out = df[['date', 'stock_code', 'factor', 'gini_raw']].dropna(subset=['factor'])
    out = out.rename(columns={'factor': 'factor_value'})
    out['stock_code'] = out['stock_code'].astype(str).str.zfill(6)
    out.to_csv(output_path, index=False)
    
    n_dates = out['date'].nunique()
    n_stocks_avg = out.groupby('date')['stock_code'].count().mean()
    print(f"\n成交量Gini因子计算完成")
    print(f"  输出: {output_path}")
    print(f"  日期数: {n_dates}")
    print(f"  平均每期股票数: {n_stocks_avg:.0f}")
    print(f"  因子均值: {out['factor_value'].mean():.4f}")
    print(f"  因子标准差: {out['factor_value'].std():.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    parser.add_argument('--output', default='data/factor_vol_gini_v1.csv')
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--mad-threshold', type=float, default=5.0)
    args = parser.parse_args()
    
    compute_gini_volume_factor(
        kline_path=args.kline,
        output_path=args.output,
        window=args.window,
        mad_threshold=args.mad_threshold
    )
