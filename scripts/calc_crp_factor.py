#!/usr/bin/env python3
"""因子计算: 日内收盘位置因子 (Close Range Position Factor)

- 原始值: (close - low) / (high - low + 1e-8)
- 20日滚动均值平滑
- 成交额OLS中性化 + MAD缩尾 + z-score
- 方向: 待验证 (高值=收在高位=正向?)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

def compute_crp_factor(kline_path: str, output_path: str,
                        window: int = 20,
                        amount_neutralize: bool = True,
                        mad_threshold: float = 5.0):
    """
    计算日内收盘位置因子 (CRP Factor)
    
    Parameters
    ----------
    kline_path : str
        输入K线数据路径 (CSV)
    output_path : str
        输出因子值CSV路径
    window : int
        滚动窗口日数，默认20
    amount_neutralize : bool
        是否做成交额中性化，默认True
    mad_threshold : float
        MAD去极值倍数，默认5.0
    """
    
    # ─── 读取数据 ───
    df = pd.read_csv(kline_path, dtype={'stock_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # ─── 计算原始CRP值: (close - low) / (high - low) ───
    hl = df['high'] - df['low']
    hl = hl.replace(0, np.nan)  # 处理HL=0的情况
    
    df['crp_raw'] = (df['close'] - df['low']) / (hl + 1e-8)
    df['crp_raw'] = df['crp_raw'].clip(0, 1)  # 理论上应在0-1之间
    
    # ─── 20日滚动均值 ───
    df['crp_ma'] = df.groupby('stock_code')['crp_raw'].transform(
        lambda x: x.rolling(window, min_periods=int(window*0.6)).mean()
    )
    
    # ─── 成交额设20日均值 ───
    df['amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    
    # ─── 截面OLS中性化（成交额） ───
    from numpy.linalg import lstsq
    
    def cross_section_neutralize(group):
        """截面OLS中性化: factor ~ log_amount_20d"""
        mask = group['crp_ma'].notna() & group['amount_20d'].notna() & (group['amount_20d'] > 0)
        if mask.sum() < 30:
            group['factor'] = np.nan
            return group
        
        y = group.loc[mask, 'crp_ma'].values.astype(float)
        x = np.log(group.loc[mask, 'amount_20d'].values.astype(float) + 1)
        x = np.column_stack([np.ones(len(x)), x])
        
        try:
            # 使用tw:lstsq进行稳健的线性回归拟合
            b, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
            residual = y - x @ b
            group.loc[mask, 'factor'] = residual
        except Exception:
            group['factor'] = np.nan
        
        group.loc[~mask, 'factor'] = np.nan
        return group
    
    df = df.groupby('date', group_keys=False).apply(cross_section_neutralize)
    
    # ─── MAD缩尾 + z-score 截面标准化 ───
    def mad_zscore(group):
        vals = group['factor'].values.astype(float)
        valid = ~np.isnan(vals)
        if valid.sum() < 30:
            group['factor'] = np.nan
            return group
        
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - med))
        if mad < 1e-10:
            group.loc[valid, 'factor'] = 0.0
            group.loc[~valid, 'factor'] = np.nan
            return group
        
        upper = med + mad_threshold * 1.4826 * mad
        lower = med - mad_threshold * 1.4826 * mad
        vals_clipped = np.clip(vals, lower, upper)
        
        mean = np.nanmean(vals_clipped)
        std = np.nanstd(vals_clipped)
        if std < 1e-10:
            group.loc[valid, 'factor'] = 0.0
        else:
            group.loc[valid, 'factor'] = (vals_clipped[valid] - mean) / std
        
        group.loc[~valid, 'factor'] = np.nan
        return group
    
    df = df.groupby('date', group_keys=False).apply(mad_zscore)
    
    # ─── 输出 ───
    out = df[['date', 'stock_code', 'factor', 'crp_ma']].dropna(subset=['factor'])
    out = out.rename(columns={'factor': 'factor_value'})
    out['stock_code'] = out['stock_code'].astype(str).str.zfill(6)
    out.to_csv(output_path, index=False)
    
    n_dates = out['date'].nunique()
    n_stocks_avg = out.groupby('date')['stock_code'].count().mean()
    print(f" CRP因子计算完成")
    print(f" 输出: {output_path}")
    print(f" 日期数: {n_dates}")
    print(f" 平均每期股票数: {n_stocks_avg:.0f}")
    print(f" 因子均值: {out['factor_value'].mean():.4f}")
    print(f" 因子标准差: {out['factor_value'].std():.4f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    parser.add_argument('--output', default='data/factor_crp_v1.csv')
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--no-neutralize', action='store_true')
    parser.add_argument('--mad-threshold', type=float, default=5.0)
    args = parser.parse_args()
    
    compute_crp_factor(
        kline_path=args.kline,
        output_path=args.output,
        window=args.window,
        amount_neutralize=not args.no_neutralize,
        mad_threshold=args.mad_threshold
    )
