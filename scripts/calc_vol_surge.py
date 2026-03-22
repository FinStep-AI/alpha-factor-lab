#!/usr/bin/env python3
"""
Volume Surge Momentum (成交量激增动量) 因子

原理：
vol_surge = log(MA20_current_volume / MA20_lagged60d_volume)
衡量当前成交量相比60天前的变化幅度。

高vol_surge → 近期成交量显著放大 → 市场关注度提升/新信息注入
低vol_surge → 成交量萎缩 → 关注度下降

Barra风格: Sentiment/Activity

参考文献:
- Gervais, Kaniel & Mingelgrin (2001) "The High-Volume Return Premium" JF
  发现异常高成交量后股票有正超额收益
- Lerman, Livnat & Mendenhall (2012) "The High-Volume Return Premium and Post-Earnings Announcement Drift" 
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compute_volume_surge(kline_path: str, output_path: str, short_window: int = 20, lag: int = 60):
    """计算Volume Surge因子"""
    print(f"Loading kline data from {kline_path}...")
    df = pd.read_csv(kline_path, parse_dates=['date'])
    df = df[df['date'] <= '2026-03-07'].copy()
    
    print(f"Computing volume surge (short={short_window}d, lag={lag}d)...")
    
    results = []
    for code, grp in df.groupby('stock_code'):
        grp = grp.sort_values('date').copy()
        
        # MA20 当前成交量
        ma_current = grp['volume'].rolling(short_window, min_periods=short_window).mean()
        
        # MA20 在lag天前的值
        ma_lagged = ma_current.shift(lag)
        
        # log ratio
        grp['vol_surge_raw'] = np.log(ma_current / ma_lagged)
        
        # 也计算20日均成交额用于中性化
        grp['log_amount_20d'] = np.log(grp['amount'].rolling(20, min_periods=20).mean() + 1)
        
        results.append(grp[['date', 'stock_code', 'vol_surge_raw', 'log_amount_20d']].dropna())
    
    factor_df = pd.concat(results, ignore_index=True)
    # 过滤Inf
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Raw factor shape: {factor_df.shape}")
    
    # 截面OLS中性化
    print("OLS neutralization by log_amount_20d...")
    neutralized = []
    for dt, cross in factor_df.groupby('date'):
        if len(cross) < 50:
            continue
        x = cross['log_amount_20d'].values
        y = cross['vol_surge_raw'].values
        
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 50:
            continue
        
        x_v, y_v = x[valid], y[valid]
        X = np.column_stack([np.ones(len(x_v)), x_v])
        try:
            beta = np.linalg.lstsq(X, y_v, rcond=None)[0]
        except:
            continue
        
        residual = np.full(len(x), np.nan)
        residual[valid] = y_v - X @ beta
        
        tmp = cross[['date', 'stock_code']].copy()
        tmp['factor_raw'] = residual
        neutralized.append(tmp)
    
    factor_df = pd.concat(neutralized, ignore_index=True).dropna()
    
    # MAD winsorize + z-score
    print("MAD winsorize + z-score...")
    final = []
    for dt, cross in factor_df.groupby('date'):
        vals = cross['factor_raw'].values
        median = np.median(vals)
        mad = np.median(np.abs(vals - median))
        if mad < 1e-10:
            continue
        
        upper = median + 5 * 1.4826 * mad
        lower = median - 5 * 1.4826 * mad
        clipped = np.clip(vals, lower, upper)
        
        mean = np.mean(clipped)
        std = np.std(clipped)
        if std < 1e-10:
            continue
        
        z = (clipped - mean) / std
        
        tmp = cross[['date', 'stock_code']].copy()
        tmp['factor_value'] = z
        final.append(tmp)
    
    result = pd.concat(final, ignore_index=True)
    result.to_csv(output_path, index=False)
    
    print(f"Output shape: {result.shape}")
    print(f"Date range: {result['date'].min()} ~ {result['date'].max()}")
    print(f"Saved to {output_path}")
    
    return result


if __name__ == '__main__':
    base = Path(__file__).resolve().parent.parent
    compute_volume_surge(
        kline_path=str(base / 'data' / 'csi1000_kline_raw.csv'),
        output_path=str(base / 'data' / 'factor_vol_surge_v1_long.csv'),
        short_window=20,
        lag=60
    )
