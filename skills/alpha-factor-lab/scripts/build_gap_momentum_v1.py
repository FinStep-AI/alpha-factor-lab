#!/usr/bin/env python3
"""
因子: 跳空缺口动量 (Gap Momentum) v1
factor_id: gap_momentum_v1

逻辑:
  - 跳空缺口 = (今日open - 昨日close) / 昨日close (即隔夜收益)
  - 20日内跳空缺口的不对称性: 上跳次数占比 - 下跳次数占比
  - 加权版本: 以缺口幅度为权重
  - 核心思想: 持续上跳的股票有更强的买方压力(如集合竞价资金持续流入)

与隔夜动量的区别:
  - 隔夜动量: 累计隔夜收益 - 累计日内收益 (幅度差)
  - 跳空缺口: 关注跳空方向的一致性和幅度加权 (方向性+集中度)
  - 互补角度: 一个看"赚了多少"，一个看"方向有多一致"

假设:
  - A股集合竞价反映知情交易者的方向判断
  - 持续单方向跳空暗示信息逐步释放，后续有动量延续
  - 中证1000小盘股跳空信号更清晰(市场关注度低，信息释放慢)
  
参考:
  - Branch & Ma (2012) "Overnight Return, the Invisible Hand"
  - 光大证券《集合竞价与跳空缺口的Alpha信号》
"""

import numpy as np
import pandas as pd
from pathlib import Path

def main():
    data_dir = Path("data")
    
    print("Loading kline data...")
    df = pd.read_csv(data_dir / "csi1000_kline_raw.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date'])
    
    print(f"Stocks: {df['stock_code'].nunique()}, Dates: {df['date'].min()} ~ {df['date'].max()}")
    
    # Calculate gap (overnight return)
    df['prev_close'] = df.groupby('stock_code')['close'].shift(1)
    df['gap'] = (df['open'] - df['prev_close']) / df['prev_close']
    
    # Clip extreme gaps (limit up/down artifacts)
    df['gap'] = df['gap'].clip(-0.11, 0.11)
    
    # Direction indicators
    df['gap_up'] = (df['gap'] > 0.003).astype(float)   # meaningful up gap (>0.3%)
    df['gap_down'] = (df['gap'] < -0.003).astype(float)  # meaningful down gap
    df['gap_abs'] = df['gap'].abs()
    
    # Gap momentum components (20-day window)
    print("Calculating gap momentum components...")
    window = 20
    min_periods = 15
    
    # 1. Gap direction consistency: proportion of up gaps - proportion of down gaps
    df['n_up'] = df.groupby('stock_code')['gap_up'].transform(
        lambda x: x.rolling(window, min_periods=min_periods).sum()
    )
    df['n_down'] = df.groupby('stock_code')['gap_down'].transform(
        lambda x: x.rolling(window, min_periods=min_periods).sum()
    )
    df['n_total'] = df.groupby('stock_code')['gap'].transform(
        lambda x: x.rolling(window, min_periods=min_periods).count()
    )
    df['gap_direction'] = (df['n_up'] - df['n_down']) / df['n_total'].replace(0, np.nan)
    
    # 2. Amplitude-weighted gap: sum of gaps weighted by absolute gap size
    # Emphasizes large gaps more
    df['gap_weighted'] = df['gap'] * df['gap_abs']  # sign-preserving quadratic weight
    df['gap_amp_sum'] = df.groupby('stock_code')['gap_weighted'].transform(
        lambda x: x.rolling(window, min_periods=min_periods).sum()
    )
    
    # 3. Gap concentration: are gaps bunched in recent days? (recency-weighted)
    # More weight to recent gaps
    def recency_weighted_gap(x):
        n = len(x)
        if n < min_periods:
            return np.nan
        weights = np.arange(1, n + 1, dtype=float)
        weights = weights / weights.sum()
        return np.nansum(x.values * weights)
    
    df['gap_recency'] = df.groupby('stock_code')['gap'].transform(
        lambda x: x.rolling(window, min_periods=min_periods).apply(recency_weighted_gap, raw=False)
    )
    
    # Composite factor: blend direction consistency + amplitude-weighted sum + recency
    # Normalize each component first
    def cross_zscore(group, col):
        vals = group[col]
        m, s = vals.mean(), vals.std()
        return (vals - m) / s if s > 0 else vals * 0
    
    print("Cross-sectional normalization of components...")
    df['gap_dir_z'] = df.groupby('date')['gap_direction'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    df['gap_amp_z'] = df.groupby('date')['gap_amp_sum'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    df['gap_rec_z'] = df.groupby('date')['gap_recency'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    # Equal-weighted composite
    df['raw_factor'] = (df['gap_dir_z'] + df['gap_amp_z'] + df['gap_rec_z']) / 3.0
    
    # Market cap proxy
    df['mktcap_proxy'] = df['amount'] / df['turnover'].replace(0, np.nan)
    df['log_mktcap'] = np.log(df['mktcap_proxy'].replace(0, np.nan))
    
    result = df[['date', 'stock_code', 'raw_factor', 'log_mktcap']].dropna().copy()
    
    # 5% winsorization
    print("5% winsorization...")
    def winsorize(s, lower=0.025, upper=0.975):
        lo, hi = s.quantile(lower), s.quantile(upper)
        return s.clip(lo, hi)
    
    result['raw_factor'] = result.groupby('date')['raw_factor'].transform(winsorize)
    
    # Cross-sectional z-score
    print("Cross-sectional standardization...")
    result['factor_zscore'] = result.groupby('date')['raw_factor'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    result['factor_zscore'] = result['factor_zscore'].clip(-3, 3)
    
    # Market cap neutralization (OLS)
    print("Market cap neutralization (OLS)...")
    def neutralize(group):
        g = group.dropna(subset=['factor_zscore', 'log_mktcap'])
        if len(g) < 10:
            g = g.copy()
            g['factor_neutral'] = np.nan
            return g[['factor_neutral']]
        x = g['log_mktcap'].values
        y = g['factor_zscore'].values
        x_mean = np.nanmean(x)
        y_mean = np.nanmean(y)
        b = np.nansum((x - x_mean) * (y - y_mean)) / (np.nansum((x - x_mean)**2) + 1e-10)
        a = y_mean - b * x_mean
        residuals = y - (a + b * x)
        g = g.copy()
        g['factor_neutral'] = residuals
        return g[['factor_neutral']]
    
    neutralized = result.groupby('date', group_keys=False).apply(neutralize)
    result['factor_neutral'] = neutralized['factor_neutral'].values
    
    # Final z-score (keep positive = bullish gap momentum)
    result['factor_value'] = result.groupby('date')['factor_neutral'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    result['factor_value'] = result['factor_value'].clip(-3, 3)
    
    # Output
    output = result[['date', 'stock_code', 'factor_value']].dropna()
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')
    
    out_path = data_dir / "factor_gap_momentum_v1.csv"
    output.to_csv(out_path, index=False)
    
    print(f"\nSaved to {out_path}")
    print(f"Shape: {output.shape}")
    print(f"Date range: {output['date'].min()} ~ {output['date'].max()}")
    print(f"Factor stats:")
    print(output['factor_value'].describe())

if __name__ == "__main__":
    main()
