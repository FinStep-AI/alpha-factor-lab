#!/usr/bin/env python3
"""
因子: 换手率衰减 (Turnover Decay) v1
factor_id: turnover_decay_v1

逻辑:
  - 近5日平均换手率 / 过去20日平均换手率 的比值
  - 比值 < 1 表示换手率在萎缩（"量能衰减"）
  - 取对数后反向使用: 换手率衰减（ratio小）→ 后续可能反弹
  - 经市值中性化 + 5%缩尾处理

假设:
  - 换手率持续萎缩意味着卖压耗尽、筹码趋于集中
  - 中证1000小盘股在量能极度萎缩后倾向反弹（均值回复）
  - 反向使用: 低衰减比（量能枯竭）→ 做多

参考:
  - Datar, Naik, Radcliffe (1998) "Liquidity and Stock Returns"
  - 华泰证券《换手率因子深度研究》
  - 国金证券《量能衰竭与短期反转效应》
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
    
    # Turnover is already in the data (%)
    # Need to handle zero/missing turnover
    df['turnover_clean'] = df['turnover'].replace(0, np.nan)
    
    # Rolling averages: short (5d) vs long (20d)
    print("Calculating turnover decay ratio...")
    df['turnover_ma5'] = df.groupby('stock_code')['turnover_clean'].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )
    df['turnover_ma20'] = df.groupby('stock_code')['turnover_clean'].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )
    
    # Decay ratio = short / long (< 1 means turnover is shrinking)
    df['decay_ratio'] = df['turnover_ma5'] / df['turnover_ma20'].replace(0, np.nan)
    
    # Log transform for better distribution
    df['raw_factor'] = np.log(df['decay_ratio'].replace(0, np.nan))
    
    # Market cap proxy for neutralization
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
    
    # Reverse sign: low decay ratio (turnover drying up) → POSITIVE factor value → long
    # This means we negate: stocks with shrinking turnover get high factor values
    result['factor_value'] = -1.0 * result.groupby('date')['factor_neutral'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    result['factor_value'] = result['factor_value'].clip(-3, 3)
    
    # Output
    output = result[['date', 'stock_code', 'factor_value']].dropna()
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')
    
    out_path = data_dir / "factor_turnover_decay_v1.csv"
    output.to_csv(out_path, index=False)
    
    print(f"\nSaved to {out_path}")
    print(f"Shape: {output.shape}")
    print(f"Date range: {output['date'].min()} ~ {output['date'].max()}")
    print(f"Factor stats:")
    print(output['factor_value'].describe())

if __name__ == "__main__":
    main()
