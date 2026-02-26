#!/usr/bin/env python3
"""
因子: 隔夜收益动量 (Overnight Return Momentum) v1
factor_id: overnight_momentum_v1

逻辑:
  - 隔夜收益 = (今日open - 昨日close) / 昨日close
  - 日内收益 = (今日close - 今日open) / 今日open  
  - 因子 = 过去20日累计隔夜收益 - 过去20日累计日内收益
  - 正值 = 隔夜一直涨但日内跌 → 可能有知情交易者在盘后买入
  - 做市值中性化处理

假设:
  - A股有显著的隔夜效应(overnight premium)
  - 隔夜收益包含更多信息(机构/知情交易者偏好集合竞价)
  - 当隔夜持续强于日内时，后续股价倾向上涨
  
参考:
  - Lou, Polk, Skouras (2019) "A Tug of War: Overnight vs Intraday Expected Returns"
  - 国泰君安《隔夜收益与日内收益的因子效应》
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
    
    # Calculate overnight and intraday returns
    df['prev_close'] = df.groupby('stock_code')['close'].shift(1)
    df['overnight_ret'] = (df['open'] - df['prev_close']) / df['prev_close']
    df['intraday_ret'] = (df['close'] - df['open']) / df['open']
    
    # Clip extremes (limit up/down artifacts)
    df['overnight_ret'] = df['overnight_ret'].clip(-0.11, 0.11)
    df['intraday_ret'] = df['intraday_ret'].clip(-0.11, 0.11)
    
    # Rolling 20-day cumulative overnight and intraday returns
    print("Calculating rolling overnight momentum...")
    df['cum_overnight_20'] = df.groupby('stock_code')['overnight_ret'].transform(
        lambda x: x.rolling(20, min_periods=15).sum()
    )
    df['cum_intraday_20'] = df.groupby('stock_code')['intraday_ret'].transform(
        lambda x: x.rolling(20, min_periods=15).sum()
    )
    
    # Factor = overnight - intraday (positive = overnight dominates)
    df['raw_factor'] = df['cum_overnight_20'] - df['cum_intraday_20']
    
    # Also compute a version with just overnight as the main signal
    df['raw_factor_overnight_only'] = df['cum_overnight_20']
    
    # Market cap proxy
    df['mktcap_proxy'] = df['amount'] / df['turnover'].replace(0, np.nan)
    df['log_mktcap'] = np.log(df['mktcap_proxy'].replace(0, np.nan))
    
    # Select factor
    result = df[['date', 'stock_code', 'raw_factor', 'log_mktcap']].dropna().copy()
    
    # Cross-sectional z-score
    print("Cross-sectional standardization...")
    result['factor_zscore'] = result.groupby('date')['raw_factor'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    result['factor_zscore'] = result['factor_zscore'].clip(-3, 3)
    
    # Market cap neutralization
    print("Market cap neutralization...")
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
    
    # Final z-score
    result['factor_value'] = result.groupby('date')['factor_neutral'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    result['factor_value'] = result['factor_value'].clip(-3, 3)
    
    # Output
    output = result[['date', 'stock_code', 'factor_value']].dropna()
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')
    
    out_path = data_dir / "factor_overnight_momentum_v1.csv"
    output.to_csv(out_path, index=False)
    
    print(f"\nSaved to {out_path}")
    print(f"Shape: {output.shape}")
    print(f"Date range: {output['date'].min()} ~ {output['date'].max()}")
    print(f"Factor stats:")
    print(output['factor_value'].describe())

if __name__ == "__main__":
    main()
