#!/usr/bin/env python3
"""
因子: 量价相关性 (Volume-Price Correlation) v1
factor_id: vol_price_corr_v1

逻辑: 
  - 计算过去20日成交量变化率与收益率的滚动相关系数
  - 高正相关 = 量价齐升/齐跌 = 趋势确认 → 动量延续
  - 低/负相关 = 量价背离 → 可能反转
  - 做市值中性化处理
  
假设:
  - 中证1000小盘股中，量价正相关的股票有持续动量
  - 正向因子(高相关→高收益)

参考:
  - 国信证券《量价因子深度研究》
  - 华泰证券《因子库更新：量价相关性因子》
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
# Use numpy OLS instead of sklearn

def main():
    data_dir = Path("data")
    
    # Load kline data
    print("Loading kline data...")
    df = pd.read_csv(data_dir / "csi1000_kline_raw.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date'])
    
    print(f"Stocks: {df['stock_code'].nunique()}, Dates: {df['date'].min()} ~ {df['date'].max()}")
    
    # Calculate daily return and volume change
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    df['vol_chg'] = df.groupby('stock_code')['volume'].pct_change()
    
    # Clip extreme volume changes (>500% is noise)
    df['vol_chg'] = df['vol_chg'].clip(-5, 5)
    
    # Rolling 20-day correlation between volume change and return
    print("Calculating rolling 20-day vol-price correlation...")
    
    def rolling_corr(group):
        group = group.copy()
        group['vp_corr'] = group['ret'].rolling(20, min_periods=15).corr(group['vol_chg'])
        return group
    
    df = df.groupby('stock_code', group_keys=False).apply(rolling_corr)
    
    # Also compute a 5-day short-term version for comparison
    def rolling_corr_5d(group):
        group = group.copy()
        group['vp_corr_5d'] = group['ret'].rolling(5, min_periods=4).corr(group['vol_chg'])
        return group
    
    df = df.groupby('stock_code', group_keys=False).apply(rolling_corr_5d)
    
    # Composite: weighted average of 20d (70%) and 5d (30%) 
    df['vp_corr_composite'] = 0.7 * df['vp_corr'] + 0.3 * df['vp_corr_5d']
    
    # Drop NaN
    factor_col = 'vp_corr_composite'
    result = df[['date', 'stock_code', factor_col]].dropna().copy()
    
    # Cross-sectional z-score per date
    print("Cross-sectional standardization...")
    result['factor_zscore'] = result.groupby('date')[factor_col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    # Winsorize at ±3 std
    result['factor_zscore'] = result['factor_zscore'].clip(-3, 3)
    
    # Market cap neutralization
    print("Market cap neutralization...")
    # We need market cap proxy. Use amount/turnover as proxy for market cap
    # amount ≈ price * volume, turnover = volume/total_shares
    # market_cap ≈ amount / turnover (roughly proportional)
    df['mktcap_proxy'] = df['amount'] / df['turnover'].replace(0, np.nan)
    df['log_mktcap'] = np.log(df['mktcap_proxy'].replace(0, np.nan))
    
    # Merge mktcap back
    result = result.merge(
        df[['date', 'stock_code', 'log_mktcap']],
        on=['date', 'stock_code'],
        how='left'
    )
    
    # Neutralize: regress factor on log_mktcap, take residual per date
    def neutralize(group):
        g = group.dropna(subset=['factor_zscore', 'log_mktcap'])
        if len(g) < 10:
            g['factor_neutral'] = np.nan
            return g[['factor_neutral']]
        x = g['log_mktcap'].values
        y = g['factor_zscore'].values
        # Simple OLS: y = a + b*x + residual
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
    
    # Final z-score after neutralization
    result['factor_value'] = result.groupby('date')['factor_neutral'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    result['factor_value'] = result['factor_value'].clip(-3, 3)
    
    # Output
    output = result[['date', 'stock_code', 'factor_value']].dropna()
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')
    
    out_path = data_dir / "factor_vol_price_corr_v1.csv"
    output.to_csv(out_path, index=False)
    
    print(f"\nSaved to {out_path}")
    print(f"Shape: {output.shape}")
    print(f"Date range: {output['date'].min()} ~ {output['date'].max()}")
    print(f"Factor stats:")
    print(output['factor_value'].describe())
    
    # Quick IC preview
    print("\n--- Quick IC Preview ---")
    returns = pd.read_csv(data_dir / "csi1000_returns.csv")
    returns['date'] = pd.to_datetime(returns['date'])
    
    # Forward 20-day return
    returns = returns.sort_values(['stock_code', 'date'])
    returns['fwd_ret_20'] = returns.groupby('stock_code')['daily_return'].transform(
        lambda x: x.shift(-1).rolling(20, min_periods=15).sum()
    )
    
    merged = output.copy()
    merged['date'] = pd.to_datetime(merged['date'])
    merged = merged.merge(returns[['date', 'stock_code', 'fwd_ret_20']], on=['date', 'stock_code'], how='inner')
    merged = merged.dropna()
    
    # IC per date
    ic_series = merged.groupby('date').apply(
        lambda g: g['factor_value'].corr(g['fwd_ret_20']) if len(g) > 10 else np.nan
    ).dropna()
    
    print(f"IC mean: {ic_series.mean():.4f}")
    print(f"IC std: {ic_series.std():.4f}")
    print(f"IC t-stat: {ic_series.mean() / ic_series.std() * np.sqrt(len(ic_series)):.2f}")
    print(f"IC positive ratio: {(ic_series > 0).mean():.3f}")
    print(f"IC count: {len(ic_series)}")

if __name__ == "__main__":
    main()
