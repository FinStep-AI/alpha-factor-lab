#!/usr/bin/env python3
"""
构建两个新因子:
1. realized_skew_v1 - 20日实现偏度（反向）
2. vol_price_corr_v1 - 20日量价相关性

数据源: csi1000_kline_raw.csv
输出: data/factor_realized_skew_v1.csv, data/factor_vol_price_corr_v1.csv
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ---- Load data ----
print("Loading data...")
df = pd.read_csv('data/csi1000_kline_raw.csv', parse_dates=['date'])
df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)

# Daily returns
df['ret'] = df.groupby('stock_code')['close'].pct_change()

# Log market cap proxy (close * volume as rough proxy, or use amount)
# Actually we need a proper market cap. Let's use amount/turnover as share_float proxy
# share_float ≈ amount / (close * turnover/100) ... but simpler: use log(close * volume) as size proxy
# Better: just use log(amount) as market cap proxy for neutralization
df['log_mktcap'] = np.log(df['amount'].clip(lower=1))

print(f"Total rows: {len(df)}, stocks: {df['stock_code'].nunique()}")

# ---- Factor 1: Realized Skewness ----
print("\n=== Building Factor 1: Realized Skewness (20d) ===")

def rolling_skew(group):
    """20日滚动偏度"""
    return group['ret'].rolling(window=20, min_periods=15).skew()

df['skew_20d'] = df.groupby('stock_code', group_keys=False).apply(rolling_skew)

# 反向使用：负偏度 → 高因子值
df['raw_skew_factor'] = -df['skew_20d']

# ---- Factor 2: Volume-Price Correlation ----
print("=== Building Factor 2: Volume-Price Correlation (20d) ===")

def rolling_vol_price_corr(group):
    """20日滚动：成交量与收益率的相关系数"""
    return group['volume'].rolling(window=20, min_periods=15).corr(group['ret'])

df['vol_price_corr_20d'] = df.groupby('stock_code', group_keys=False).apply(rolling_vol_price_corr)

# ---- Cross-sectional neutralization ----
print("\n=== Market Cap Neutralization ===")

def neutralize_cross_section(df_day, factor_col, neutral_col='log_mktcap'):
    """截面回归取残差做市值中性化"""
    mask = df_day[factor_col].notna() & df_day[neutral_col].notna() & np.isfinite(df_day[factor_col]) & np.isfinite(df_day[neutral_col])
    result = pd.Series(np.nan, index=df_day.index)
    if mask.sum() < 30:
        return result
    x = df_day.loc[mask, neutral_col].values
    y = df_day.loc[mask, factor_col].values
    
    # Winsorize outliers (MAD method)
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad > 0:
        lower = med - 5 * 1.4826 * mad
        upper = med + 5 * 1.4826 * mad
        y = np.clip(y, lower, upper)
    
    # OLS
    x_with_const = np.column_stack([np.ones_like(x), x])
    try:
        beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
        residuals = y - x_with_const @ beta
        # Z-score
        std = residuals.std()
        if std > 0:
            residuals = (residuals - residuals.mean()) / std
        result.iloc[mask.values.nonzero()[0]] = residuals
    except:
        pass
    return result

# Process each date
dates = df['date'].unique()
print(f"Processing {len(dates)} dates...")

factor_skew_list = []
factor_vpcorr_list = []

for i, d in enumerate(dates):
    if i % 100 == 0:
        print(f"  Date {i}/{len(dates)}: {d}")
    day_mask = df['date'] == d
    day_df = df[day_mask].copy()
    
    # Skew factor neutralization
    skew_neutral = neutralize_cross_section(day_df, 'raw_skew_factor')
    factor_skew_list.append(skew_neutral)
    
    # Vol-price corr neutralization
    vpcorr_neutral = neutralize_cross_section(day_df, 'vol_price_corr_20d')
    factor_vpcorr_list.append(vpcorr_neutral)

df['factor_skew_neutral'] = pd.concat(factor_skew_list)
df['factor_vpcorr_neutral'] = pd.concat(factor_vpcorr_list)

# ---- Output ----
print("\n=== Saving factors ===")

# Factor 1: Realized Skewness
out_skew = df[['date', 'stock_code', 'factor_skew_neutral']].dropna(subset=['factor_skew_neutral'])
out_skew = out_skew.rename(columns={'factor_skew_neutral': 'factor_value'})
out_skew.to_csv('data/factor_realized_skew_v1.csv', index=False)
print(f"Factor 1 (Realized Skewness): {len(out_skew)} rows, {out_skew['stock_code'].nunique()} stocks")
print(f"  Date range: {out_skew['date'].min()} ~ {out_skew['date'].max()}")
print(f"  Factor stats: mean={out_skew['factor_value'].mean():.4f}, std={out_skew['factor_value'].std():.4f}")

# Factor 2: Vol-Price Correlation
out_vpcorr = df[['date', 'stock_code', 'factor_vpcorr_neutral']].dropna(subset=['factor_vpcorr_neutral'])
out_vpcorr = out_vpcorr.rename(columns={'factor_vpcorr_neutral': 'factor_value'})
out_vpcorr.to_csv('data/factor_vol_price_corr_v1.csv', index=False)
print(f"Factor 2 (Vol-Price Corr): {len(out_vpcorr)} rows, {out_vpcorr['stock_code'].nunique()} stocks")
print(f"  Date range: {out_vpcorr['date'].min()} ~ {out_vpcorr['date'].max()}")
print(f"  Factor stats: mean={out_vpcorr['factor_value'].mean():.4f}, std={out_vpcorr['factor_value'].std():.4f}")

print("\nDone!")
