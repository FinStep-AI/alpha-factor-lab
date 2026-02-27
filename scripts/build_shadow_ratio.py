#!/usr/bin/env python3
"""
构建因子: 日内多空力量比 (Intraday Bear-Bull Ratio)
= 20日均值 of (high - close) / (close - low + epsilon)
高值 = 上影线长（日内多头被打压），低值 = 下影线长（日内空头被打回）
假设：上影线长的股票短期内反转概率高（散户追高被砸），反向做多下影线长的

方向：负向（低因子值=下影线长=多头力量更强→高收益）
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv('data/csi1000_kline_raw.csv', parse_dates=['date'])
df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)

# 市值代理
df['log_mktcap'] = np.log(df['amount'].clip(lower=1))

# 日内上下影线比
eps = 1e-8
df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
df['body'] = (df['close'] - df['open']).abs()
df['total_range'] = df['high'] - df['low']

# 上影线占比 (0~1)
df['upper_ratio'] = df['upper_shadow'] / (df['total_range'] + eps)

# 20日滚动均值
df['upper_ratio_20d'] = df.groupby('stock_code')['upper_ratio'].transform(
    lambda x: x.rolling(window=20, min_periods=15).mean()
)

# 反向使用：高上影线比=做空信号，取负值
df['raw_factor'] = -df['upper_ratio_20d']

# Cross-sectional market-cap neutralization
print("Market cap neutralization...")
dates = df['date'].unique()
factor_list = []

for i, d in enumerate(dates):
    if i % 100 == 0:
        print(f"  Date {i}/{len(dates)}: {d}")
    day_df = df[df['date'] == d].copy()
    mask = day_df['raw_factor'].notna() & day_df['log_mktcap'].notna() & np.isfinite(day_df['raw_factor'])
    result = pd.Series(np.nan, index=day_df.index)
    
    if mask.sum() < 30:
        factor_list.append(result)
        continue
    
    y = day_df.loc[mask, 'raw_factor'].values
    x = day_df.loc[mask, 'log_mktcap'].values
    
    # MAD winsorize
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
        std = residuals.std()
        if std > 0:
            residuals = (residuals - residuals.mean()) / std
        result.iloc[mask.values.nonzero()[0]] = residuals
    except:
        pass
    factor_list.append(result)

df['factor_neutral'] = pd.concat(factor_list)

# Save
out = df[['date', 'stock_code', 'factor_neutral']].dropna(subset=['factor_neutral'])
out = out.rename(columns={'factor_neutral': 'factor_value'})
out.to_csv('data/factor_shadow_ratio_v1.csv', index=False)
print(f"\nSaved: {len(out)} rows, {out['stock_code'].nunique()} stocks")
print(f"Date range: {out['date'].min()} ~ {out['date'].max()}")
print(f"Stats: mean={out['factor_value'].mean():.4f}, std={out['factor_value'].std():.4f}")
print("Done!")
