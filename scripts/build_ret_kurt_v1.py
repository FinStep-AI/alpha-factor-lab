#!/usr/bin/env python3 -u
"""构建收益率峰度因子 v1 (ret_kurt_v1)"""
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

print("Loading data...", flush=True)
df = pd.read_csv('data/csi1000_kline_raw.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
df['prev_close'] = df.groupby('stock_code')['close'].shift(1)
df['daily_ret'] = (df['close'] - df['prev_close']) / df['prev_close']
df['log_amount'] = np.log(df['amount'] + 1)
df = df[df['daily_ret'].notna() & (df['daily_ret'].abs() < 0.21)].copy()
print(f"数据: {len(df):,} 行, {df['stock_code'].nunique()} 只", flush=True)

# 20日超额峰度
print("Computing rolling kurtosis...", flush=True)

def kurt_20d(group):
    return group.rolling(20, min_periods=15).apply(
        lambda x: scipy_stats.kurtosis(x, fisher=True), raw=True
    )

df['kurt_20d'] = df.groupby('stock_code')['daily_ret'].transform(kurt_20d)
kurt_data = df[['date', 'stock_code', 'kurt_20d', 'log_amount']].dropna()
print(f"Kurtosis done: {len(kurt_data):,} rows", flush=True)

# OLS中性化
factor_records = []
dates_list = sorted(kurt_data['date'].unique())
print(f"截面天数: {len(dates_list)}, 开始中性化...", flush=True)

from numpy.linalg import lstsq

for i, dt in enumerate(dates_list):
    if i % 100 == 0:
        print(f"  [{i}] {dt}...", flush=True)
    sub = kurt_data[kurt_data['date'] == dt].copy()
    if len(sub) < 50:
        continue
    y = sub['kurt_20d'].values.reshape(-1, 1)
    x = np.column_stack([np.ones(len(sub)), sub['log_amount'].values])
    beta, _, _, _ = lstsq(x, y, rcond=None)
    residuals = y.flatten() - x @ beta
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    if mad > 0:
        upper = med + 5.5 * 1.4826 * mad
        lower = med - 5.5 * 1.4826 * mad
        residuals = np.clip(residuals, lower, upper)
    mean_f, std_f = residuals.mean(), residuals.std()
    if std_f > 0:
        z = (residuals - mean_f) / std_f
        for idx, row in sub.iterrows():
            factor_records.append({'date': dt, 'code': row['stock_code'], 'ret_kurt_v1': z[sub.index.get_loc(idx)]})

print(f"\n中性化完成: {len(factor_records)} 条", flush=True)
factor_df = pd.DataFrame(factor_records)
factor_df['date'] = pd.to_datetime(factor_df['date']).dt.strftime('%Y-%m-%d')
factor_df = factor_df.sort_values(['date', 'code']).reset_index(drop=True)

out_path = 'data/factor_ret_kurt_v1.csv'
factor_df.to_csv(out_path, index=False)
print(f"\n✅ 保存: {out_path}")
print(f"  行数: {len(factor_df):,}, 日期: {factor_df['date'].iloc[0]} ~ {factor_df['date'].iloc[-1]}")
