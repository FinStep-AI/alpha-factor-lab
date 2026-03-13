#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
振幅集中度因子 (amp_concentration_v1)

逻辑：
  计算过去N日的日振幅(amplitude)的集中度。
  使用Herfindahl指数衡量振幅是否集中在少数交易日。
  
  HHI = sum((amp_i / sum_amp)^2) for i in [1..N]
  
  高HHI = 振幅集中在少数日→说明有少数大波动事件（跳跃型）
  低HHI = 振幅均匀分散→说明波动稳定（扩散型）
  
  假说：振幅均匀分散的股票（低HHI）表现更好，
  因为稳定的波动意味着价格发现更有序。
  
  反向使用：-HHI 作为因子值（越低HHI→越高因子值）

Barra风格: Volatility / Quality（波动质量）
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import warnings

warnings.filterwarnings("ignore")

N = 20

print("[1/4] 加载数据...")
kline = pd.read_csv("data/csi1000_kline_raw.csv")
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline['date'] = pd.to_datetime(kline['date'])
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)

print("[2/4] 计算振幅集中度 (HHI)...")

# 使用vectorized rolling计算
def calc_amp_hhi(group):
    group = group.sort_values('date')
    amp = group['amplitude'].values
    n = len(amp)
    hhi = np.full(n, np.nan)
    
    for i in range(N - 1, n):
        window = amp[i - N + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 10:
            continue
        total = np.sum(valid)
        if total <= 0:
            continue
        shares = valid / total
        hhi[i] = np.sum(shares ** 2)
    
    group['amp_hhi'] = hhi
    return group

results = []
stocks = kline['stock_code'].unique()
for i, stock in enumerate(stocks):
    grp = kline[kline['stock_code'] == stock].copy()
    res = calc_amp_hhi(grp)
    results.append(res)
    if (i + 1) % 200 == 0:
        print(f"  处理: {i+1}/{len(stocks)}")

factor_df = pd.concat(results, ignore_index=True)
factor_df = factor_df.dropna(subset=['amp_hhi'])

# 反向：低HHI → 高因子值
factor_df['raw_factor'] = -factor_df['amp_hhi']
print(f"  有效记录: {len(factor_df)}")

print("[3/4] 市值中性化 + 标准化...")

factor_df['amount_ma20'] = factor_df.groupby('stock_code')['amount'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

def neutralize_and_normalize(group):
    y = group['raw_factor'].values
    x = np.log(group['amount_ma20'].values + 1)
    mask = np.isfinite(y) & np.isfinite(x) & (x > 0)
    if mask.sum() < 30:
        group['factor_value'] = np.nan
        return group
    slope, intercept, _, _, _ = sp_stats.linregress(x[mask], y[mask])
    residuals = np.full(len(y), np.nan)
    residuals[mask] = y[mask] - (intercept + slope * x[mask])
    
    res_series = pd.Series(residuals[mask])
    median = res_series.median()
    mad = (res_series - median).abs().median()
    if mad > 0:
        lower = median - 5 * mad * 1.4826
        upper = median + 5 * mad * 1.4826
        residuals[mask] = np.clip(residuals[mask], lower, upper)
    
    valid = residuals[mask]
    mean, std = np.nanmean(valid), np.nanstd(valid)
    if std > 0:
        residuals[mask] = (residuals[mask] - mean) / std
    group['factor_value'] = residuals
    return group

factor_df = factor_df.groupby('date', group_keys=False).apply(neutralize_and_normalize)
factor_df = factor_df.dropna(subset=['factor_value'])

print("[4/4] 输出...")
output = factor_df[['date', 'stock_code', 'factor_value']].copy()
output['date'] = output['date'].dt.strftime('%Y-%m-%d')
output = output.sort_values(['date', 'stock_code'])
output.to_csv('data/factor_amp_concentration_v1.csv', index=False)
print(f"  输出: data/factor_amp_concentration_v1.csv ({len(output)} 行)")
print(f"  日期范围: {output['date'].min()} ~ {output['date'].max()}")
print(f"  每日平均: {output.groupby('date').size().mean():.0f} 股")
print("完成!")
