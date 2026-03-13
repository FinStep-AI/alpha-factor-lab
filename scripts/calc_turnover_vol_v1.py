#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成交额波动率因子 (turnover_vol_v1)

逻辑：
  计算过去20日换手率的变异系数(CV = std/mean)。
  换手率波动大 → 交易行为不稳定 → 信息不对称程度高
  
  反向因子：CV越低（交易越稳定）→ 因子值越高 → 预期收益越高
  
  原理：换手率稳定的股票，说明市场参与者对其定价有共识，
  信息传播较为充分，这类股票的定价效率更高，
  而高换手率波动的股票面临更多不确定性和噪音交易。

Barra风格: Quality（交易质量）
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import warnings

warnings.filterwarnings("ignore")

N = 20  # 回看窗口

print("[1/4] 加载数据...")
kline = pd.read_csv("data/csi1000_kline_raw.csv")
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline['date'] = pd.to_datetime(kline['date'])
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)

print("[2/4] 计算换手率变异系数...")

def calc_turnover_cv(group):
    group = group.sort_values('date')
    turnover = group['turnover'].values
    n = len(turnover)
    cv = np.full(n, np.nan)
    
    for i in range(N - 1, n):
        window = turnover[i - N + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 10 and np.mean(valid) > 0:
            cv[i] = np.std(valid) / np.mean(valid)
    
    group['turnover_cv'] = cv
    return group

results = []
stocks = kline['stock_code'].unique()
for i, stock in enumerate(stocks):
    grp = kline[kline['stock_code'] == stock].copy()
    res = calc_turnover_cv(grp)
    results.append(res)
    if (i + 1) % 200 == 0:
        print(f"  处理: {i+1}/{len(stocks)}")

factor_df = pd.concat(results, ignore_index=True)
factor_df = factor_df.dropna(subset=['turnover_cv'])

# 反向：低CV → 高因子值
factor_df['raw_factor'] = -factor_df['turnover_cv']
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
    mean = np.nanmean(valid)
    std = np.nanstd(valid)
    if std > 0:
        residuals[mask] = (residuals[mask] - mean) / std
    else:
        residuals[mask] = 0.0
    
    group['factor_value'] = residuals
    return group

factor_df = factor_df.groupby('date', group_keys=False).apply(neutralize_and_normalize)
factor_df = factor_df.dropna(subset=['factor_value'])

print("[4/4] 输出因子CSV...")
output = factor_df[['date', 'stock_code', 'factor_value']].copy()
output['date'] = output['date'].dt.strftime('%Y-%m-%d')
output = output.sort_values(['date', 'stock_code'])
output.to_csv('data/factor_turnover_vol_v1.csv', index=False)
print(f"  输出: data/factor_turnover_vol_v1.csv ({len(output)} 行)")
print(f"  日期范围: {output['date'].min()} ~ {output['date'].max()}")
print(f"  每日平均股票数: {output.groupby('date').size().mean():.0f}")
print("完成!")
