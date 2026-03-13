#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量价非对称因子 (vol_ret_asym_v1)

逻辑：
  衡量上涨日和下跌日的成交量不对称性。
  
  计算过去N日：
  - 上涨日平均成交量 vs 下跌日平均成交量
  - 因子 = log(avg_vol_up / avg_vol_down)
  
  上涨放量+下跌缩量 → 因子值高 → 买方力量主导 → 看涨
  下跌放量+上涨缩量 → 因子值低 → 卖方力量主导 → 看跌
  
  这是经典的OBV衍生因子，衡量资金流向方向。

Barra风格: 微观结构（资金流向）
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

print("[2/4] 计算量价非对称因子...")

def calc_vol_ret_asym(group):
    group = group.sort_values('date')
    close = group['close'].values
    volume = group['volume'].values
    n = len(close)
    factor = np.full(n, np.nan)
    
    # 日收益率
    ret = np.zeros(n)
    ret[1:] = (close[1:] - close[:-1]) / close[:-1]
    ret[0] = np.nan
    
    for i in range(N, n):
        window_ret = ret[i - N + 1:i + 1]
        window_vol = volume[i - N + 1:i + 1]
        
        up_mask = window_ret > 0
        down_mask = window_ret < 0
        
        up_count = np.sum(up_mask)
        down_count = np.sum(down_mask)
        
        if up_count < 3 or down_count < 3:
            continue
        
        avg_vol_up = np.mean(window_vol[up_mask])
        avg_vol_down = np.mean(window_vol[down_mask])
        
        if avg_vol_down > 0 and avg_vol_up > 0:
            factor[i] = np.log(avg_vol_up / avg_vol_down)
    
    group['raw_factor'] = factor
    return group

results = []
stocks = kline['stock_code'].unique()
for i, stock in enumerate(stocks):
    grp = kline[kline['stock_code'] == stock].copy()
    res = calc_vol_ret_asym(grp)
    results.append(res)
    if (i + 1) % 200 == 0:
        print(f"  处理: {i+1}/{len(stocks)}")

factor_df = pd.concat(results, ignore_index=True)
factor_df = factor_df.dropna(subset=['raw_factor'])
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
output.to_csv('data/factor_vol_ret_asym_v1.csv', index=False)
print(f"  输出: data/factor_vol_ret_asym_v1.csv ({len(output)} 行)")
print(f"  日期范围: {output['date'].min()} ~ {output['date'].max()}")
print(f"  每日平均: {output.groupby('date').size().mean():.0f} 股")

# 也测试反向
output_neg = output.copy()
output_neg['factor_value'] = -output_neg['factor_value']
output_neg.to_csv('data/factor_vol_ret_asym_neg_v1.csv', index=False)
print("完成! (同时输出反向因子)")
