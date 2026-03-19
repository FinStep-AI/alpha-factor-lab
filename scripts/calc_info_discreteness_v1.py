#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信息离散度因子 (Information Discreteness) v1

基于 Da, Gurun, Warachka (2014) "Frog in the Pan"
逻辑：
  ID = sgn(cum_ret_20) - (pos_days - neg_days) / 20
  
  - cum_ret_20: 过去20个交易日的累计收益率
  - pos_days: 正收益天数, neg_days: 负收益天数
  - sgn(cum_ret_20): 累计收益方向符号 (+1/-1)

  当累计涨了(sgn=+1)但正收益天数不多时，ID接近+2(极值)，
  说明上涨集中在少数几天→信息集中释放→后续反转概率高。
  
  做空信号：高ID（信息集中释放的上涨）→ 后续跑输
  做多信号：低ID（信息逐步释放/分散） → 后续跑赢

  反向使用：factor_value = -ID (因为高ID预测负收益)

市值中性化：用amount_ma20做成交额代理市值的OLS回归取残差

Barra风格: Reversal/Momentum (信息质量因子)
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import warnings

warnings.filterwarnings("ignore")

WINDOW = 20
OUTPUT_PATH = "data/factor_info_discreteness_v1.csv"

# === 1. 加载数据 ===
print("[1/5] 加载数据...")
kline = pd.read_csv("data/csi1000_kline_raw.csv")
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline['date'] = pd.to_datetime(kline['date'])
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)

print(f"  股票数: {kline['stock_code'].nunique()}, 总行数: {len(kline)}")

# === 2. 计算信息离散度 ===
print("[2/5] 计算信息离散度...")

# 计算日收益率
kline['ret'] = kline.groupby('stock_code')['close'].pct_change()

# 标记正负收益
kline['pos_day'] = (kline['ret'] > 0).astype(float)
kline['neg_day'] = (kline['ret'] < 0).astype(float)

# 滚动窗口统计
grouped = kline.groupby('stock_code')

kline['cum_ret'] = grouped['ret'].transform(
    lambda x: x.rolling(WINDOW, min_periods=15).sum()
)
kline['pos_count'] = grouped['pos_day'].transform(
    lambda x: x.rolling(WINDOW, min_periods=15).sum()
)
kline['neg_count'] = grouped['neg_day'].transform(
    lambda x: x.rolling(WINDOW, min_periods=15).sum()
)

# 信息离散度
kline['sign_cum'] = np.sign(kline['cum_ret'])
kline['info_disc'] = kline['sign_cum'] - (kline['pos_count'] - kline['neg_count']) / WINDOW

# 反向使用（高ID→负超额收益，所以取负值作为因子）
kline['raw_factor'] = -kline['info_disc']

# 去掉NaN
factor_df = kline[['date', 'stock_code', 'raw_factor']].dropna().copy()
print(f"  原始因子值: {len(factor_df)} 条")
print(f"  日期范围: {factor_df['date'].min()} ~ {factor_df['date'].max()}")

# === 3. 市值中性化 ===
print("[3/5] 市值中性化 (用amount_ma20代理)...")

kline['amount_ma20'] = grouped['amount'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

factor_df = factor_df.merge(
    kline[['date', 'stock_code', 'amount_ma20']],
    on=['date', 'stock_code'],
    how='left'
)

def neutralize_cross_section(group):
    y = group['raw_factor'].values
    x = np.log(group['amount_ma20'].values + 1)
    mask = np.isfinite(y) & np.isfinite(x) & (x > 0)
    if mask.sum() < 50:
        group['factor_value'] = np.nan
        return group
    slope, intercept, _, _, _ = sp_stats.linregress(x[mask], y[mask])
    residuals = np.full(len(y), np.nan)
    residuals[mask] = y[mask] - (intercept + slope * x[mask])
    group['factor_value'] = residuals
    return group

factor_df = factor_df.groupby('date', group_keys=False).apply(neutralize_cross_section)

# === 4. Winsorize + Z-score ===
print("[4/5] Winsorize + Z-score...")

def winsorize_mad(s, n_mad=5):
    median = s.median()
    mad = (s - median).abs().median()
    if mad == 0:
        return s
    lower = median - n_mad * mad * 1.4826
    upper = median + n_mad * mad * 1.4826
    return s.clip(lower, upper)

def cross_section_normalize(group):
    vals = group['factor_value']
    vals = winsorize_mad(vals)
    mean = vals.mean()
    std = vals.std()
    if std == 0 or np.isnan(std):
        group['factor_value'] = 0.0
    else:
        group['factor_value'] = (vals - mean) / std
    return group

factor_df = factor_df.groupby('date', group_keys=False).apply(cross_section_normalize)
factor_df = factor_df.dropna(subset=['factor_value'])

# === 5. 输出 ===
print("[5/5] 输出...")
output = factor_df[['date', 'stock_code', 'factor_value']].copy()
output['date'] = output['date'].dt.strftime('%Y-%m-%d')
output = output.sort_values(['date', 'stock_code'])
output.to_csv(OUTPUT_PATH, index=False)

print(f"  输出: {OUTPUT_PATH}")
print(f"  行数: {len(output)}")
print(f"  日期范围: {output['date'].min()} ~ {output['date'].max()}")
print(f"  每日平均股票数: {output.groupby('date').size().mean():.0f}")

# 检查因子分布
print("\n因子值分布:")
print(output['factor_value'].describe())
print("\n完成!")
