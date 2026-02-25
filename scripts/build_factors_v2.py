#!/usr/bin/env python3
"""
升级版因子构建 — 中证1000
包含：市值中性化（用流通市值代理：amount/turnover）
因子列表：
  1. amihud_illiq_v2: Amihud非流动性（升级）
  2. revision_accel_v2: 预期修正加速度（升级）
  3. amplitude_fatigue_v2: 振幅疲劳（升级）
"""
import pandas as pd
import numpy as np
import os, sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')

print("=== 加载中证1000 K线数据 ===")
df = pd.read_csv(os.path.join(DATA_DIR, 'csi1000_kline_raw.csv'))
df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)

print(f"  {df['stock_code'].nunique()} 只股票, {df['date'].nunique()} 交易日, {len(df)} 行")

# === 市值代理：用 amount/turnover*100 作为流通市值代理 ===
df['mktcap_proxy'] = df['amount'] / (df['turnover'].clip(lower=0.01) / 100)
df['log_mktcap'] = np.log(df['mktcap_proxy'].clip(lower=1))

# === 辅助函数 ===
def neutralize_by_mktcap(factor_series, mktcap_series):
    """市值中性化：OLS回归取残差"""
    mask = factor_series.notna() & mktcap_series.notna()
    if mask.sum() < 10:
        return factor_series
    x = mktcap_series[mask].values.reshape(-1, 1)
    y = factor_series[mask].values
    # OLS
    x_with_const = np.column_stack([np.ones(len(x)), x])
    try:
        beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
        resid = pd.Series(np.nan, index=factor_series.index)
        resid[mask] = y - x_with_const @ beta
        return resid
    except:
        return factor_series

def cross_section_zscore(group):
    """截面标准化"""
    mean = group.mean()
    std = group.std()
    if std == 0 or pd.isna(std):
        return group * 0
    return (group - mean) / std

def winsorize(s, lower=0.01, upper=0.99):
    """缩尾处理"""
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)

# ====================================================================
# 因子1: Amihud非流动性 v2
# 原始: |ret| / amount, 20日均值
# 升级: 加入市值中性化 + 缩尾 + 更稳健的计算
# ====================================================================
print("\n=== 因子1: Amihud非流动性 v2 ===")

df['abs_ret'] = df['pct_change'].abs() / 100
df['amihud_raw'] = df['abs_ret'] / (df['amount'].clip(lower=1) / 1e8)  # 单位亿

# 20日滚动均值
df['amihud_20d'] = df.groupby('stock_code')['amihud_raw'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

# 对数变换（减少右偏）
df['amihud_log'] = np.log(df['amihud_20d'].clip(lower=1e-10))

# 截面缩尾
df['amihud_log_w'] = df.groupby('date')['amihud_log'].transform(
    lambda x: winsorize(x, 0.02, 0.98)
)

# 市值中性化
factor1_list = []
for dt, gdf in df.groupby('date'):
    resid = neutralize_by_mktcap(gdf['amihud_log_w'], gdf['log_mktcap'])
    tmp = gdf[['date', 'stock_code']].copy()
    tmp['amihud_illiq_v2'] = resid.values
    factor1_list.append(tmp)

factor1 = pd.concat(factor1_list, ignore_index=True)

# 截面标准化
factor1['amihud_illiq_v2'] = factor1.groupby('date')['amihud_illiq_v2'].transform(cross_section_zscore)

f1_path = os.path.join(DATA_DIR, 'factor_amihud_illiq_v2.csv')
factor1.to_csv(f1_path, index=False)
non_null = factor1['amihud_illiq_v2'].notna().sum()
print(f"  保存: {f1_path}, 非空因子值: {non_null}")

# ====================================================================
# 因子2: 预期修正加速度 v2 (超额收益代理)
# 原始: (近20日累计超额 - 前20日累计超额) / (40日超额波动×√20)
# 升级: 用中证1000指数作基准 + 市值中性化 + 参数微调
# ====================================================================
print("\n=== 因子2: 预期修正加速度 v2 ===")

# 计算等权指数作基准
index_ret = df.groupby('date')['pct_change'].mean().rename('index_ret')
df = df.merge(index_ret, on='date', how='left')
df['excess_ret'] = df['pct_change'] - df['index_ret']

# 20日滚动累计超额
df['excess_cum_20'] = df.groupby('stock_code')['excess_ret'].transform(
    lambda x: x.rolling(20, min_periods=15).sum()
)
# 前20日累计超额（lag 20日）
df['excess_cum_20_lag'] = df.groupby('stock_code')['excess_cum_20'].shift(20)

# 40日超额波动
df['excess_vol_40'] = df.groupby('stock_code')['excess_ret'].transform(
    lambda x: x.rolling(40, min_periods=30).std()
)

# 二阶导 = (近期动量 - 远期动量) / 波动
df['revision_raw'] = (df['excess_cum_20'] - df['excess_cum_20_lag']) / (
    df['excess_vol_40'].clip(lower=0.1) * np.sqrt(20)
)

# 截面缩尾
df['revision_w'] = df.groupby('date')['revision_raw'].transform(
    lambda x: winsorize(x, 0.02, 0.98)
)

# 市值中性化
factor2_list = []
for dt, gdf in df.groupby('date'):
    resid = neutralize_by_mktcap(gdf['revision_w'], gdf['log_mktcap'])
    tmp = gdf[['date', 'stock_code']].copy()
    tmp['revision_accel_v2'] = resid.values
    factor2_list.append(tmp)

factor2 = pd.concat(factor2_list, ignore_index=True)
factor2['revision_accel_v2'] = factor2.groupby('date')['revision_accel_v2'].transform(cross_section_zscore)

f2_path = os.path.join(DATA_DIR, 'factor_revision_accel_v2.csv')
factor2.to_csv(f2_path, index=False)
non_null = factor2['revision_accel_v2'].notna().sum()
print(f"  保存: {f2_path}, 非空因子值: {non_null}")

# ====================================================================
# 因子3: 振幅疲劳 v2
# 原始: 当前振幅 / 20日平均振幅 的反转信号
# 升级: 加入量价背离信号 + 市值中性化 + 反转方向确认
# ====================================================================
print("\n=== 因子3: 振幅疲劳 v2 ===")

# 振幅均值
df['amp_ma20'] = df.groupby('stock_code')['amplitude'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)
# 振幅比率（当前 vs 历史）
df['amp_ratio'] = df['amplitude'] / df['amp_ma20'].clip(lower=0.1)

# 5日平滑（减少噪声）
df['amp_ratio_5d'] = df.groupby('stock_code')['amp_ratio'].transform(
    lambda x: x.rolling(5, min_periods=3).mean()
)

# 量价背离信号: 振幅扩大但成交量萎缩
df['vol_ma20'] = df.groupby('stock_code')['volume'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)
df['vol_ratio'] = df['volume'] / df['vol_ma20'].clip(lower=1)

# 复合因子: -振幅比率 × (2 - 量比), 高振幅+低量=极度疲劳(负值)
df['fatigue_raw'] = -df['amp_ratio_5d'] * (2 - df['vol_ratio'].clip(0.2, 3))

# 截面缩尾
df['fatigue_w'] = df.groupby('date')['fatigue_raw'].transform(
    lambda x: winsorize(x, 0.02, 0.98)
)

# 市值中性化
factor3_list = []
for dt, gdf in df.groupby('date'):
    resid = neutralize_by_mktcap(gdf['fatigue_w'], gdf['log_mktcap'])
    tmp = gdf[['date', 'stock_code']].copy()
    tmp['amplitude_fatigue_v2'] = resid.values
    factor3_list.append(tmp)

factor3 = pd.concat(factor3_list, ignore_index=True)
factor3['amplitude_fatigue_v2'] = factor3.groupby('date')['amplitude_fatigue_v2'].transform(cross_section_zscore)

f3_path = os.path.join(DATA_DIR, 'factor_amplitude_fatigue_v2.csv')
factor3.to_csv(f3_path, index=False)
non_null = factor3['amplitude_fatigue_v2'].notna().sum()
print(f"  保存: {f3_path}, 非空因子值: {non_null}")

print("\n=== 全部因子构建完成 ===")
