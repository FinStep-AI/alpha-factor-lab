#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日内/隔夜波动率比因子 (Intraday-Overnight Volatility Ratio, IOVR)

论文基础：
  - Muravyev & Ni (2020) "Why Do Option Returns Change Sign from Day to Night?" JFE
  - Kelly & Clark (2011) "Returns in Trading vs Non-Trading Hours: The Difference
    Is Day and Night"
  - 核心：分解波动率为日内(open→close)和隔夜(close→open)两部分

思路：
  日内波动 = var(close/open - 1) 过去20天
  隔夜波动 = var(open/prev_close - 1) 过去20天
  IOVR = 日内波动 / 隔夜波动
  
  高IOVR = 价格主要在盘中变动 = 散户驱动(散户只能盘中交易)
  低IOVR = 价格主要隔夜变动 = 机构/知情交易者驱动(集合竞价+隔夜信息消化)
  
  假设：低IOVR(机构信息多)= 更高信息质量 = 更好的Quality
  反之，高IOVR(散户噪音)= 低质量 = 可能存在反转溢价
  
  实际方向需要回测验证。

与现有因子区别：
  - overnight_momentum: 隔夜收益的均值(方向性)
  - 本因子: 隔夜波动率的占比(不看方向，看波动结构)
  - idio_vol: 总波动率
  - 本因子: 波动率的成分分解
"""

import numpy as np
import pandas as pd
import sys

def calc_iovr(kline_path, output_path, window=20):
    print(f"📥 读取: {kline_path}")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # 日内收益: close/open - 1
    df['intraday_ret'] = df['close'] / df['open'].replace(0, np.nan) - 1
    
    # 隔夜收益: open/prev_close - 1
    df['prev_close'] = df.groupby('stock_code')['close'].shift(1)
    df['overnight_ret'] = df['open'] / df['prev_close'].replace(0, np.nan) - 1
    
    # 滚动方差
    df['intraday_var'] = df.groupby('stock_code')['intraday_ret'].transform(
        lambda x: x.rolling(window, min_periods=int(window * 0.6)).var()
    )
    df['overnight_var'] = df.groupby('stock_code')['overnight_ret'].transform(
        lambda x: x.rolling(window, min_periods=int(window * 0.6)).var()
    )
    
    # IOVR = 日内方差 / 隔夜方差
    df['iovr'] = df['intraday_var'] / df['overnight_var'].replace(0, np.nan)
    
    # Log transform for better distribution
    df['log_iovr'] = np.log(df['iovr'].clip(lower=1e-6))
    
    print(f"📊 IOVR统计: mean={df['iovr'].mean():.4f}, median={df['iovr'].median():.4f}")
    print(f"📊 log_IOVR: mean={df['log_iovr'].mean():.4f}, std={df['log_iovr'].std():.4f}")
    print(f"  日内方差均值: {df['intraday_var'].mean():.8f}")
    print(f"  隔夜方差均值: {df['overnight_var'].mean():.8f}")
    print(f"  比值(日内/隔夜): {df['intraday_var'].mean() / df['overnight_var'].mean():.2f}")
    
    factor_raw = df[['date', 'stock_code', 'log_iovr']].dropna().copy()
    factor_raw.columns = ['date', 'stock_code', 'factor_raw']
    
    # 成交额中性化
    print("⚙️  成交额中性化...")
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    amt_df = df[['date', 'stock_code', 'log_amount_20d']].drop_duplicates()
    factor_raw = factor_raw.merge(amt_df, on=['date', 'stock_code'], how='left')
    
    def neutralize(group):
        y = group['factor_raw'].values
        x = group['log_amount_20d'].values
        valid = ~(np.isnan(y) | np.isnan(x) | np.isinf(y) | np.isinf(x))
        if valid.sum() < 30:
            group['factor'] = np.nan
            return group
        X = np.column_stack([np.ones(valid.sum()), x[valid]])
        try:
            beta = np.linalg.lstsq(X, y[valid], rcond=None)[0]
            resid = np.full(len(y), np.nan)
            resid[valid] = y[valid] - X @ beta
            group['factor'] = resid
        except:
            group['factor'] = np.nan
        return group
    
    factor_raw = factor_raw.groupby('date', group_keys=False).apply(neutralize)
    
    # MAD Winsorize + Z-score
    print("⚙️  Winsorize + Z-score...")
    def winsorize_zscore(group):
        vals = group['factor'].values.copy()
        valid = ~(np.isnan(vals) | np.isinf(vals))
        if valid.sum() < 10:
            group['factor'] = np.nan
            return group
        med = np.nanmedian(vals[valid])
        mad = np.nanmedian(np.abs(vals[valid] - med))
        if mad > 1e-10:
            lower = med - 5 * 1.4826 * mad
            upper = med + 5 * 1.4826 * mad
            vals = np.where(valid, np.clip(vals, lower, upper), np.nan)
        mu = np.nanmean(vals[~np.isnan(vals)])
        std = np.nanstd(vals[~np.isnan(vals)])
        if std < 1e-10:
            group['factor'] = 0.0
        else:
            group['factor'] = np.where(np.isnan(vals), np.nan, (vals - mu) / std)
        return group
    
    factor_raw = factor_raw.groupby('date', group_keys=False).apply(winsorize_zscore)
    
    output = factor_raw[['date', 'stock_code', 'factor']].dropna()
    output = output.sort_values(['date', 'stock_code'])
    output.to_csv(output_path, index=False)
    
    print(f"\n✅ 保存: {output_path}")
    print(f"  行数: {len(output)}, 日期: {output['date'].nunique()}")
    print(f"  范围: {output['date'].min()} ~ {output['date'].max()}")
    
    return output

if __name__ == '__main__':
    kline = sys.argv[1] if len(sys.argv) > 1 else 'data/csi1000_kline_raw.csv'
    out = sys.argv[2] if len(sys.argv) > 2 else 'data/factor_iovr_v1.csv'
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    calc_iovr(kline, out, window)
