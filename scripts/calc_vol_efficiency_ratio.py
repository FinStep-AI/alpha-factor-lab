#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
波动率效率比因子 (Volatility Efficiency Ratio, VER)

论文基础：
  - Garman & Klass (1980) "On the Estimation of Security Price Volatilities 
    from Historical Data" Journal of Business
  - Molnar (2012) "Properties of range-based volatility estimators"
  - Alizadeh, Brandt & Diebold (2002) "Range-based estimation of stochastic 
    volatility models" JF

核心思想：
  Garman-Klass波动率利用OHLC全部信息，比close-to-close波动率更准确。
  两者的比值(VER = CC_vol / GK_vol)揭示了价格变动的"效率"：
  
  - VER高(CC > GK): 收盘价变动大但日内波动小 → 跳空主导，信息冲击明确
  - VER低(CC < GK): 日内波动大但收盘价变动小 → 日内噪音大，信息不确定
  
  高VER股票 = 信息传递效率高 = Quality proxy
  低VER股票 = 日内噪音重 = 信息不确定性高

构造：
  1. CC_var = (close/prev_close - 1)^2 的20日均值
  2. GK_var = 0.5*(log(high/low))^2 - (2*ln2-1)*(log(close/open))^2 的20日均值
  3. VER = sqrt(CC_var) / sqrt(GK_var) — 越高表示close-to-close效率越好
  4. 成交额中性化 + MAD winsorize + z-score

与现有因子区别：
  - idio_vol: 只用close-to-close
  - amp_compress/expand: 只看振幅趋势  
  - range_vol: 只看range波动
  - VER是两种波动率的比值，全新维度
"""

import numpy as np
import pandas as pd
import sys

def calc_vol_efficiency_ratio(kline_path, output_path, window=20):
    print(f"📥 读取: {kline_path}")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # Close-to-close return squared
    df['prev_close'] = df.groupby('stock_code')['close'].shift(1)
    df['cc_ret'] = df['close'] / df['prev_close'] - 1
    df['cc_var_daily'] = df['cc_ret'] ** 2
    
    # Garman-Klass variance components
    # GK = 0.5 * (log(H/L))^2 - (2*ln2 - 1) * (log(C/O))^2
    log_hl = np.log(df['high'] / df['low'].replace(0, np.nan))
    log_co = np.log(df['close'] / df['open'].replace(0, np.nan))
    df['gk_var_daily'] = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    
    # Clamp GK to be positive (numerical edge case when high==low)
    df['gk_var_daily'] = df['gk_var_daily'].clip(lower=1e-12)
    
    # Rolling means
    df['cc_var_20'] = df.groupby('stock_code')['cc_var_daily'].transform(
        lambda x: x.rolling(window, min_periods=int(window * 0.6)).mean()
    )
    df['gk_var_20'] = df.groupby('stock_code')['gk_var_daily'].transform(
        lambda x: x.rolling(window, min_periods=int(window * 0.6)).mean()
    )
    
    # VER = sqrt(CC) / sqrt(GK)
    df['ver'] = np.sqrt(df['cc_var_20'].clip(lower=1e-12)) / np.sqrt(df['gk_var_20'].clip(lower=1e-12))
    
    print(f"📊 VER统计: mean={df['ver'].mean():.4f}, std={df['ver'].std():.4f}, median={df['ver'].median():.4f}")
    print(f"  VER>1 占比: {(df['ver'] > 1).mean():.3f}")
    print(f"  VER<0.5 占比: {(df['ver'] < 0.5).mean():.3f}")
    
    factor_raw = df[['date', 'stock_code', 'ver']].dropna().copy()
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
        valid = ~(np.isnan(y) | np.isnan(x))
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
        valid = ~np.isnan(vals)
        if valid.sum() < 10:
            group['factor'] = np.nan
            return group
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals[valid] - med))
        if mad > 1e-10:
            lower = med - 5 * 1.4826 * mad
            upper = med + 5 * 1.4826 * mad
            vals = np.clip(vals, lower, upper)
        mu = np.nanmean(vals)
        std = np.nanstd(vals)
        if std < 1e-10:
            group['factor'] = 0.0
        else:
            group['factor'] = (vals - mu) / std
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
    out = sys.argv[2] if len(sys.argv) > 2 else 'data/factor_vol_eff_ratio_v1.csv'
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    calc_vol_efficiency_ratio(kline, out, window)
