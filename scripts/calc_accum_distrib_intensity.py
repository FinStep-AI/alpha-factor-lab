#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化累积-分配强度因子 (Accumulation-Distribution Intensity, ADI)

论文基础：
  - 结合 Williams %R / Chaikin Oscillator / Marc Chaikin's AD Line 思想
  - 创新：用相对成交量加权的日内位置累积，衡量"聪明资金"的方向
  - 近期论文支持：Frazzini (2006) "The Disposition Effect and Underreaction 
    to News" JF, Lo & Wang (2000) "Trading Volume" NBER

核心思想：
  如果股票在放量日（高关注度）收在日内高位 → 买方主导 → 看多
  如果股票在放量日收在日内低位 → 卖方主导 → 看空
  20日累积的量价位置，捕捉持续性的供需失衡

构造：
  1. CLV (Close Location Value) = 2 * (close - low) / (high - low) - 1  ∈ [-1, 1]
     - close==high → CLV=1 (强势收盘)
     - close==low → CLV=-1 (弱势收盘)
  2. rel_vol = volume / MA20(volume) — 相对成交量
  3. ADI_daily = CLV * rel_vol — 量加权的日内位置
  4. ADI_20d = sum(ADI_daily, 20) — 20日累积

中性化：对 log(20日均成交额) OLS 回归取残差
"""

import numpy as np
import pandas as pd
import sys

def calc_adi(kline_path, output_path, window=20):
    print(f"📥 读取: {kline_path}")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # CLV: Close Location Value
    hl_range = df['high'] - df['low']
    df['clv'] = np.where(hl_range > 1e-6, 
                          2 * (df['close'] - df['low']) / hl_range - 1, 
                          0)
    
    # Relative Volume
    df['vol_ma20'] = df.groupby('stock_code')['volume'].transform(
        lambda x: x.rolling(window, min_periods=int(window * 0.6)).mean()
    )
    df['rel_vol'] = df['volume'] / df['vol_ma20'].replace(0, np.nan)
    
    # Daily ADI
    df['adi_daily'] = df['clv'] * df['rel_vol']
    
    # Rolling sum
    df['adi_20d'] = df.groupby('stock_code')['adi_daily'].transform(
        lambda x: x.rolling(window, min_periods=int(window * 0.6)).sum()
    )
    
    print(f"📊 CLV统计: mean={df['clv'].mean():.4f}, std={df['clv'].std():.4f}")
    print(f"📊 rel_vol统计: mean={df['rel_vol'].mean():.4f}, std={df['rel_vol'].std():.4f}")
    print(f"📊 ADI_20d: mean={df['adi_20d'].mean():.4f}, std={df['adi_20d'].std():.4f}")
    
    factor_raw = df[['date', 'stock_code', 'adi_20d']].dropna().copy()
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
    out = sys.argv[2] if len(sys.argv) > 2 else 'data/factor_adi_v1.csv'
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    calc_adi(kline, out, window)
