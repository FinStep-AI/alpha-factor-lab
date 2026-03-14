#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSI反转因子 (RSI Reversal)

基于经典 Wilder RSI (1978) 做横截面因子化

原理：
  RSI = 100 - 100/(1 + RS), RS = avg_gain / avg_loss over N days
  - RSI高(>70) → 超买，预期均值回归(看空)
  - RSI低(<30) → 超卖，预期均值回归(看多)
  
横截面使用：
  - 每天对所有股票的RSI做排名 → 做空高RSI，做多低RSI
  - 本质是一个短期反转因子，但用RSI的非线性变换比简单收益率反转更好
  - RSI的非线性放大了极端区域的信号，比线性反转更稳健

论文支持：
  - DeMark indicators / RSI in academic finance
  - Chong & Ng (2008) "Technical analysis and the London stock exchange"
  - A股实证：RSI作为选股因子在中小盘有效（多篇研报）

构造：
  1. 14日RSI (经典参数)
  2. 取负值(做反转)：factor = -RSI → 低RSI(超卖)得高因子值
  3. 成交额中性化 + MAD winsorize + z-score
"""

import numpy as np
import pandas as pd
import sys

def calc_rsi_factor(kline_path, output_path, period=14):
    print(f"📥 读取: {kline_path}")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # 日收益率
    if 'pct_change' in df.columns:
        df['ret'] = df['pct_change'] / 100.0
    else:
        df['ret'] = df.groupby('stock_code')['close'].pct_change()
    
    # 计算RSI
    df['gain'] = df['ret'].clip(lower=0)
    df['loss'] = (-df['ret']).clip(lower=0)
    
    # Wilder平滑 (指数移动平均)
    df['avg_gain'] = df.groupby('stock_code')['gain'].transform(
        lambda x: x.ewm(span=period, adjust=False).mean()
    )
    df['avg_loss'] = df.groupby('stock_code')['loss'].transform(
        lambda x: x.ewm(span=period, adjust=False).mean()
    )
    
    df['rs'] = df['avg_gain'] / df['avg_loss'].replace(0, np.nan)
    df['rsi'] = 100 - 100 / (1 + df['rs'])
    
    # 处理 avg_loss=0 的情况 (连续上涨)
    df.loc[df['avg_loss'] == 0, 'rsi'] = 100
    df.loc[df['avg_gain'] == 0, 'rsi'] = 0
    
    print(f"📊 RSI统计: mean={df['rsi'].mean():.2f}, std={df['rsi'].std():.2f}")
    print(f"  >70占比: {(df['rsi'] > 70).mean():.3f}")
    print(f"  <30占比: {(df['rsi'] < 30).mean():.3f}")
    
    # 取负值做反转 (低RSI = 好)
    df['factor_raw'] = -df['rsi']
    
    factor_raw = df[['date', 'stock_code', 'factor_raw']].dropna().copy()
    
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
    out = sys.argv[2] if len(sys.argv) > 2 else 'data/factor_rsi_reversal_v1.csv'
    period = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    
    calc_rsi_factor(kline, out, period)
