#!/usr/bin/env python3
"""
因子: 高成交量日远期收益增强 (High-Volume Ticket Return, HVTR v1)

论文参考:
  Gervais, Kaniel & Mingelgrin (2001) JF: "The High-Volume Return Premium"
  发现高成交量日(>正常均值+1SD)后的一周+N月表现均有正超额(美股1963-1996)

日频近似构造:
  - spike日: volume > MA20(volume) + 1*std(volume, 20d)
  - 因子 = 过去20日中(高成交量日远期收益均值 - 正常日远期收益均值)
  - 高值 = 高成交量日后有正超额 → 后续做多正alpha

方向: 正向
Barra风格: Sentiment/MICRO (行为金融-成交量信号)
"""

import numpy as np
import pandas as pd
import sys

def compute_factor(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    grouped = df.groupby('stock_code')
    df['ret'] = grouped['close'].pct_change()
    # 5日远期收益
    df['fwd_ret_5d'] = grouped['close'].transform(lambda x: x.shift(-5) / x - 1)
    # 20日远期收益
    df['fwd_ret_20d'] = grouped['close'].transform(lambda x: x.shift(-20) / x - 1)
    
    # volume stats
    df['vol_ma_20'] = grouped['volume'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df['vol_std_20'] = grouped['volume'].transform(
        lambda x: x.rolling(20, min_periods=10).std().fillna(0)
    )
    
    # spike threshold: MA20 + k*std
    k_threshold = 1.0
    df['spike_threshold'] = df['vol_ma_20'] + k_threshold * df['vol_std_20']
    df['is_spike'] = (df['volume'] > df['spike_threshold']).astype(int)
    
    # 20日窗口内: spike日和non-spike日的远期收益均值
    def spike_returns(group):
        spike_fwd = group['fwd_ret_20d'].where(group['is_spike'] == 1)
        normal_fwd = group['fwd_ret_20d'].where(group['is_spike'] == 0)
        spike_mean = spike_fwd.rolling(20, min_periods=8).mean()
        normal_mean = normal_fwd.rolling(20, min_periods=8).mean()
        diff = spike_mean - normal_mean
        return diff
    
    print("[INFO] 计算高成交量日远期收益差异...")
    df['factor_raw'] = grouped.apply(spike_returns).reset_index(level=0, drop=True)
    
    # 市值中性化
    df['log_amount_20d'] = np.log(df['amount'].rolling(20, min_periods=10).mean() + 1)
    
    results = []
    for date, group in df.groupby('date'):
        g = group.dropna(subset=['factor_raw', 'log_amount_20d'])
        if len(g) < 30:
            continue
        
        x = g['log_amount_20d'].values
        y = g['factor_raw'].values
        
        X = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residual = y - X @ beta
        except:
            continue
        
        med = np.median(residual)
        mad = np.median(np.abs(residual - med)) * 1.4826
        if mad < 1e-10:
            continue
        residual = np.clip(residual, med - 3*mad, med + 3*mad)
        mu, sigma = residual.mean(), residual.std()
        if sigma < 1e-10:
            continue
        z = (residual - mu) / sigma
        
        g = g.copy()
        g['factor_neutral'] = z
        results.append(g[['date', 'stock_code', 'factor_neutral']])
    
    result_df = pd.concat(results, ignore_index=True)
    print(f"[INFO] HVTR 因子: {result_df['date'].min()} ~ {result_df['date'].max()}, "
          f"{result_df['stock_code'].nunique()} stocks")
    print(f"[INFO] stats: mean={result_df['factor_neutral'].mean():.4f}, std={result_df['factor_neutral'].std():.4f}")
    
    out_path = 'data/factor_hvtr_v1.csv'
    result_df.to_csv(out_path, index=False)
    print(f"[INFO] 保存 {out_path}")
    return out_path

if __name__ == '__main__':
    from datetime import datetime
    compute_factor(pd.read_csv('data/csi1000_kline_raw.csv'))
