#!/usr/bin/env python3
"""
因子: 隔夜收益变异系数 (Overnight Return CV)
category: 动量/波动率

逻辑:
  隔夜收益率变异性衡量收益分布的不确定性。
  Harvey & Siddique (2000) 条件性CAPM: 条件波动率是风险溢价来源。
  高CV = 隔夜波动剧烈 = 不确定性高 = 风险溢价 → 高预期收益
  
  与 existing 'overnight_vol_v1' 的区别:
  - overnight_vol_v1: std(overnight_ret, 20d) — 隔夜波动率
  - 本因子: CV = std / mean (变异系数) — 相对波动率/不确定性比例
    这比绝对波动率更能反映"不确定性的相对程度"

公式:
  overnight_ret = (open_t - close_{t-1}) / close_{t-1}
  cv_20d = std(overnight_ret, 20d) / abs(mean(overnight_ret, 20d) + eps)
  
  取对数 + 市值中性化

Barra风格: Momentum (信息的持续性代理)
方向: 待定（先试正向）
"""

import numpy as np
import pandas as pd
import sys

def compute_factor(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # 隔夜收益率: (open_t - close_{t-1}) / close_{t-1}
    grouped = df.groupby('stock_code')
    df['close_prev'] = grouped['close'].shift(1)
    df['overnight_ret'] = (df['open'] - df['close_prev']) / df['close_prev']
    
    # 20日均值和标准差
    df['on_ret_mean_20'] = grouped['overnight_ret'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    df['on_ret_std_20'] = grouped['overnight_ret'].transform(
        lambda x: x.rolling(20, min_periods=10).std()
    )
    
    # 变异系数: std / |mean|
    eps = 1e-8
    df['factor_raw'] = df['on_ret_std_20'] / (np.abs(df['on_ret_mean_20']) + eps)
    
    # 取对数 (处理偏态)
    df['factor_raw'] = np.log(1 + df['factor_raw'])
    
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
    print(f"[INFO] Overnight CV 因子: {result_df['date'].min()} ~ {result_df['date'].max()}, "
          f"{result_df['stock_code'].nunique()} stocks")
    print(f"[INFO] stats: mean={result_df['factor_neutral'].mean():.4f}, std={result_df['factor_neutral'].std():.4f}")
    
    out_path = 'data/factor_on_cv_v1.csv'
    result_df.to_csv(out_path, index=False)
    print(f"[INFO] 保存到 {out_path}")
    return out_path

if __name__ == '__main__':
    compute_factor(pd.read_csv('data/csi1000_kline_raw.csv'))
