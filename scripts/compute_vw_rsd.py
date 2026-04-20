#!/usr/bin/env python3
"""
因子: 成交量加权收益率离散度 (VW-RSD v1)
reference: Chordia & Subrahmanyam (2004) JFE - Order Imbalance

逻辑:
  高成交量日的收益率方向如果不一致 → 知情交易者意见分歧/信息效率低 → 低预期收益
  低离散度 → 知情交易者一致 → 信号清晰 → 高预期收益
  方向: 反向 (低离散度 = 高因子值 = 高预期收益)

公式 (raw):
  weighted_ret_i = ret_i * volume_i / sum(volume, 20)
  vw_rsd = std(weighted_ret, 20d)  # 成交量加权收益率的20日标准差

中性化:
  log(1 + vw_rsd) → OLS中性化 by log(amount_20d) → MAD winsorize → z-score

预期方向: 反向 (-1), 即: 低 vw_rsd = 高因子值 = 高预期收益
Barra风格: MICRO (微观结构/信息效率)
"""

import numpy as np
import pandas as pd
import sys
import subprocess
import json

def compute_factor(df):
    """计算 VW-RSD 因子"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # 日收益率
    grouped = df.groupby('stock_code')
    df['ret'] = grouped['close'].pct_change()
    
    # 20日滚动 sum(volume) 用于权重
    df['vol_sum_20'] = grouped['volume'].transform(lambda x: x.rolling(20, min_periods=10).sum())
    
    # 成交量加权收益率
    df['vw_ret'] = df['ret'] * df['volume'] / df['vol_sum_20']
    
    # 20日成交量加权收益率的标准差
    df['vw_rsd_raw'] = grouped['vw_ret'].transform(
        lambda x: x.rolling(20, min_periods=10).std()
    )
    
    # 取对数 (处理零值和分布)
    df['factor_raw'] = np.log(1 + df['vw_rsd_raw'])
    
    # 成交额20日均值 (中性化变量)
    df['log_amount_20d'] = np.log(df['amount'].rolling(20, min_periods=10).mean() + 1)
    
    # 截面中性化 (每天的 OLS 中性化 by log_amount_20d)
    results = []
    for date, group in df.groupby('date'):
        g = group.dropna(subset=['factor_raw', 'log_amount_20d'])
        if len(g) < 30:
            continue
        
        x = g['log_amount_20d'].values
        y = g['factor_raw'].values
        
        # OLS regression: y = a + b*x + residual
        X = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residual = y - X @ beta
        except:
            continue
        
        # MAD winsorize
        med = np.median(residual)
        mad = np.median(np.abs(residual - med)) * 1.4826
        if mad < 1e-10:
            continue
        upper, lower = med + 3 * mad, med - 3 * mad
        residual = np.clip(residual, lower, upper)
        
        # z-score
        mu, sigma = residual.mean(), residual.std()
        if sigma < 1e-10:
            continue
        z = (residual - mu) / sigma
        
        g = g.copy()
        g['factor_neutral'] = z
        results.append(g[['date', 'stock_code', 'factor_neutral']])
    
    if not results:
        print("ERROR: 截面中性化无数据", file=sys.stderr)
        sys.exit(1)
    
    result_df = pd.concat(results, ignore_index=True)
    print(f"[INFO] VW-RSD 因子计算完成: {result_df['date'].min()} ~ {result_df['date'].max()}, "
          f"{result_df['stock_code'].nunique()} 只股票")
    print(f"[INFO] 因子统计: mean={result_df['factor_neutral'].mean():.4f}, "
          f"std={result_df['factor_neutral'].std():.4f}")
    
    # 输出
    output_path = 'data/factor_vw_rsd_v1.csv'
    result_df.to_csv(output_path, index=False)
    print(f"[INFO] 因子值已保存到 {output_path}")
    
    return output_path

if __name__ == '__main__':
    data_path = 'data/csi1000_kline_raw.csv'
    print(f"[INFO] 加载数据: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[INFO] 数据量: {len(df)} 行, {df['stock_code'].nunique()} 只股票")
    compute_factor(df)
