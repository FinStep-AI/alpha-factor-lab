#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实体强度因子 (Candle Body Strength, cbi_v1 - Candle Body Intensity V2)
==========================================================

构造 (V2修复):
  每日实体强度: (close - open) / (high - low + eps)
  高实体比 = 价格从开盘到收盘的明确方向移动
  20日纯方向持续性: rl_score = MA20((high-close)/(high-low)) 的反方向
  取 正 方向: 高因子值 = 收盘接近高位 (短线强势)

参数:
  window = 20日
  中性化: log(amount_20d) OLS + MAD winsorize + z-score
"""

import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import lstsq

def compute_factor(df: pd.DataFrame, window: int = 20, eps: float = 1e-8) -> pd.DataFrame:
    """
    因子定义:
      rl_raw = (high - close) / (high - low)
      rl_score = -MA20(rl_raw) = close.close位置信号 (高 = 强势)
    """
    df = df.copy()
    df['rl_raw'] = (df['high'] - df['close']) / (df['high'] - df['low'] + eps)
    
    # raw = 高 = 收盘接近低
    # factor = -MA20(rl_raw) → high = 收盘接近高
    df['rl_score_raw'] = df.groupby('stock_code')['rl_raw'].transform(
        lambda x: -x.rolling(window, min_periods=int(window*0.7)).mean()
    )
    
    # 20日成交额（中性化变量）
    df['amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    df['log_amount'] = np.log1p(df['amount_20d'])
    
    return df

def neutralize_factor(df: pd.DataFrame, factor_col: str, neutralizer_col: str) -> pd.Series:
    """成交额OLS中性化 + MAD winsorize + z-score"""
    result_idx, result_val = [], []
    
    for dt, g in df.groupby('date', sort=False):
        g2 = g[[factor_col, neutralizer_col, 'stock_code']].dropna()
        if len(g2) < 30:
            continue
        
        y = g2[factor_col].values.astype(float)
        x = g2[neutralizer_col].values.astype(float)
        X = np.column_stack([np.ones(len(x)), x])
        try:
            coeffs, _, _, _ = lstsq(X, y, rcond=None)
            resid = y - X @ coeffs
        except:
            continue
        
        med = np.median(resid)
        mad = np.median(np.abs(resid - med)) * 1.4826
        if mad < 1e-10:
            continue
        z = (resid - med) / mad
        z = np.clip(z, -5, 5)
        mu, sigma = np.mean(z), np.std(z)
        if sigma < 1e-10:
            continue
        z = (z - mu) / sigma
        
        for idx, val in zip(g2.index, z):
            result_idx.append(idx)
            result_val.append(val)
    
    return pd.Series(result_val, index=result_idx)

def main():
    output = Path('data/factor_cbi_v2.csv')
    df = pd.read_csv('data/csi1000_kline_raw.csv', parse_dates=['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    print(f"加载 {len(df)} 行, {df['stock_code'].nunique()} 只股票")
    df = compute_factor(df)
    
    print("中性化...")
    df['factor_value'] = neutralize_factor(df, 'rl_score_raw', 'log_amount')
    
    result = df[['date', 'stock_code', 'factor_value']].dropna(subset=['factor_value'])
    print(f"有效截面: {result['date'].nunique()} 日")
    
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"已保存: {output}")
    print(result['factor_value'].describe())
    return result

if __name__ == '__main__':
    main()
