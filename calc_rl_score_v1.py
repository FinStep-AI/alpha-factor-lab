#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Close-to-Low区间位置因子 (Range Location Score, rl_score_v1)
========================================================

构造:
  过去20日内每日: (high - close) / (high - low + eps)
  取均值即为rl_raw → 高 = 收盘接近日内低点 = 日内卖压信号。
  本因子取反拉伸: rl_score = -rl_raw → 高 = 收盘在日内高位 = 强势。
  做成交额OLS中性化 + MAD winsorize + z-score。

理论支持:
  日内价格位置(close在high-low中的相对位置)是日内方向性的简洁代理。
  在过去20日内持续收高位(接近high)反映信息持续推升信号，
  中证1000小盘股日内方向信号的截面持续性已被close_low_v1(反转版)验证成功。
  本因子是close_low_v1(低点收盘反转)的"正向(强势收盘)"版本，在学术文献中属直接应用。

参数:
  window = 20日
  中性化: log(amount_20d) OLS
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import lstsq
import sys

def compute_rl_score(df: pd.DataFrame, window: int = 20, eps: float = 1e-8) -> pd.DataFrame:
    """
    计算 RL 分数 = -MA20((high - close) / (high - low))
    
    高 rl_raw = 接近日内低点（收盘价接近低点）
    高 rl_score（取反后）= 接近日内高点（收盘在高位）
    """
    df = df.copy()
    df['rl_raw'] = (df['high'] - df['close']) / (df['high'] - df['low'] + eps)
    # -MA20(rl_raw) 得到 rl_score
    df['rl_score_raw'] = df.groupby('stock_code')['rl_raw'].transform(
        lambda x: -x.rolling(window, min_periods=int(window*0.6)).mean()
    )
    return df

def neutralize_ols(df: pd.DataFrame, factor_col: str, neutralizer_col: str) -> pd.Series:
    """
    成交额OLS截量中性化 + MAD winsorize + z-score
    
    Parameters
    ----------
    df : DataFrame with date, stock_code, factor_col, neutralizer_col
    factor_col : 因子原始值列名
    neutralizer_col : 中性化变量列名
    
    Returns
    -------
    factor_final : Series indexed by original DataFrame
    """
    result_idx, result_val = [], []
    
    for dt, group in df.groupby('date', sort=False):
        g = group[[factor_col, neutralizer_col, 'stock_code']].dropna()
        if len(g) < 30:
            continue
        
        y = g[factor_col].values.astype(float)
        x = g[neutralizer_col].values.astype(float)
        
        X = np.column_stack([np.ones(len(x)), x])
        try:
            coeffs, _, _, _ = lstsq(X, y, rcond=None)
            residual = y - X @ coeffs
        except:
            continue
        
        # MAD winsorize 3σ
        med = np.median(residual)
        mad = np.median(np.abs(residual - med)) * 1.4826
        if mad < 1e-10:
            continue
        z = (residual - med) / mad
        z = np.clip(z, -5, 5)
        
        # Final z-score
        mu, sigma = np.mean(z), np.std(z)
        if sigma < 1e-10:
            continue
        z = (z - mu) / sigma
        
        for idx, val in zip(g.index, z):
            result_idx.append(idx)
            result_val.append(val)
    
    factor_final = pd.Series(result_val, index=result_idx, dtype=float)
    return factor_final

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data', help='数据目录')
    parser.add_argument('--output', default='data/factor_rl_score_v1.csv', help='输出因子CSV')
    parser.add_argument('--window', type=int, default=20, help='回看窗口')
    args = parser.parse_args()
    output_path = Path(args.output)
    
    print(f"=== 计算 Close-to-High 区间位置因子 ===")
    
    # 加载K线数据
    kline_path = Path(args.input) / 'csi1000_kline_raw.csv'
    df = pd.read_csv(kline_path, parse_dates=['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    print(f"加载 {len(df)} 行, {df['stock_code'].nunique()} 只股票")
    
    # 计算20日平均成交额（中性化变量）
    df['amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    df['log_amount_20d'] = np.log1p(df['amount_20d'])
    
    # 计算因子
    df = compute_rl_score(df, window=args.window)
    
    # 中性化
    print("OLS中性化 + MAD winsorize + z-score...")
    df['factor_value'] = neutralize_ols(df, 'rl_score_raw', 'log_amount_20d')
    
    # 过滤有效截面
    result = df[['date', 'stock_code', 'factor_value']].dropna(subset=['factor_value'])
    print(f"有效截面数: {result['date'].nunique()} 日, {result['stock_code'].nunique()} 支股票")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"因子值已保存: {output_path}")
    print("因子统计:")
    print(result['factor_value'].describe())

if __name__ == '__main__':
    main()
