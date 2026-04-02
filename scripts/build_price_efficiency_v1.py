#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Price Efficiency Ratio (PER) Factor v1

概念：衡量价格路径效率，灵感来自 Fractal Efficiency / Kaufman Efficiency Ratio
公式：PER(N) = |Close(t) - Close(t-N)| / Σ|Close(i) - Close(i-1)|  for i in [t-N+1, t]
     值域 [0, 1]：1=完美趋势，0=纯噪音
     
选股逻辑（反向）：低效率 = 高噪音 = 短期超跌/盘整后均值回复
  -> 方向: -1 (低PER → 高收益)

输出：因子值CSV (date, code, factor_value)
"""

import argparse
import numpy as np
import pandas as pd
import sys
from pathlib import Path


def compute_price_efficiency(df_kline: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """计算价格效率比因子"""
    
    df = df_kline.copy()
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    
    results = []
    
    for code, grp in df.groupby('code'):
        grp = grp.sort_values('date').reset_index(drop=True)
        close = grp['close'].values
        dates = grp['date'].values
        
        if len(close) < window + 1:
            continue
            
        # 逐日差分绝对值
        abs_diff = np.abs(np.diff(close))
        
        factor_vals = np.full(len(close), np.nan)
        
        for i in range(window, len(close)):
            net_move = abs(close[i] - close[i - window])
            path_length = np.sum(abs_diff[i - window:i])
            
            if path_length > 1e-10:
                factor_vals[i] = net_move / path_length
            else:
                factor_vals[i] = np.nan
        
        for i in range(len(close)):
            if not np.isnan(factor_vals[i]):
                results.append({
                    'date': dates[i],
                    'code': code,
                    'factor_value': factor_vals[i]
                })
    
    return pd.DataFrame(results)


def neutralize_by_market_cap(factor_df: pd.DataFrame, kline_df: pd.DataFrame) -> pd.DataFrame:
    """市值中性化：用log(amount)作为市值代理，横截面回归取残差"""
    
    # 用成交额作为市值代理
    amt = kline_df[['date', 'code', 'amount']].copy()
    amt['log_amount'] = np.log(amt['amount'].clip(lower=1))
    
    merged = factor_df.merge(amt[['date', 'code', 'log_amount']], on=['date', 'code'], how='left')
    
    results = []
    for date, grp in merged.groupby('date'):
        valid = grp.dropna(subset=['factor_value', 'log_amount'])
        if len(valid) < 30:
            continue
        
        x = valid['log_amount'].values
        y = valid['factor_value'].values
        
        # OLS 回归取残差
        x_dm = x - x.mean()
        beta = np.dot(x_dm, y) / (np.dot(x_dm, x_dm) + 1e-10)
        alpha = y.mean() - beta * x.mean()
        residual = y - (alpha + beta * x)
        
        # z-score 标准化
        std = residual.std()
        if std > 1e-10:
            residual = (residual - residual.mean()) / std
        
        for idx, r in zip(valid.index, residual):
            results.append({
                'date': valid.loc[idx, 'date'],
                'code': valid.loc[idx, 'code'],
                'factor_value': r
            })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Price Efficiency Ratio Factor')
    parser.add_argument('--data', default='data/csi1000_kline_raw.csv', help='K线数据')
    parser.add_argument('--window', type=int, default=20, help='回看窗口')
    parser.add_argument('--neutralize', action='store_true', default=True, help='市值中性化')
    parser.add_argument('--output', default='data/factor_price_efficiency_v1.csv', help='输出路径')
    args = parser.parse_args()
    
    print(f"读取K线数据: {args.data}")
    df = pd.read_csv(args.data)
    # 统一字段名
    if 'stock_code' in df.columns and 'code' not in df.columns:
        df.rename(columns={'stock_code': 'code'}, inplace=True)
    print(f"  共 {df['code'].nunique()} 只股票, {len(df)} 行")
    
    print(f"计算Price Efficiency Ratio (window={args.window})...")
    factor_df = compute_price_efficiency(df, window=args.window)
    print(f"  原始因子: {len(factor_df)} 行, {factor_df['date'].nunique()} 个截面")
    
    if args.neutralize:
        print("市值中性化 (log_amount 回归取残差)...")
        factor_df = neutralize_by_market_cap(factor_df, df)
        print(f"  中性化后: {len(factor_df)} 行")
    
    factor_df.to_csv(args.output, index=False)
    print(f"输出: {args.output}")
    
    # 基本统计
    print(f"\n因子统计:")
    print(f"  均值: {factor_df['factor_value'].mean():.4f}")
    print(f"  标准差: {factor_df['factor_value'].std():.4f}")
    print(f"  日期范围: {factor_df['date'].min()} ~ {factor_df['date'].max()}")
    print(f"  每日股票数: {factor_df.groupby('date')['code'].count().mean():.0f}")


if __name__ == '__main__':
    main()
