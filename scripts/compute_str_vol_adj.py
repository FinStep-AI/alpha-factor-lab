#!/usr/bin/env python3
"""
因子：短期反转+成交额加权 (Short-Term Reversal, Volume-Adjusted)
ID: str_vol_adj_v1

逻辑：
- 经典短期反转: 过去5天收益率取反
- 改进: 用成交额加权，放量跌的日子权重更大（因为放量跌=恐慌卖出=更可能反弹）
- factor = -sum(ret_i * amount_i, 5d) / sum(amount_i, 5d)
  即：成交额加权的5日反转收益
  
- 然后做成交额中性化，避免成交额本身的影响

为什么不用简单反转？
- 简单反转IC可能不够，加入成交额加权可以提升信噪比
- 放量下跌→恐慌卖出→超卖→反弹更强
- 放量上涨→FOMO追涨→超买→回调更强
"""

import pandas as pd
import numpy as np
import os

def compute_factor(kline_path, output_path):
    print("Loading kline data...")
    df = pd.read_csv(kline_path, dtype={'stock_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    df['daily_ret'] = df.groupby('stock_code')['close'].pct_change()
    
    window = 5
    
    g = df.groupby('stock_code')
    
    # 成交额加权收益
    df['ret_x_amt'] = df['daily_ret'] * df['amount']
    
    df['sum_ret_x_amt'] = g['ret_x_amt'].transform(
        lambda x: x.rolling(window, min_periods=4).sum()
    )
    df['sum_amt'] = g['amount'].transform(
        lambda x: x.rolling(window, min_periods=4).sum()
    )
    
    # 反转: 取反
    df['factor_raw'] = -(df['sum_ret_x_amt'] / df['sum_amt'])
    
    # log(20日平均成交额)
    df['mean_amt'] = g['amount'].transform(
        lambda x: x.rolling(20, min_periods=16).mean()
    )
    df['log_amount_20d'] = np.log(df['mean_amt'].clip(lower=1))
    
    factor_df = df[['date', 'stock_code', 'factor_raw', 'log_amount_20d']].dropna().copy()
    print(f"Raw factor: {len(factor_df)} rows, {factor_df['date'].nunique()} dates")
    
    # 截面中性化
    print("Neutralizing...")
    def neutralize_cs(group):
        y = group['factor_raw'].values.copy()
        x = group['log_amount_20d'].values.copy()
        valid = np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 30:
            return pd.Series(np.nan, index=group.index, name='factor')
        y_v, x_v = y[valid], x[valid]
        med = np.median(y_v)
        mad = np.median(np.abs(y_v - med)) * 1.4826
        if mad > 0:
            y_v = np.clip(y_v, med - 3*mad, med + 3*mad)
        x_mat = np.column_stack([np.ones(len(x_v)), x_v])
        beta = np.linalg.lstsq(x_mat, y_v, rcond=None)[0]
        resid = y_v - x_mat @ beta
        std = np.std(resid)
        if std > 0:
            resid = (resid - np.mean(resid)) / std
        out = np.full(len(y), np.nan)
        out[valid] = resid
        return pd.Series(out, index=group.index, name='factor')
    
    factor_df['factor'] = factor_df.groupby('date', group_keys=False).apply(
        lambda g: neutralize_cs(g)
    ).values
    
    output = factor_df[['date', 'stock_code', 'factor']].dropna().copy()
    output['stock_code'] = output['stock_code'].astype(str).str.zfill(6)
    output = output.sort_values(['date', 'stock_code'])
    output.to_csv(output_path, index=False)
    print(f"Factor saved: {output_path} ({len(output)} rows)")
    print(f"Date range: {output['date'].min()} ~ {output['date'].max()}")
    print(f"Stocks/date: {output.groupby('date')['stock_code'].count().mean():.0f}")

if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    compute_factor(
        os.path.join(base, 'data', 'csi1000_kline_raw.csv'),
        os.path.join(base, 'data', 'factor_str_vol_adj_v1.csv')
    )
