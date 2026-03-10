#!/usr/bin/env python3
"""
因子：下跌日占比反转 (Down-Day Ratio Reversal)
ID: down_day_ratio_v1

逻辑：
- 过去20天中收益为负的天数占比
- 高占比 = 连续下跌 = 均值回复反弹
- 类似于shadow_pressure(卖压耗尽后反弹)的逻辑，但更直接
- 与CVaR不同：CVaR看极端跌幅（尾部），本因子看下跌频率（持续性）
- 反向使用：高下跌占比→做多

中性化：成交额 OLS残差
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
    
    window = 20
    print("Computing down-day ratio...")
    
    g = df.groupby('stock_code')
    
    # 下跌标志 (ret < 0)
    df['is_down'] = (df['daily_ret'] < 0).astype(float)
    # NaN不算
    df['is_down'] = np.where(df['daily_ret'].isna(), np.nan, df['is_down'])
    
    # 20日下跌天数占比
    df['down_ratio'] = g['is_down'].transform(
        lambda x: x.rolling(window, min_periods=16).mean()
    )
    
    # 反向使用：高下跌占比 → 高因子值
    df['factor_raw'] = df['down_ratio']
    
    # log(20日平均成交额)
    df['mean_amt'] = g['amount'].transform(
        lambda x: x.rolling(window, min_periods=16).mean()
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
        os.path.join(base, 'data', 'factor_down_day_ratio_v1.csv')
    )
