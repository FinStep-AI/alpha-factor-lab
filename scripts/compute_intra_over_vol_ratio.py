#!/usr/bin/env python3
"""
因子：下跌持续性强度 (Drawdown Persistence Intensity)
ID: dd_persist_v1

逻辑：
  过去20天，计算"连续下跌段"的特征：
  1. 找到所有连续下跌段（consecutive negative return days）
  2. 对每个连续下跌段，计算 length × cumulative_loss
  3. 因子 = sum(all segments) / 20
  
  高值 = 近期有长时间的持续下跌 = 基本面恶化/机构出货
  与CVaR不同: CVaR看极端单日跌幅, 本因子看下跌的持续性（连跌）
  
  连跌3天累跌-5%比单日跌5%然后反弹有更不同的含义：
  - 连跌 = 持续卖压 = 可能有坏消息在释放
  - 单日跌 = 可能是流动性冲击 = 更可能反弹（CVaR已捕捉）
  
  反向使用: 做多连跌强度低(没有连跌段)的股票

实际上这太复杂了。让我换一个更简洁的角度。

改为：最大连跌天数 (Max Consecutive Down Days)
  = 过去20天中最长的连续下跌天数
  
  做多: 低连跌天数（稳定）
  做空: 高连跌天数（有持续卖压）

实际上：连跌天数跟CVaR高度相关，换一个。

最终选择：收益波动率的时变性 (Volatility of Volatility / Vol Clustering)
  = 过去40天中，前20天波动率 vs 后20天波动率的变化率
  = vol_recent / vol_old - 1
  
  vol上升 → 风险增大 → 做空
  vol下降 → 风险减小 → 做多

这比idio_vol/vol_term_structure(已失败)更直接
"""

# 最终决定：收益率波动率变化 (Volatility Change)
# 但这已经被vol_term_structure_v1(失败)覆盖了。

# 让我做一个真正不同的因子：
# 日内波动率 vs 隔夜波动率 的比值
# intraday_vol = std(close/open - 1, 20d)
# overnight_vol = std(open/prev_close - 1, 20d)  
# ratio = intraday_vol / overnight_vol
# 高比值 = 日内波动大于隔夜 = 日内噪声交易多 = 散户主导
# 低比值 = 隔夜波动大于日内 = 信息驱动 = 机构主导
# 做多低比值(机构主导)? 或做多高比值(散户主导→被低估)?

import pandas as pd
import numpy as np
import os

def compute_factor(kline_path, output_path):
    print("Loading kline data...")
    df = pd.read_csv(kline_path, dtype={'stock_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    g = df.groupby('stock_code')
    
    # 日内收益
    df['intraday_ret'] = df['close'] / df['open'] - 1
    
    # 隔夜收益
    df['prev_close'] = g['close'].shift(1)
    df['overnight_ret'] = df['open'] / df['prev_close'] - 1
    
    window = 20
    
    # 日内波动率
    df['intraday_vol'] = g['intraday_ret'].transform(
        lambda x: x.rolling(window, min_periods=16).std()
    )
    
    # 隔夜波动率
    df['overnight_vol'] = g['overnight_ret'].transform(
        lambda x: x.rolling(window, min_periods=16).std()
    )
    
    # 比值 (日内/隔夜)
    # 取log使之对称
    df['factor_raw'] = np.log(df['intraday_vol'] / df['overnight_vol'].clip(lower=1e-6))
    
    # log(20日平均成交额)
    df['mean_amt'] = g['amount'].transform(
        lambda x: x.rolling(window, min_periods=16).mean()
    )
    df['log_amount_20d'] = np.log(df['mean_amt'].clip(lower=1))
    
    factor_df = df[['date', 'stock_code', 'factor_raw', 'log_amount_20d']].dropna().copy()
    # 过滤异常值
    factor_df = factor_df[np.isfinite(factor_df['factor_raw'])].copy()
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
        os.path.join(base, 'data', 'factor_intra_over_vol_ratio_v1.csv')
    )
