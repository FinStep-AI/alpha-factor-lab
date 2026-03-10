#!/usr/bin/env python3
"""
因子：缺口回补率 (Gap Fill Ratio)
ID: gap_fill_ratio_v1

逻辑：
- 每天计算gap = open/prev_close - 1
- 每天计算intraday_move = close/open - 1
- 当gap > 0且intraday_move < 0: 高开低走（卖方力量 / 分销）
- 当gap < 0且intraday_move > 0: 低开高走（买方力量 / 吸筹）
- gap_fill_ratio = 20天中"反向回补"的频率
  即: 高开低走天数 + 低开高走天数 的占比
- 更精细: 加权版本 = sum(sign(gap) != sign(intraday)) * |intraday| / |gap|
  这衡量了"多少gap被日内交易回补"
  
- 高回补率 → 市场在纠正过度反应 → 均值回复型股票 → 可能后续更稳定
- 低回补率 → gap方向和日内一致 → 趋势延续型 → 可能继续

本因子核心创新: 利用open-close vs gap的不一致性刻画反转压力
"""

import pandas as pd
import numpy as np
import os

def compute_factor(kline_path, output_path):
    print("Loading kline data...")
    df = pd.read_csv(kline_path, dtype={'stock_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    window = 20
    
    g = df.groupby('stock_code')
    
    # 跳空幅度
    df['prev_close'] = g['close'].shift(1)
    df['gap'] = df['open'] / df['prev_close'] - 1
    
    # 日内涨跌
    df['intraday'] = df['close'] / df['open'] - 1
    
    # 反向标志: gap和intraday方向相反
    # sign(gap) * sign(intraday) < 0 = 反向
    df['gap_reversed'] = ((df['gap'] * df['intraday']) < 0).astype(float)
    
    # 加权回补度: 当反向时，|intraday/gap| 衡量回补幅度 (cap at 2.0)
    df['fill_intensity'] = np.where(
        (df['gap'] * df['intraday']) < 0,
        np.clip(np.abs(df['intraday'] / df['gap'].replace(0, np.nan)), 0, 2.0),
        0
    )
    
    # NaN处理
    df['gap_reversed'] = np.where(df['gap'].isna() | df['intraday'].isna(), np.nan, df['gap_reversed'])
    df['fill_intensity'] = np.where(df['gap'].isna() | df['intraday'].isna(), np.nan, df['fill_intensity'])
    
    # 20日滚动均值
    df['factor_raw'] = g['fill_intensity'].transform(
        lambda x: x.rolling(window, min_periods=16).mean()
    )
    
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
        os.path.join(base, 'data', 'factor_gap_fill_ratio_v1.csv')
    )
