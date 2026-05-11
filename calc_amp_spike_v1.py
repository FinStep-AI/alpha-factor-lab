#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
振幅突破聚集度因子 (Amplitude Spike Concentration, amp_spike_v1)
========================================================

定义: 过去20日中，振幅高于20日均值的比率。
      衡量波动率聚集信号的持续性和突破频率。

公式:
  amt_mean_20d = MA20(amplitude)
  spike_indicator = (amplitude > amt_mean_20d).astype(float)
  spike_ratio = MA20(spike_indicator)   <- 即过去20日内,超均振幅日占比

最终因子 (中性化后): 因子原始值 = spike_ratio
                   中性化 : OLS回归残差 ~ log_amount_20d
                   标准化 : MAD winsorize → z-score

理论:
  波动率聚集效应(Volatility Clustering): 大幅波动后容易持续大幅波动。
  截面应用: 近期振幅持续突破均值的股票,信息事件密度高,价格发现过程
  更活跃,后续存在动量延续/信息扩散。

Args:
  --input      数据目录 (默认: .)
  --output     输出因子CSV (默认: data/factor_amp_spike_v1.csv)
  --min-days   最少交易日要求 (默认: 20)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

def compute_spike_ratio(amplitude_series: pd.Series, window: int = 20) -> pd.Series:
    """
    计算单只股票的振幅突破比率
    
    Parameters
    ----------
    amplitude_series : Series, 振幅序列 (按日期升序)
    window : 回看窗口 (默认20日)
    
    Returns
    -------
    spike_ratio : Series, 每日的spike_ratio值
    """
    amplitude = amplitude_series.values.astype(float)
    n = len(amplitude)
    spike_ratio = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        # 过去20日振幅的均值
        amp_win = amplitude[i - window + 1:i + 1]
        amt_mean = np.nanmean(amp_win)
        
        if np.isnan(amt_mean) or amt_mean == 0:
            spike_ratio[i] = np.nan
            continue
        
        # 过去WINDOW日内振幅超过均值的日数占比
        spike_ratio[i] = np.sum(amp_win > amt_mean) / window
    
    return pd.Series(spike_ratio, index=amplitude_series.index)

def load_kline_data(data_dir: Path) -> pd.DataFrame:
    """从csi1000_kline_raw.csv加载K线数据"""
    kline_path = data_dir / 'csi1000_kline_raw.csv'
    if not kline_path.exists():
        print(f"错误: 找不到 {kline_path}")
        sys.exit(1)
    
    df = pd.read_csv(kline_path, parse_dates=['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    print(f"已加载 {len(df)} 行, {df['stock_code'].nunique()} 只股票")
    return df

def compute_amount_20d(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """计算20日平均成交额"""
    df = df.copy()
    df['amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(window, min_periods=10).mean()
    )
    return df

def neutralize_factor(df: pd.DataFrame, factor_col: str, neutralize_col: str) -> pd.DataFrame:
    """
    成交额OLS中性化 + MAD去极值 + z-score
    
    Parameters:
    -----------
    df : 包含因子值和中性化变量的DataFrame
    factor_col : 因子列名
    neutralize_col : 中性化列名 (log_amount_20d)
    """
    from numpy.linalg import lstsq
    
    result = []
    
    for dt, group in df.groupby('date'):
        g = group[[factor_col, neutralize_col, 'stock_code']].dropna()
        if len(g) < 30:
            continue
        
        y = g[factor_col].values
        x = g[neutralize_col].values
        
        # OLS: y = alpha + beta * x + residual
        X = np.column_stack([np.ones(len(x)), x])
        try:
            coeffs, _, _, _ = lstsq(X, y, rcond=None)
            residuals = y - X @ coeffs
        except:
            continue
        
        # MAD winsorize
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med)) * 1.4826
        if mad < 1e-10:
            continue
        z = (residuals - med) / mad
        z = np.clip(z, -5, 5)  # 最终5σ截断
        
        # z-score
        mu, sigma = np.mean(z), np.std(z)
        if sigma < 1e-10:
            continue
        z = (z - mu) / sigma
        
        for idx, val in zip(g.index, z):
            result.append((dt, g.loc[idx, 'stock_code'], val))
    
    res_df = pd.DataFrame(result, columns=['date', 'stock_code', 'factor_value'])
    print(f"中性化后有效截面数: {res_df['date'].nunique()}")
    return res_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='.', help='数据目录')
    parser.add_argument('--output', default='data/factor_amp_spike_v1.csv', help='输出CSV')
    parser.add_argument('--window', type=int, default=20, help='回看窗口')
    parser.add_argument('--min-days', type=int, default=20, help='最少交易日')
    args = parser.parse_args()
    
    data_dir = Path(args.input)
    output = Path(args.output)
    print(f"=== 计算振幅突破聚集度因子 ===")
    
    # 加载K线数据
    df = load_kline_data(data_dir)
    
    # 计算20日平均成交额（用于中性化）
    df = compute_amount_20d(df)
    df['log_amount_20d'] = np.log1p(df['amount_20d'])
    
    # 计算振幅突破比率
    print(f"计算{args.window}日振幅突破比率...")
    df['amp_spike_raw'] = df.groupby('stock_code')['amplitude'].transform(
        lambda x: compute_spike_ratio(x, args.window).values
    )
    
    # 过滤有效数据
    df = df.dropna(subset=['amp_spike_raw', 'log_amount_20d'])
    print(f"有效数据: {len(df)} 行, {df['date'].nunique()} 个截面")
    
    # 中性化 + MAD + z-score
    print("OLS中性化 + MAD winsorize + z-score...")
    result = neutralize_factor(df, 'amp_spike_raw', 'log_amount_20d')
    
    # 保存
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"因子值已保存到: {output}")
    print(f"因子统计:\n{result['factor_value'].describe()}")
    return result

if __name__ == '__main__':
    main()
