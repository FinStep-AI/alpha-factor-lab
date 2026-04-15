#!/usr/bin/env python3
"""
因子：换手率峰度 (Turnover Kurtosis) - 快速版
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

WINDOW = 20

def rolling_kurtosis_fast(arr, window=20, min_periods=15):
    """用向量化方式计算滚动峰度"""
    n = len(arr)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        w = arr[i - window + 1: i + 1]
        valid = w[~np.isnan(w)]
        if len(valid) < min_periods:
            continue
        mu = np.mean(valid)
        std = np.std(valid, ddof=1)
        if std < 1e-10:
            result[i] = 0.0
            continue
        m4 = np.mean((valid - mu) ** 4)
        kurt = m4 / (std ** 4) - 3.0  # excess kurtosis
        result[i] = kurt
    
    return result


def main():
    base = Path(__file__).resolve().parent.parent
    data_dir = base / 'data'
    
    print("读取K线数据...")
    df = pd.read_csv(data_dir / 'csi1000_kline_raw.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['pct_change'])
    
    print(f"股票数: {df['stock_code'].nunique()}, 行数: {len(df)}")
    
    # 计算换手率的滚动峰度
    print(f"计算{WINDOW}日换手率峰度...")
    stocks = df['stock_code'].unique()
    factor_vals = np.full(len(df), np.nan)
    
    for i, code in enumerate(stocks):
        if (i + 1) % 200 == 0:
            print(f"  进度: {i+1}/{len(stocks)}")
        idx = df[df['stock_code'] == code].index
        turnover = df.loc[idx, 'turnover'].values
        kurt = rolling_kurtosis_fast(turnover, WINDOW, 15)
        factor_vals[idx] = kurt
    
    df['factor'] = factor_vals
    
    # 20日平均成交额(中性化用)
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    
    # OLS中性化
    print("成交额OLS中性化...")
    from numpy.linalg import lstsq
    result = df['factor'].copy()
    for date in df['date'].unique():
        mask = df['date'] == date
        sub = df.loc[mask, ['factor', 'log_amount_20d']].dropna()
        if len(sub) < 30:
            continue
        y = sub['factor'].values
        X = np.column_stack([np.ones(len(y)), sub['log_amount_20d'].values])
        try:
            beta, _, _, _ = lstsq(X, y, rcond=None)
            residual = y - X @ beta
            result.loc[sub.index] = residual
        except:
            pass
    df['factor'] = result
    
    # MAD winsorize + z-score per day
    print("MAD winsorize + z-score...")
    def mad_wz(s):
        median = s.median()
        mad = (s - median).abs().median()
        if mad < 1e-10:
            return s * 0
        upper = median + 5 * 1.4826 * mad
        lower = median - 5 * 1.4826 * mad
        s = s.clip(lower, upper)
        mu = s.mean()
        sd = s.std()
        if sd < 1e-10:
            return s * 0
        return (s - mu) / sd
    
    df['factor'] = df.groupby('date')['factor'].transform(mad_wz)
    
    # 输出(正方向)
    output = df[['date', 'stock_code', 'factor']].dropna(subset=['factor']).copy()
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')
    
    out_path = data_dir / 'factor_turnover_kurt_v1.csv'
    output.to_csv(out_path, index=False)
    print(f"因子文件: {out_path}")
    print(f"记录数: {len(output)}, 日期: {output['date'].min()} ~ {output['date'].max()}")
    print(f"因子统计: mean={output['factor'].mean():.4f}, std={output['factor'].std():.4f}")
    date_counts = output.groupby('date')['stock_code'].count()
    print(f"每日覆盖: {date_counts.mean():.0f}")

if __name__ == '__main__':
    main()
