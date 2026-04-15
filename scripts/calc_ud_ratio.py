#!/usr/bin/env python3
"""
因子：日内上下行比率 (Upside-Downside Range Ratio)
ID: ud_ratio_v1

逻辑：
  upside = (high - open) / open   日内上冲幅度
  downside = (open - low) / open  日内下探幅度
  ratio = MA20(upside / downside)  (取对数消除偏度)
  
  高ratio = 日内更多向上探索 = 买方力量强 = 做多
  低ratio = 日内更多向下探索 = 卖方力量强 = 做空
  
  这和shadow_pressure不同：
  - shadow_pressure用上影线/下影线（相对于实体），衡量冲高回落
  - ud_ratio用(high-open)/(open-low)，衡量相对于开盘的上下探索
  
  成交额OLS中性化 + MAD winsorize + z-score。
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

WINDOW = 20

def main():
    base = Path(__file__).resolve().parent.parent
    data_dir = base / 'data'
    
    print("读取K线数据...")
    df = pd.read_csv(data_dir / 'csi1000_kline_raw.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['pct_change'])
    
    print(f"股票数: {df['stock_code'].nunique()}, 行数: {len(df)}")
    
    # 计算日内上冲和下探
    df['upside'] = (df['high'] - df['open']) / df['open']
    df['downside'] = (df['open'] - df['low']) / df['open']
    
    # 避免除以零: downside为0时设为NaN
    df['daily_ratio'] = np.where(
        df['downside'] > 1e-6,
        df['upside'] / df['downside'],
        np.nan
    )
    
    # 取对数
    df['log_ratio'] = np.log(df['daily_ratio'].clip(lower=0.01))
    
    # 20日滚动均值
    df['factor'] = df.groupby('stock_code')['log_ratio'].transform(
        lambda x: x.rolling(WINDOW, min_periods=10).mean()
    )
    
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
    
    # 输出
    output = df[['date', 'stock_code', 'factor']].dropna(subset=['factor']).copy()
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')
    
    out_path = data_dir / 'factor_ud_ratio_v1.csv'
    output.to_csv(out_path, index=False)
    print(f"因子文件: {out_path}")
    print(f"记录数: {len(output)}, 日期: {output['date'].min()} ~ {output['date'].max()}")
    print(f"因子统计: mean={output['factor'].mean():.4f}, std={output['factor'].std():.4f}")
    date_counts = output.groupby('date')['stock_code'].count()
    print(f"每日覆盖: {date_counts.mean():.0f}")

if __name__ == '__main__':
    main()
