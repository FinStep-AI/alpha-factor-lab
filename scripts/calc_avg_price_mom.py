#!/usr/bin/env python3
"""
因子：平均成交价格偏离动量 (Average Trade Price Deviation Momentum)
ID: avg_price_dev_v1

逻辑：
  avg_price = amount / volume = 当日加权平均成交价格(VWAP代理)
  vwap_ratio = avg_price / close
  当avg_price > close → 日内买方力量在较高价格成交更多 → 买方主导
  当avg_price < close → 收盘弱于均价 → 卖方主导尾盘
  
  对vwap_ratio取20日均值，然后计算其5日变化率（动量）：
  factor = MA20(vwap_ratio)_t / MA20(vwap_ratio)_{t-5} - 1
  
  这捕捉的是VWAP相对收盘价偏离的趋势方向变化。
  如果偏离持续扩大（vwap_ratio上升趋势）=买方力量在增强。

  成交额OLS中性化 + MAD winsorize + z-score。
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

WINDOW = 20
MOM_LAG = 5

def main():
    base = Path(__file__).resolve().parent.parent
    data_dir = base / 'data'
    
    print("读取K线数据...")
    df = pd.read_csv(data_dir / 'csi1000_kline_raw.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['pct_change'])
    
    print(f"股票数: {df['stock_code'].nunique()}, 行数: {len(df)}")
    
    # 计算avg_price
    df['avg_price'] = df['amount'] / (df['volume'] * 100 + 1e-10)  # volume是手，*100=股
    df['vwap_ratio'] = df['avg_price'] / df['close']
    
    # 滚动20日均值
    df['vwap_ratio_ma'] = df.groupby('stock_code')['vwap_ratio'].transform(
        lambda x: x.rolling(WINDOW, min_periods=10).mean()
    )
    
    # 5日动量
    df['factor'] = df.groupby('stock_code')['vwap_ratio_ma'].transform(
        lambda x: x / x.shift(MOM_LAG) - 1
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
    
    out_path = data_dir / 'factor_avg_price_dev_v1.csv'
    output.to_csv(out_path, index=False)
    print(f"因子文件: {out_path}")
    print(f"记录数: {len(output)}, 日期: {output['date'].min()} ~ {output['date'].max()}")
    print(f"因子统计: mean={output['factor'].mean():.4f}, std={output['factor'].std():.4f}")
    date_counts = output.groupby('date')['stock_code'].count()
    print(f"每日覆盖: {date_counts.mean():.0f}")

if __name__ == '__main__':
    main()
