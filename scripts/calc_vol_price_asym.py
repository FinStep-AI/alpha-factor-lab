#!/usr/bin/env python3
"""
因子：放量涨跌不对称性 (Volume-Price Asymmetry)
ID: vol_price_asym_v1

逻辑：
  统计过去N天中：
    - 高换手率(>中位数)且收益为正的天数比例 = buy_pressure
    - 高换手率(>中位数)且收益为负的天数比例 = sell_pressure
    factor = buy_pressure - sell_pressure
  
  高因子值 = 放量时更多是上涨 = 买方力量在大成交量时占优
  低因子值 = 放量时更多是下跌 = 卖方出逃
  
  区别于pv_corr(线性相关)：这是非线性的，关注的是条件分布。
  区别于turnover类因子：不看换手率本身，看换手率与涨跌的交互。
  
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
    
    # 对每只股票计算滚动中位数换手率
    print("计算换手率中位数和放量涨跌不对称性...")
    
    df['ret'] = df['pct_change'] / 100.0
    
    # 用turnover作为换手率
    # 计算每只股票的20日滚动中位数
    df['turnover_median'] = df.groupby('stock_code')['turnover'].transform(
        lambda x: x.rolling(WINDOW, min_periods=10).median()
    )
    
    # 标记高换手率日(>滚动中位数)
    df['high_turnover'] = (df['turnover'] > df['turnover_median']).astype(float)
    
    # 放量上涨 = high_turnover AND ret > 0
    df['hv_up'] = ((df['high_turnover'] == 1) & (df['ret'] > 0)).astype(float)
    # 放量下跌 = high_turnover AND ret < 0
    df['hv_down'] = ((df['high_turnover'] == 1) & (df['ret'] < 0)).astype(float)
    
    # 20日滚动求和得到天数比例
    df['buy_ratio'] = df.groupby('stock_code')['hv_up'].transform(
        lambda x: x.rolling(WINDOW, min_periods=10).mean()
    )
    df['sell_ratio'] = df.groupby('stock_code')['hv_down'].transform(
        lambda x: x.rolling(WINDOW, min_periods=10).mean()
    )
    
    # 因子 = 放量涨比例 - 放量跌比例
    df['factor'] = df['buy_ratio'] - df['sell_ratio']
    
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
    
    out_path = data_dir / 'factor_vol_price_asym_v1.csv'
    output.to_csv(out_path, index=False)
    print(f"因子文件: {out_path}")
    print(f"记录数: {len(output)}, 日期: {output['date'].min()} ~ {output['date'].max()}")
    print(f"因子统计: mean={output['factor'].mean():.4f}, std={output['factor'].std():.4f}")
    date_counts = output.groupby('date')['stock_code'].count()
    print(f"每日覆盖: {date_counts.mean():.0f}")

if __name__ == '__main__':
    main()
