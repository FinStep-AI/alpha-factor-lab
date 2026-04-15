#!/usr/bin/env python3
"""
Volume Shock Stability (VSS) Factor v1
源自：方正金工「适度冒险因子」的多因子选股系列研究之一

核心逻辑（适配日频数据）：
  - 检测成交量激增时刻（日频近似：amt/z_volume > 阈值）
  - 计算激增前后的收益波动
  - 低波动（稳定反应）= 市场对信息有共识 = 正alpha

因子：
  vss_vol20: 过去20日里成交量激增日的收益率标准差
  vss_stable20: 过去20日里vss_vol20的变异系数（越低越稳定）
  vss_composite: - (vss_vol20 + vss_stable20) / 2  越低越好
  经对数成交额中性化 + 横截面z-score
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def calc(kline_path='data/csi1000_kline_raw.csv', output_path=None):
    df = pd.read_csv(kline_path, parse_dates=['date'])
    df['code'] = df['stock_code']
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    grp = df.groupby('code')
    
    # 20-day rolling z-score of volume (standardized volume)
    df['vol_z20'] = grp['volume'].transform(lambda x: (x - x.rolling(20, min_periods=10).mean()) / 
                                                  (x.rolling(20, min_periods=10).std() + 1e-10))
    
    # Volume shock day: z-score > 1.2
    df['vol_shock'] = (df['vol_z20'] > 1.2).astype(int)
    
    # On shock days, compute return
    df['shock_ret'] = df['log_ret'].where(df['vol_shock'] == 1)
    
    # 20-day rolling volatility of shock returns (月度均"耀眼波动率")
    df['vss_vol20'] = grp['shock_ret'].transform(
        lambda x: x.rolling(20, min_periods=3).std()
    )
    
    # 20-day CV of vss_vol20 (月度稳"耀稳波动率"，越低越稳定)
    vss_ma = grp['vss_vol20'].transform(lambda x: x.rolling(20, min_periods=3).mean())
    vss_sd = grp['vss_vol20'].transform(lambda x: x.rolling(20, min_periods=3).std())
    df['vss_stable20'] = -(vss_sd / (vss_ma + 1e-10))  # negative: low CV = stable = better
    
    # Composite: -(vss_vol20 + vss_stable20) / 2
    # Lower = less volatile shock reactions = more consensus = better
    df['raw'] = -(df['vss_vol20'].fillna(0) + df['vss_stable20'].fillna(0)) / 2
    
    # Market-cap proxy neutralization (log_amount_20)
    df['log_amount_20'] = grp['amount'].transform(lambda x: np.log(x.rolling(20).mean().replace(0, np.nan)))
    
    result = df[['date', 'code', 'raw', 'log_amount_20']].dropna(subset=['raw', 'log_amount_20']).copy()
    
    all_res = []
    for date, g in result.groupby('date'):
        X = g['log_amount_20'].values.reshape(-1, 1)
        y = g['raw'].values
        lr = LinearRegression().fit(X, y)
        residual = y - lr.predict(X)
        sub = g[['date', 'code']].copy()
        sub['vss_v1'] = (residual - residual.mean()) / (residual.std() + 1e-10)
        all_res.append(sub)
    
    result = pd.concat(all_res, ignore_index=True)
    
    if output_path is None:
        output_path = 'data/factor_vss_v1.csv'
    result.to_csv(output_path, index=False)
    print(f"✅ {output_path} | {len(result)} rows, {result['code'].nunique()} stocks | last: {result['date'].max()}")
    return output_path

if __name__ == '__main__':
    calc()
