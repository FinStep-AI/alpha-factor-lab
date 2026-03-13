"""
开盘竞价强度因子 (Open Auction Power) v1
========================================
逻辑：
- 集合竞价阶段是知情交易者最活跃的时段
- 如果开盘跳空占当日波幅的比例很高，说明信息主要在开盘定价
- 如果开盘跳空只占小比例，说明连续交易阶段更重要(散户/噪声)
  
构造：
  open_power = |open - prev_close| / (high - low + 0.001)
  高 open_power = 集合竞价主导(信息驱动)
  低 open_power = 连续交易主导(噪声驱动)

  factor = 20日加权均值，近日权重更大

也测试：
  v2: open_power × sign(cum_ret_5d) — 方向性加权
  v3: open_power × (turnover / turnover_20d) — 量能异常日的竞价强度
"""

import pandas as pd
import numpy as np

def compute_factor():
    print("Loading data...")
    df = pd.read_csv('data/csi1000_kline_raw.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    df['prev_close'] = df.groupby('stock_code')['close'].shift(1)
    df['daily_ret'] = df.groupby('stock_code')['close'].pct_change()
    
    # Open auction power
    df['range'] = df['high'] - df['low']
    df['gap'] = (df['open'] - df['prev_close']).abs()
    df['open_power'] = df['gap'] / (df['range'] + 0.001)
    df['open_power'] = df['open_power'].clip(upper=1.0)
    
    # Direction of gap
    df['gap_sign'] = np.sign(df['open'] - df['prev_close'])
    
    # Signed open power
    df['signed_op'] = df['gap_sign'] * df['open_power']
    
    # v1: 20d EWM of open_power
    df['factor_v1'] = df.groupby('stock_code')['open_power'].transform(
        lambda x: x.ewm(span=20, min_periods=10).mean()
    )
    
    # v2: signed version - 20d sum of signed_op
    df['factor_v2'] = df.groupby('stock_code')['signed_op'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    
    # v3: open_power × daily return interaction → captures if gap-driven days predict direction
    df['op_ret'] = df['open_power'] * df['daily_ret']
    df['factor_v3'] = df.groupby('stock_code')['op_ret'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    
    # Market cap neutralization
    df['avg_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    df['log_amount'] = np.log(df['avg_amount_20d'].clip(lower=1))
    
    for version, col in [('v1', 'factor_v1'), ('v2', 'factor_v2'), ('v3', 'factor_v3')]:
        fdf = df[['date', 'stock_code', col, 'log_amount']].copy()
        fdf = fdf.rename(columns={col: 'factor_raw'})
        fdf = fdf.dropna(subset=['factor_raw', 'log_amount'])
        
        def neutralize(group):
            y = group['factor_raw'].values
            x = group['log_amount'].values
            valid = ~(np.isnan(y) | np.isnan(x))
            if valid.sum() < 30:
                group['factor'] = np.nan
                return group
            y_v, x_v = y[valid], x[valid]
            X = np.column_stack([np.ones(len(x_v)), x_v])
            try:
                beta = np.linalg.lstsq(X, y_v, rcond=None)[0]
                res = np.full(len(y), np.nan)
                res[valid] = y_v - X @ beta
                group['factor'] = res
            except:
                group['factor'] = np.nan
            return group
        
        fdf = fdf.groupby('date', group_keys=False).apply(neutralize)
        
        def winsorize_mad(group):
            vals = group['factor'].values
            valid = ~np.isnan(vals)
            if valid.sum() < 10: return group
            med = np.nanmedian(vals)
            mad = np.nanmedian(np.abs(vals[valid] - med))
            if mad < 1e-10: return group
            group['factor'] = np.clip(vals, med - 3*1.4826*mad, med + 3*1.4826*mad)
            return group
        
        fdf = fdf.groupby('date', group_keys=False).apply(winsorize_mad)
        
        def zscore(group):
            vals = group['factor'].values
            valid = ~np.isnan(vals)
            if valid.sum() < 10: return group
            m, s = np.nanmean(vals[valid]), np.nanstd(vals[valid])
            if s < 1e-10: group['factor'] = 0.0
            else: group['factor'] = (vals - m) / s
            return group
        
        fdf = fdf.groupby('date', group_keys=False).apply(zscore)
        
        out = fdf[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
        out.to_csv(f'data/factor_open_power_{version}.csv', index=False)
        print(f"Open Power {version}: {out.shape}")

if __name__ == '__main__':
    compute_factor()
