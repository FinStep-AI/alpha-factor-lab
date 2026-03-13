"""
跳空缺口强度因子 (Gap Intensity) v1
====================================
逻辑：
- A股集合竞价是知情交易者博弈最集中的时段
- 跳空缺口的大小相对于前一天的波动范围(high-low)反映了新信息冲击的强度
- 持续出现强跳空(无论方向)的股票 = 信息密集到达 = 高关注度
- 持续出现弱跳空的股票 = 信息平静 = 被忽视

构造方法（简洁版）：
  gap_intensity = |open_t - close_{t-1}| / (high_{t-1} - low_{t-1})
  factor = -EWM(gap_intensity, 10d)
  
  方向假设：高跳空强度（被关注）→ 未来表现差（overreaction/attention premium）
  
也测试正方向：高跳空 → 高收益（动量/信息传播）
"""

import pandas as pd
import numpy as np

def compute_factor():
    print("Loading data...")
    df = pd.read_csv('data/csi1000_kline_raw.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    df['prev_close'] = df.groupby('stock_code')['close'].shift(1)
    df['prev_range'] = df.groupby('stock_code').apply(
        lambda g: g['high'] - g['low']
    ).reset_index(level=0, drop=True)
    df['prev_range_shift'] = df.groupby('stock_code')['prev_range'].shift(0)
    
    # Actually use previous day's range
    df['prev_high'] = df.groupby('stock_code')['high'].shift(1)
    df['prev_low'] = df.groupby('stock_code')['low'].shift(1)
    df['prev_day_range'] = df['prev_high'] - df['prev_low']
    
    # Gap intensity
    df['gap'] = (df['open'] - df['prev_close']).abs()
    df['gap_intensity'] = df['gap'] / (df['prev_day_range'] + 0.001)
    df['gap_intensity'] = df['gap_intensity'].clip(upper=5.0)  # cap outliers
    
    # Also compute signed gap relative to range
    df['signed_gap_intensity'] = (df['open'] - df['prev_close']) / (df['prev_day_range'] + 0.001)
    df['signed_gap_intensity'] = df['signed_gap_intensity'].clip(lower=-5.0, upper=5.0)
    
    # v1: -EWM of gap_intensity (high attention → low factor → underperform hypothesis)
    df['factor_v1'] = df.groupby('stock_code')['gap_intensity'].transform(
        lambda x: -x.ewm(span=10, min_periods=5).mean()
    )
    
    # v2: +EWM of gap_intensity (momentum hypothesis)
    df['factor_v2'] = df.groupby('stock_code')['gap_intensity'].transform(
        lambda x: x.ewm(span=10, min_periods=5).mean()
    )
    
    # v3: EWM of signed_gap_intensity (directional gap)
    df['factor_v3'] = df.groupby('stock_code')['signed_gap_intensity'].transform(
        lambda x: x.ewm(span=10, min_periods=5).mean()
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
        
        def neutralize(g):
            y, x = g['factor_raw'].values, g['log_amount'].values
            v = ~(np.isnan(y)|np.isnan(x))
            if v.sum()<30: g['factor']=np.nan; return g
            X = np.column_stack([np.ones(v.sum()), x[v]])
            try:
                b = np.linalg.lstsq(X, y[v], rcond=None)[0]
                r = np.full(len(y), np.nan); r[v] = y[v] - X@b; g['factor'] = r
            except: g['factor'] = np.nan
            return g
        
        fdf = fdf.groupby('date', group_keys=False).apply(neutralize)
        
        def winsorize_mad(g):
            v = g['factor'].values; m = ~np.isnan(v)
            if m.sum()<10: return g
            med = np.nanmedian(v); mad = np.nanmedian(np.abs(v[m]-med))
            if mad<1e-10: return g
            g['factor'] = np.clip(v, med-3*1.4826*mad, med+3*1.4826*mad)
            return g
        
        fdf = fdf.groupby('date', group_keys=False).apply(winsorize_mad)
        
        def zscore(g):
            v = g['factor'].values; m = ~np.isnan(v)
            if m.sum()<10: return g
            mn, s = np.nanmean(v[m]), np.nanstd(v[m])
            g['factor'] = (v-mn)/s if s>1e-10 else 0.0
            return g
        
        fdf = fdf.groupby('date', group_keys=False).apply(zscore)
        out = fdf[['date','stock_code','factor']].dropna(subset=['factor'])
        out.to_csv(f'data/factor_gap_intensity_{version}.csv', index=False)
        print(f"Gap Intensity {version}: {out.shape}")

if __name__ == '__main__':
    compute_factor()
