"""
换手率加速度因子 (Turnover Acceleration)
========================================
逻辑:
  - MA5(turnover) / MA20(turnover) - 1 = 换手率短期相对长期的变化
  - 正向: 换手率加速上升 = 关注度突然提升 = 信息事件正在发酵
  
  另外测试:
  v1: MA5/MA20 - 1 (短期/长期比)
  v2: turnover的20日线性回归斜率 (趋势方向)
  v3: MA5/MA60 (更长参考窗口)
  
成交额OLS中性化 + MAD winsorize + z-score
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression

kline = pd.read_csv('data/csi1000_kline_raw.csv', parse_dates=['date'])
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)

print(f"行情: {len(kline)} 行, {kline['stock_code'].nunique()} 只股票")

# 各种MA
kline['to_ma5'] = kline.groupby('stock_code')['turnover'].transform(
    lambda x: x.rolling(5, min_periods=3).mean())
kline['to_ma20'] = kline.groupby('stock_code')['turnover'].transform(
    lambda x: x.rolling(20, min_periods=15).mean())
kline['to_ma60'] = kline.groupby('stock_code')['turnover'].transform(
    lambda x: x.rolling(60, min_periods=40).mean())

# 20日平均成交额 (中性化用)
kline['log_amt_20'] = np.log(
    kline.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=15).mean()) + 1)

# ── 构建多个变体 ──────────────────────────────────────────
# v1: MA5/MA20 - 1 (短/长比)
kline['v1_raw'] = kline['to_ma5'] / (kline['to_ma20'] + 1e-10) - 1

# v2: 20日换手率趋势斜率
def rolling_slope(x, window=20):
    result = np.full(len(x), np.nan)
    t = np.arange(window)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()
    for i in range(window - 1, len(x)):
        y = x.iloc[i - window + 1:i + 1].values
        if np.isnan(y).sum() > 5:
            continue
        y_clean = np.where(np.isnan(y), np.nanmean(y), y)
        slope = ((t - t_mean) * (y_clean - y_clean.mean())).sum() / t_var
        result[i] = slope
    return pd.Series(result, index=x.index)

# v2太慢，用numpy vectorize
def fast_slope_20(series):
    """快速计算20日rolling slope"""
    vals = series.values
    n = 20
    result = np.full(len(vals), np.nan)
    t = np.arange(n, dtype=float)
    t_mean = t.mean()
    t_demeaned = t - t_mean
    ss_t = (t_demeaned ** 2).sum()
    
    for i in range(n - 1, len(vals)):
        window = vals[i - n + 1:i + 1]
        if np.isnan(window).sum() > 5:
            continue
        window_clean = np.where(np.isnan(window), np.nanmean(window), window)
        slope = (t_demeaned * (window_clean - window_clean.mean())).sum() / ss_t
        result[i] = slope
    return pd.Series(result, index=series.index)

print("计算换手率趋势斜率(v2)...")
kline['v2_raw'] = kline.groupby('stock_code')['turnover'].transform(fast_slope_20)

# v3: MA5/MA60 (更长期基准)
kline['v3_raw'] = kline['to_ma5'] / (kline['to_ma60'] + 1e-10) - 1

def neutralize_and_zscore(df_in, col):
    """中性化 + MAD + z-score"""
    df = df_in.dropna(subset=[col, 'log_amt_20']).copy()
    
    def neutralize(group):
        y = group[col].values
        X = group['log_amt_20'].values.reshape(-1, 1)
        mask = np.isfinite(y) & np.isfinite(X.ravel())
        if mask.sum() < 30:
            return pd.Series(np.nan, index=group.index)
        reg = LinearRegression().fit(X[mask], y[mask])
        resid = np.full(len(y), np.nan)
        resid[mask] = y[mask] - reg.predict(X[mask])
        return pd.Series(resid, index=group.index)
    
    df['neutral'] = df.groupby('date').apply(neutralize).reset_index(level=0, drop=True)
    
    def mad_zscore(group):
        vals = group['neutral']
        median = vals.median()
        mad = (vals - median).abs().median()
        if mad < 1e-10:
            return pd.Series(0.0, index=group.index)
        upper = median + 5.2 * mad
        lower = median - 5.2 * mad
        clipped = vals.clip(lower, upper)
        mean = clipped.mean()
        std = clipped.std()
        if std < 1e-10:
            return pd.Series(0.0, index=group.index)
        return (clipped - mean) / std
    
    df['factor_value'] = df.groupby('date').apply(mad_zscore).reset_index(level=0, drop=True)
    return df[['date', 'stock_code', 'factor_value']].dropna()

# 处理3个变体
for name, col in [('v1', 'v1_raw'), ('v2', 'v2_raw'), ('v3', 'v3_raw')]:
    print(f"\n处理 {name}...")
    out = neutralize_and_zscore(kline, col)
    out.to_csv(f'data/factor_to_accel_{name}.csv', index=False)
    print(f"  输出: {len(out)} 行, {out['date'].min()} ~ {out['date'].max()}")
    
    # 也输出反向版本
    out_neg = out.copy()
    out_neg['factor_value'] = -out_neg['factor_value']
    out_neg.to_csv(f'data/factor_to_accel_{name}_neg.csv', index=False)

print("\n完成！")
