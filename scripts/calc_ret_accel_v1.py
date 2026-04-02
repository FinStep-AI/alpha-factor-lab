"""
因子：收益率趋势加速度 (Return Trend Acceleration) v1
逻辑：对每只股票过去20日日收益率序列做时间回归，取斜率。
      正斜率 = 收益率在加速上升(趋势增强)
      负斜率 = 收益率在减速(趋势减弱)
      与均线离散度不同：离散度看趋势是否存在，加速度看趋势是否在增强
市值中性化：OLS on log_amount_20d
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# --- load data ---
kline = pd.read_csv('data/csi1000_kline_raw.csv', parse_dates=['date'])
kline.sort_values(['stock_code', 'date'], inplace=True)

# compute daily return
kline['ret'] = kline.groupby('stock_code')['close'].pct_change()

# 20d rolling OLS slope of returns on time index [0,1,...,19]
WINDOW = 20

def rolling_slope(arr):
    """OLS slope of arr on [0,1,...,len-1], returns slope."""
    n = len(arr)
    valid = ~np.isnan(arr)
    if valid.sum() < n * 0.7:
        return np.nan
    x = np.arange(n, dtype=float)
    y = np.where(valid, arr, 0)
    # fast OLS: slope = cov(x,y)/var(x)
    x_mean = x.mean()
    y_valid = arr[valid]
    x_valid = x[valid]
    x_mean = x_valid.mean()
    y_mean = y_valid.mean()
    slope = np.sum((x_valid - x_mean) * (y_valid - y_mean)) / np.sum((x_valid - x_mean) ** 2)
    return slope

records = []
for code, grp in kline.groupby('stock_code'):
    grp = grp.sort_values('date').reset_index(drop=True)
    rets = grp['ret'].values
    dates = grp['date'].values
    amounts = grp['amount'].values
    
    for i in range(WINDOW, len(grp)):
        window_rets = rets[i-WINDOW:i]
        # slope
        valid = ~np.isnan(window_rets)
        if valid.sum() < WINDOW * 0.7:
            continue
        x = np.arange(WINDOW, dtype=float)
        y = window_rets
        x_v = x[valid]
        y_v = y[valid]
        xm = x_v.mean()
        ym = y_v.mean()
        slope = np.sum((x_v - xm) * (y_v - ym)) / np.sum((x_v - xm) ** 2)
        
        # 20d avg amount for neutralization
        amt_window = amounts[max(0,i-WINDOW):i]
        avg_amt = np.nanmean(amt_window)
        
        records.append({
            'date': dates[i],
            'stock_code': code,
            'raw_factor': slope,
            'log_amount_20d': np.log(avg_amt + 1)
        })

df = pd.DataFrame(records)
print(f"Raw records: {len(df)}, dates: {df['date'].nunique()}, stocks: {df['stock_code'].nunique()}")

# --- Cross-sectional neutralization ---
def neutralize_and_normalize(group):
    y = group['raw_factor'].values
    x = group['log_amount_20d'].values
    
    valid = ~(np.isnan(y) | np.isnan(x))
    if valid.sum() < 50:
        group['factor'] = np.nan
        return group
    
    # OLS: y = a + b*x + epsilon
    from numpy.linalg import lstsq
    X = np.column_stack([np.ones(valid.sum()), x[valid]])
    beta, _, _, _ = lstsq(X, y[valid], rcond=None)
    resid = np.full(len(y), np.nan)
    resid[valid] = y[valid] - X @ beta
    
    # MAD winsorize
    med = np.nanmedian(resid[valid])
    mad = np.nanmedian(np.abs(resid[valid] - med))
    if mad < 1e-12:
        group['factor'] = np.nan
        return group
    upper = med + 5 * 1.4826 * mad
    lower = med - 5 * 1.4826 * mad
    resid = np.clip(resid, lower, upper)
    
    # z-score
    mu = np.nanmean(resid)
    std = np.nanstd(resid)
    if std < 1e-12:
        group['factor'] = np.nan
        return group
    group['factor'] = (resid - mu) / std
    return group

df = df.groupby('date', group_keys=False).apply(neutralize_and_normalize)
df = df[['date', 'stock_code', 'factor']].dropna()
print(f"After neutralization: {len(df)}")

# save
df.to_csv('data/factor_ret_accel_v1.csv', index=False)
print("Saved to data/factor_ret_accel_v1.csv")
print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")
print(f"Sample stats: mean={df['factor'].mean():.4f}, std={df['factor'].std():.4f}")
