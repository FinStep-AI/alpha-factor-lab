"""
因子: 近期低点反弹幅度 (Bounce from Recent Low)
逻辑: close / min(low, 10d) - 1
      高值 = 近期大幅反弹 = 可能是动量信号
      低值 = 仍在低位附近 = 可能是价值/反转信号
      
测试两个方向:
  正向: 做多反弹大的（动量延续）
  反向: 做多仍在低位的（反转）

also try: close / max(high, 10d) - 1 (distance from high)

中性化: 成交额OLS + MAD + z-score
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
kline = pd.read_csv('data/csi1000_kline_raw.csv')
kline['date'] = pd.to_datetime(kline['date'])
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)

# Pivot
close_wide = kline.pivot(index='date', columns='stock_code', values='close')
low_wide = kline.pivot(index='date', columns='stock_code', values='low')
high_wide = kline.pivot(index='date', columns='stock_code', values='high')
amt_wide = kline.pivot(index='date', columns='stock_code', values='amount')

dates = close_wide.index
stocks = close_wide.columns

print(f"Data shape: {close_wide.shape}")

# Factor 1: Bounce from 10d low
window = 10
roll_min_low = low_wide.rolling(window=window, min_periods=7).min()
bounce_low = (close_wide / roll_min_low) - 1  # How far above 10d low

# Factor 2: Distance from 10d high  
roll_max_high = high_wide.rolling(window=window, min_periods=7).max()
dist_high = (close_wide / roll_max_high) - 1  # How far below 10d high (negative)

# Factor 3: Combined range position: (close - 10d_low) / (10d_high - 10d_low)
range_pos = (close_wide - roll_min_low) / (roll_max_high - roll_min_low + 1e-10)

# Neutralization setup
log_amt = np.log(amt_wide.clip(lower=1))
roll_log_amt = log_amt.rolling(window=20, min_periods=15).mean()

def neutralize_and_zscore(factor_vals, neutralizer_vals, min_count=50):
    mask = np.isfinite(factor_vals) & np.isfinite(neutralizer_vals)
    if mask.sum() < min_count:
        return np.full_like(factor_vals, np.nan)
    f_w = factor_vals.copy()
    valid = f_w[mask]
    median = np.median(valid)
    mad = np.median(np.abs(valid - median))
    if mad > 1e-10:
        upper = median + 5 * 1.4826 * mad
        lower = median - 5 * 1.4826 * mad
        f_w = np.clip(f_w, lower, upper)
    X = neutralizer_vals[mask]
    y = f_w[mask]
    X_mean = X.mean()
    y_mean = y.mean()
    beta = np.sum((X - X_mean) * (y - y_mean)) / (np.sum((X - X_mean)**2) + 1e-10)
    alpha = y_mean - beta * X_mean
    result = np.full_like(factor_vals, np.nan)
    result[mask] = y - (alpha + beta * X)
    valid_r = result[np.isfinite(result)]
    if len(valid_r) < 30:
        return np.full_like(factor_vals, np.nan)
    mu = valid_r.mean()
    sigma = valid_r.std()
    if sigma < 1e-10:
        return np.full_like(factor_vals, np.nan)
    return (result - mu) / sigma

# Process bounce_low factor
print("Processing bounce_low factor...")
records = []
bl_vals = bounce_low.values
la_vals = roll_log_amt.values

for i, date in enumerate(dates):
    z = neutralize_and_zscore(bl_vals[i, :], la_vals[i, :])
    for j in range(len(stocks)):
        if np.isfinite(z[j]):
            records.append({'date': date, 'stock_code': stocks[j], 'factor_value': z[j]})

factor_df = pd.DataFrame(records)
factor_df['date'] = pd.to_datetime(factor_df['date'])
factor_df.to_csv('data/factor_bounce_low_v1.csv', index=False)
print(f"bounce_low: {len(factor_df)} rows, saved")

# Process range_pos factor  
print("Processing range_pos factor...")
records2 = []
rp_vals = range_pos.values

for i, date in enumerate(dates):
    z = neutralize_and_zscore(rp_vals[i, :], la_vals[i, :])
    for j in range(len(stocks)):
        if np.isfinite(z[j]):
            records2.append({'date': date, 'stock_code': stocks[j], 'factor_value': z[j]})

factor_df2 = pd.DataFrame(records2)
factor_df2['date'] = pd.to_datetime(factor_df2['date'])
factor_df2.to_csv('data/factor_range_pos_v1.csv', index=False)
print(f"range_pos: {len(factor_df2)} rows, saved")
