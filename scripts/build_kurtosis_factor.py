"""
因子: 收益率峰度 (Return Kurtosis Factor)
逻辑: 过去20日日收益率的超额峰度(excess kurtosis)
      高峰度 = 收益分布厚尾 = 更多极端事件

向量化计算，避免rolling apply的性能问题
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
kline = pd.read_csv('data/csi1000_kline_raw.csv')
kline['date'] = pd.to_datetime(kline['date'])
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline['ret'] = kline['pct_change']  # already ratio form
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)

print(f"Data: {kline['stock_code'].nunique()} stocks, {kline['date'].nunique()} dates")

# Pivot to wide format for vectorized rolling
print("Pivoting to wide format...")
ret_wide = kline.pivot(index='date', columns='stock_code', values='ret')
amt_wide = kline.pivot(index='date', columns='stock_code', values='amount')

dates = ret_wide.index
stocks = ret_wide.columns

print(f"Wide shape: {ret_wide.shape}")

# Vectorized rolling kurtosis using rolling moments
window = 20
min_periods = 15

print("Computing rolling kurtosis (vectorized)...")

# Rolling mean, var, and 4th central moment
roll_mean = ret_wide.rolling(window=window, min_periods=min_periods).mean()
roll_std = ret_wide.rolling(window=window, min_periods=min_periods).std()
roll_count = ret_wide.rolling(window=window, min_periods=min_periods).count()

# For kurtosis, use pandas rolling + apply with numpy
# But we need a faster approach. Let's compute rolling sum of (x-mean)^4 / (n*std^4) - 3
# Use the formula: kurt = n*(n+1)/((n-1)*(n-2)*(n-3)) * sum((x-mean)^4)/var^2 - 3*(n-1)^2/((n-2)*(n-3))

# Since vectorized rolling kurtosis isn't in pandas, compute using rolling apply on columns
# But limit to trading dates only to reduce computation

# Alternative: Use pd.DataFrame.rolling().kurt() which IS available!
print("Using pandas rolling().kurt()...")
roll_kurt = ret_wide.rolling(window=window, min_periods=min_periods).kurt()

print(f"Rolling kurtosis computed. Non-NaN count: {roll_kurt.notna().sum().sum()}")
print(f"Kurtosis stats: mean={roll_kurt.stack().mean():.3f}, median={roll_kurt.stack().median():.3f}")

# Rolling log average amount for neutralization
log_amt = np.log(amt_wide.clip(lower=1))
roll_log_amt = log_amt.rolling(window=window, min_periods=min_periods).mean()

# Stack back to long format for processing
print("Processing daily cross-sections...")

def mad_winsorize(s, n_mad=5):
    median = s.median()
    mad = (s - median).abs().median()
    if mad < 1e-10:
        return s
    upper = median + n_mad * 1.4826 * mad
    lower = median - n_mad * 1.4826 * mad
    return s.clip(lower, upper)

def neutralize_ols(factor_vals, neutralizer_vals):
    mask = np.isfinite(factor_vals) & np.isfinite(neutralizer_vals)
    if mask.sum() < 30:
        return np.full_like(factor_vals, np.nan)
    
    X = neutralizer_vals[mask]
    y = factor_vals[mask]
    X_mean = X.mean()
    y_mean = y.mean()
    beta = np.sum((X - X_mean) * (y - y_mean)) / (np.sum((X - X_mean)**2) + 1e-10)
    alpha = y_mean - beta * X_mean
    
    result = np.full_like(factor_vals, np.nan)
    result[mask] = y - (alpha + beta * X)
    return result

all_records = []
processed_dates = 0

for date in dates:
    kurt_vals = roll_kurt.loc[date].values
    amt_vals = roll_log_amt.loc[date].values
    stock_codes = stocks.values
    
    # Filter valid
    valid_mask = np.isfinite(kurt_vals) & np.isfinite(amt_vals)
    if valid_mask.sum() < 50:
        continue
    
    # MAD winsorize kurtosis
    kurt_series = pd.Series(kurt_vals)
    kurt_w = mad_winsorize(kurt_series).values
    
    # OLS neutralize
    residual = neutralize_ols(kurt_w, amt_vals)
    
    # Z-score
    valid_resid = residual[np.isfinite(residual)]
    if len(valid_resid) < 30:
        continue
    mu = valid_resid.mean()
    sigma = valid_resid.std()
    if sigma < 1e-10:
        continue
    z = (residual - mu) / sigma
    
    for i in range(len(stock_codes)):
        if np.isfinite(z[i]):
            all_records.append({
                'date': date,
                'stock_code': stock_codes[i],
                'factor_value': z[i]
            })
    
    processed_dates += 1

print(f"Processed {processed_dates} dates")

factor_df = pd.DataFrame(all_records)
factor_df['date'] = pd.to_datetime(factor_df['date'])
print(f"Final factor: {len(factor_df)} rows")
print(f"Date range: {factor_df['date'].min()} to {factor_df['date'].max()}")
print(f"Factor stats: mean={factor_df['factor_value'].mean():.4f}, std={factor_df['factor_value'].std():.4f}")

factor_df.to_csv('data/factor_ret_kurtosis_v1.csv', index=False)
print("Saved to data/factor_ret_kurtosis_v1.csv")
