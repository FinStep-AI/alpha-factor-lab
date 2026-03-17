"""
因子: 成交额排名变化 (Amount Rank Change Factor)
逻辑: 每只股票在截面中的成交额排名(百分位)的20日变化
      stock_rank_today - stock_rank_20d_ago
      正值 = 相对成交额排名上升 = 关注度相对增加
      
假设: 排名上升=相对关注度增加，与turnover_level(绝对水平)不同
     可能捕捉到"从冷门变热门"的过程
     
中性化: 因为排名本身已经cross-sectional normalized,
       只做MAD + z-score，不额外做成交额OLS中性化
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

# Pivot amount
amt_wide = kline.pivot(index='date', columns='stock_code', values='amount')
dates = amt_wide.index
stocks = amt_wide.columns

print(f"Data shape: {amt_wide.shape}")

# For each date, compute cross-sectional percentile rank of amount
print("Computing daily amount ranks...")
rank_wide = amt_wide.rank(axis=1, pct=True)  # percentile rank across stocks each day

# 5-day smoothed rank (reduce noise)
rank_smooth = rank_wide.rolling(window=5, min_periods=3).mean()

# Rank change: current smooth rank - 20d ago smooth rank
rank_change = rank_smooth - rank_smooth.shift(20)

print(f"Rank change stats: mean={rank_change.stack().mean():.4f}, std={rank_change.stack().std():.4f}")

# Also compute log_amt for optional neutralization
log_amt = np.log(amt_wide.clip(lower=1))
roll_log_amt = log_amt.rolling(window=20, min_periods=15).mean()

# Process daily cross-sections
def neutralize_and_zscore(factor_vals, neutralizer_vals, min_count=50):
    mask = np.isfinite(factor_vals) & np.isfinite(neutralizer_vals)
    if mask.sum() < min_count:
        return np.full_like(factor_vals, np.nan)
    
    # MAD winsorize
    f_w = factor_vals.copy()
    valid = f_w[mask]
    median = np.median(valid)
    mad = np.median(np.abs(valid - median))
    if mad > 1e-10:
        upper = median + 5 * 1.4826 * mad
        lower = median - 5 * 1.4826 * mad
        f_w = np.clip(f_w, lower, upper)
    
    # OLS neutralize by log_amt
    X = neutralizer_vals[mask]
    y = f_w[mask]
    X_mean = X.mean()
    y_mean = y.mean()
    beta = np.sum((X - X_mean) * (y - y_mean)) / (np.sum((X - X_mean)**2) + 1e-10)
    alpha = y_mean - beta * X_mean
    
    result = np.full_like(factor_vals, np.nan)
    result[mask] = y - (alpha + beta * X)
    
    # Z-score
    valid_r = result[np.isfinite(result)]
    if len(valid_r) < 30:
        return np.full_like(factor_vals, np.nan)
    mu = valid_r.mean()
    sigma = valid_r.std()
    if sigma < 1e-10:
        return np.full_like(factor_vals, np.nan)
    result = (result - mu) / sigma
    
    return result

records = []
rc_vals = rank_change.values
la_vals = roll_log_amt.values

for i, date in enumerate(dates):
    z = neutralize_and_zscore(rc_vals[i, :], la_vals[i, :])
    for j in range(len(stocks)):
        if np.isfinite(z[j]):
            records.append({
                'date': date,
                'stock_code': stocks[j],
                'factor_value': z[j]
            })

factor_df = pd.DataFrame(records)
factor_df['date'] = pd.to_datetime(factor_df['date'])
print(f"Final factor: {len(factor_df)} rows")
print(f"Date range: {factor_df['date'].min()} to {factor_df['date'].max()}")

factor_df.to_csv('data/factor_amt_rank_change_v1.csv', index=False)
print("Saved to data/factor_amt_rank_change_v1.csv")
