"""
因子: 成交额波动率 (Amount Volatility)
逻辑: 过去20日 log(amount) 的标准差
      高值 = 成交额波动大 = 交易兴趣不稳定
      
假设(正向): 成交额波动大的股票可能有事件驱动，
           信息不确定性高 → 风险溢价 → 高预期收益
假设(反向): 低成交额波动 = 稳定交易 = 质量好
           
中性化: 成交额均值OLS中性化 + MAD winsorize + z-score
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
amt_wide = kline.pivot(index='date', columns='stock_code', values='amount')
dates = amt_wide.index
stocks = amt_wide.columns

print(f"Data shape: {amt_wide.shape}")

window = 20
min_periods = 15

# log(amount) then rolling std
log_amt = np.log(amt_wide.clip(lower=1))
roll_std = log_amt.rolling(window=window, min_periods=min_periods).std()
roll_mean = log_amt.rolling(window=window, min_periods=min_periods).mean()

print(f"Amount vol stats: mean={roll_std.stack().mean():.4f}, std={roll_std.stack().std():.4f}")

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
    
    # OLS neutralize
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
std_vals = roll_std.values
mean_vals = roll_mean.values

for i, date in enumerate(dates):
    z = neutralize_and_zscore(std_vals[i, :], mean_vals[i, :])
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

factor_df.to_csv('data/factor_amt_vol_v1.csv', index=False)
print("Saved to data/factor_amt_vol_v1.csv")
