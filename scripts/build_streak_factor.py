"""
因子: 连涨连跌比 (Streak Asymmetry Factor)
逻辑: 过去20日中 max_down_streak - max_up_streak
      高值 = 近期有较长连跌但没有等长连涨 = 卖压释放信号
      
灵感: 类似neg_day_freq(做多近期多次下跌)但更强调连续性
      连续下跌 = 抛售惯性 → 一旦停止则反弹力度大
      
中性化: 成交额OLS中性化 + MAD winsorize + z-score
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
kline = pd.read_csv('data/csi1000_kline_raw.csv')
kline['date'] = pd.to_datetime(kline['date'])
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline['ret'] = kline['pct_change']
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)

print(f"Data: {kline['stock_code'].nunique()} stocks, {kline['date'].nunique()} dates")

# Pivot
ret_wide = kline.pivot(index='date', columns='stock_code', values='ret')
amt_wide = kline.pivot(index='date', columns='stock_code', values='amount')

dates = ret_wide.index
stocks = ret_wide.columns.values

window = 20
min_periods = 15

print("Computing streak factors...")

# For each stock, compute rolling max consecutive down streak and up streak
# We'll process column by column since streaks are inherently sequential

def compute_rolling_streaks(ret_series, window=20):
    """Compute rolling max up-streak and max down-streak within window"""
    n = len(ret_series)
    max_up_streak = np.full(n, np.nan)
    max_down_streak = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        w = ret_series[i - window + 1:i + 1]
        valid = ~np.isnan(w)
        if valid.sum() < min_periods:
            continue
        
        # Compute streaks
        max_up = 0
        max_down = 0
        cur_up = 0
        cur_down = 0
        
        for v in w:
            if np.isnan(v):
                continue
            if v > 0:
                cur_up += 1
                cur_down = 0
                if cur_up > max_up:
                    max_up = cur_up
            elif v < 0:
                cur_down += 1
                cur_up = 0
                if cur_down > max_down:
                    max_down = cur_down
            else:
                cur_up = 0
                cur_down = 0
        
        max_up_streak[i] = max_up
        max_down_streak[i] = max_down
    
    return max_up_streak, max_down_streak

# Process all stocks - this is sequential per stock but stocks are independent
up_streaks = np.full((len(dates), len(stocks)), np.nan)
down_streaks = np.full((len(dates), len(stocks)), np.nan)

for j, stock in enumerate(stocks):
    if j % 100 == 0:
        print(f"  Processing stock {j}/{len(stocks)}...")
    ret_vals = ret_wide[stock].values
    up_s, down_s = compute_rolling_streaks(ret_vals, window)
    up_streaks[:, j] = up_s
    down_streaks[:, j] = down_s

print("Computing factor values...")

# Factor = max_down_streak - max_up_streak (做多连跌多的)
# Alternative: just max_down_streak (纯下跌连续性)
# Let's try both and output the better one

streak_diff = down_streaks - up_streaks  # positive = more down streaks

# Also compute log_amt for neutralization
log_amt = np.log(np.clip(amt_wide.values, 1, None))
roll_log_amt = pd.DataFrame(log_amt, index=dates, columns=stocks).rolling(window=window, min_periods=min_periods).mean().values

def mad_winsorize(arr, n_mad=5):
    valid = arr[np.isfinite(arr)]
    if len(valid) < 10:
        return arr
    median = np.median(valid)
    mad = np.median(np.abs(valid - median))
    if mad < 1e-10:
        return arr
    upper = median + n_mad * 1.4826 * mad
    lower = median - n_mad * 1.4826 * mad
    return np.clip(arr, lower, upper)

def neutralize_and_zscore(factor_row, amt_row):
    mask = np.isfinite(factor_row) & np.isfinite(amt_row)
    if mask.sum() < 50:
        return np.full_like(factor_row, np.nan)
    
    # MAD winsorize
    f_w = factor_row.copy()
    valid_vals = f_w[mask]
    median = np.median(valid_vals)
    mad = np.median(np.abs(valid_vals - median))
    if mad > 1e-10:
        upper = median + 5 * 1.4826 * mad
        lower = median - 5 * 1.4826 * mad
        f_w = np.clip(f_w, lower, upper)
    
    # OLS neutralize
    X = amt_row[mask]
    y = f_w[mask]
    X_mean = X.mean()
    y_mean = y.mean()
    beta = np.sum((X - X_mean) * (y - y_mean)) / (np.sum((X - X_mean)**2) + 1e-10)
    alpha = y_mean - beta * X_mean
    
    result = np.full_like(factor_row, np.nan)
    result[mask] = y - (alpha + beta * X)
    
    # Z-score
    valid_r = result[np.isfinite(result)]
    if len(valid_r) < 30:
        return np.full_like(factor_row, np.nan)
    mu = valid_r.mean()
    sigma = valid_r.std()
    if sigma < 1e-10:
        return np.full_like(factor_row, np.nan)
    result = (result - mu) / sigma
    
    return result

# Process
records = []
for i, date in enumerate(dates):
    factor_row = streak_diff[i, :]
    amt_row = roll_log_amt[i, :]
    
    z = neutralize_and_zscore(factor_row, amt_row)
    
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
print(f"Factor stats: mean={factor_df['factor_value'].mean():.4f}, std={factor_df['factor_value'].std():.4f}")

factor_df.to_csv('data/factor_streak_asym_v1.csv', index=False)
print("Saved to data/factor_streak_asym_v1.csv")
