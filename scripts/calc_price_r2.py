#!/usr/bin/env python3
"""
价格趋势R²因子 (Price Trend R-Squared)
=======================================
衡量过去N天收盘价走势的趋势平滑度。

Factor construction:
  对每只股票过去window天的log(close)对时间t做OLS回归:
    log(close_t) = alpha + beta * t + epsilon
  R² = 1 - SSR/SST
  
  高R² = 价格走势非常接近一条直线 = 趋势清晰/平滑
  低R² = 价格走势杂乱 = 无明确趋势

同时提取 slope 信息，构造方向性调整版本：
  v1: R² (方向无关的趋势平滑度)
  v2: sign(slope) * R² (方向+平滑度的联合信号)

Neutralization: OLS on log_amount + MAD winsorize + z-score
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression


def compute_price_r2(df, window=60):
    """Compute rolling Price Trend R² for all stocks."""
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    results = []
    t_arr = np.arange(window).reshape(-1, 1)  # time indices
    
    for code, grp in df.groupby('stock_code'):
        grp = grp.sort_values('date').reset_index(drop=True)
        if len(grp) < window + 5:
            continue
        
        log_close = np.log(grp['close'].values)
        dates = grp['date'].values
        
        for i in range(window - 1, len(grp)):
            y = log_close[i - window + 1: i + 1]
            
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                continue
            
            # OLS: y = a + b*t
            y_mean = np.mean(y)
            t_mean = (window - 1) / 2.0
            
            ss_tt = np.sum((np.arange(window) - t_mean) ** 2)
            ss_yt = np.sum((y - y_mean) * (np.arange(window) - t_mean))
            
            if ss_tt < 1e-12:
                continue
            
            slope = ss_yt / ss_tt
            y_pred = y_mean + slope * (np.arange(window) - t_mean)
            
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            
            if ss_tot < 1e-12:
                continue
            
            r2 = 1.0 - ss_res / ss_tot
            r2 = max(0, min(1, r2))
            
            results.append({
                'date': dates[i],
                'stock_code': code,
                'r2': r2,
                'slope': slope,
                'signed_r2': np.sign(slope) * r2
            })
    
    return pd.DataFrame(results)


def neutralize(factor_df, kline_df, col, window=60):
    """OLS neutralize on log_amount, MAD winsorize, z-score."""
    kline_sorted = kline_df.sort_values(['stock_code', 'date'])
    amt_data = []
    for code, grp in kline_sorted.groupby('stock_code'):
        grp = grp.sort_values('date')
        log_amt = np.log(grp['amount'].clip(lower=1))
        log_amt_ma = log_amt.rolling(window, min_periods=window).mean()
        for i in range(len(grp)):
            if pd.notna(log_amt_ma.iloc[i]):
                amt_data.append({
                    'date': grp['date'].iloc[i],
                    'stock_code': code,
                    'log_amount': log_amt_ma.iloc[i]
                })
    amt_df = pd.DataFrame(amt_data)
    merged = factor_df.merge(amt_df, on=['date', 'stock_code'], how='inner')
    
    output = []
    for dt, day_df in merged.groupby('date'):
        day_df = day_df.dropna(subset=[col, 'log_amount'])
        if len(day_df) < 30:
            continue
        
        X = day_df[['log_amount']].values
        y = day_df[col].values
        
        lr = LinearRegression()
        lr.fit(X, y)
        residual = y - lr.predict(X)
        
        med = np.median(residual)
        mad = np.median(np.abs(residual - med))
        if mad < 1e-10:
            mad = np.std(residual)
        upper = med + 5 * 1.4826 * mad
        lower = med - 5 * 1.4826 * mad
        residual = np.clip(residual, lower, upper)
        
        mean_r = np.mean(residual)
        std_r = np.std(residual)
        if std_r < 1e-10:
            continue
        z = (residual - mean_r) / std_r
        z = np.clip(z, -3, 3)
        
        for j, idx in enumerate(day_df.index):
            output.append({
                'date': dt,
                'stock_code': day_df.loc[idx, 'stock_code'],
                'factor_value': z[j]
            })
    
    return pd.DataFrame(output)


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    print("Loading kline data...")
    kline = pd.read_csv(os.path.join(data_dir, 'csi1000_kline_raw.csv'),
                        dtype={'stock_code': str})
    kline['date'] = pd.to_datetime(kline['date'])
    kline['stock_code'] = kline['stock_code'].str.zfill(6)
    print(f"Loaded {len(kline)} rows, {kline['stock_code'].nunique()} stocks")
    
    print("Computing Price Trend R² (window=60)...")
    factor_raw = compute_price_r2(kline, window=60)
    print(f"Raw: {len(factor_raw)} rows")
    
    r2_vals = factor_raw['r2']
    print(f"R² stats: mean={r2_vals.mean():.3f}, median={r2_vals.median():.3f}, "
          f"std={r2_vals.std():.3f}")
    
    # v1: R² (direction-agnostic smoothness)
    print("\n--- v1: R² (trend smoothness) ---")
    v1 = neutralize(factor_raw, kline, col='r2', window=60)
    v1.to_csv(os.path.join(data_dir, 'factor_price_r2_v1.csv'), index=False)
    print(f"Saved {len(v1)} rows")
    
    # v2: signed_r2 (direction + smoothness)
    print("\n--- v2: signed R² (trend direction × smoothness) ---")
    v2 = neutralize(factor_raw, kline, col='signed_r2', window=60)
    v2.to_csv(os.path.join(data_dir, 'factor_price_r2_v2.csv'), index=False)
    print(f"Saved {len(v2)} rows")
    
    # v3: negative R² (low smoothness = chaotic = reversal opportunity)
    print("\n--- v3: negative R² (reversal hypothesis) ---")
    v3 = v1.copy()
    v3['factor_value'] = -v3['factor_value']
    v3.to_csv(os.path.join(data_dir, 'factor_price_r2_v3_neg.csv'), index=False)
    print(f"Saved {len(v3)} rows")


if __name__ == '__main__':
    main()
