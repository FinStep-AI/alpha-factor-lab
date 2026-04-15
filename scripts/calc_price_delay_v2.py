#!/usr/bin/env python3
"""
Price Delay Factor v2 - with rank transform and optimized parameters.
Tries: rank transform, different windows (40d, 80d), D2 (coefficient-weighted delay).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings, sys
warnings.filterwarnings('ignore')

def compute_delay_vectorized(returns_path, kline_path, output_path,
                             window=60, n_lags=5, min_obs=40, 
                             use_rank=True, delay_type='d1'):
    """
    delay_type:
      'd1' = 1 - R2_restricted/R2_unrestricted (original Hou-Moskowitz)
      'd2' = sum(|beta_lag_n * lag_n_R2_contrib|) / R2_unrestricted (coefficient-weighted)
    """
    print(f"Config: window={window}, lags={n_lags}, rank={use_rank}, type={delay_type}")
    
    returns = pd.read_csv(returns_path)
    kline = pd.read_csv(kline_path)
    returns['stock_code'] = returns['stock_code'].astype(str).str.zfill(6)
    kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
    
    # Market return
    mkt_ret = returns.groupby('date')['return'].mean()
    dates_sorted = sorted(returns['date'].unique())
    
    # Build market return array with lags
    mkt_arr = np.array([mkt_ret.get(d, np.nan) for d in dates_sorted])
    date_to_idx = {d: i for i, d in enumerate(dates_sorted)}
    
    results = []
    stocks = returns['stock_code'].unique()
    total = len(stocks)
    
    for si, stk in enumerate(stocks):
        if si % 200 == 0:
            print(f"  {si}/{total}...")
        
        stk_df = returns[returns['stock_code'] == stk].sort_values('date')
        stk_dates = stk_df['date'].values
        stk_rets = stk_df['return'].values
        stk_idxs = np.array([date_to_idx[d] for d in stk_dates])
        
        n = len(stk_df)
        
        for i in range(window - 1, n):
            start = i - window + 1
            end = i + 1
            
            y = stk_rets[start:end]
            idxs = stk_idxs[start:end]
            
            # Build X: mkt_return + lags
            X_cols = [mkt_arr[idxs]]
            for lag in range(1, n_lags + 1):
                lagged_idxs = idxs - lag
                valid_lag = lagged_idxs >= 0
                lag_vals = np.full(len(idxs), np.nan)
                lag_vals[valid_lag] = mkt_arr[lagged_idxs[valid_lag]]
                X_cols.append(lag_vals)
            
            X_all = np.column_stack(X_cols)  # (window, 1+n_lags)
            
            # Valid mask
            valid = ~(np.isnan(y) | np.any(np.isnan(X_all), axis=1))
            if valid.sum() < min_obs:
                continue
            
            y_v = y[valid]
            X_v = X_all[valid]
            ones = np.ones((len(y_v), 1))
            
            ss_tot = np.sum((y_v - y_v.mean()) ** 2)
            if ss_tot < 1e-20:
                continue
            
            try:
                # Unrestricted
                Xu = np.hstack([ones, X_v])
                beta_u = np.linalg.lstsq(Xu, y_v, rcond=None)[0]
                ss_res_u = np.sum((y_v - Xu @ beta_u) ** 2)
                r2_u = 1 - ss_res_u / ss_tot
                
                # Restricted
                Xr = np.hstack([ones, X_v[:, :1]])
                beta_r = np.linalg.lstsq(Xr, y_v, rcond=None)[0]
                ss_res_r = np.sum((y_v - Xr @ beta_r) ** 2)
                r2_r = 1 - ss_res_r / ss_tot
                
                r2_u = max(r2_u, 1e-10)
                r2_r = max(r2_r, 0.0)
                
                if delay_type == 'd1':
                    delay = 1 - r2_r / r2_u
                elif delay_type == 'd2':
                    # D2: sum of |t_lag_n * beta_lag_n| / sum of |t_all|
                    # Simplified: ratio of lag coefficients' contribution
                    # Use sum(|beta_lag_n|) / sum(|all betas excl intercept|)
                    betas_no_intercept = beta_u[1:]  # beta0, beta1, ..., beta_n_lags
                    total_abs = np.sum(np.abs(betas_no_intercept))
                    if total_abs < 1e-10:
                        delay = 0.0
                    else:
                        lag_abs = np.sum(np.abs(betas_no_intercept[1:]))  # only lag betas
                        delay = lag_abs / total_abs
                else:
                    delay = 1 - r2_r / r2_u
                
                delay = np.clip(delay, 0, 1)
                
                # We want NEGATIVE delay (low delay = high factor value)
                factor_val = -delay
                
                results.append({
                    'date': stk_dates[i],
                    'stock_code': stk,
                    'factor_value': factor_val
                })
            except:
                continue
    
    print(f"Got {len(results)} observations")
    factor_df = pd.DataFrame(results)
    
    if use_rank:
        print("Applying rank transform...")
        factor_df['factor_value'] = factor_df.groupby('date')['factor_value'].rank(pct=True)
        factor_df['factor_value'] = factor_df['factor_value'] - 0.5  # center
    
    # Neutralize by log_amount
    print("Neutralizing...")
    kline_amt = kline[['date', 'stock_code', 'amount']].sort_values(['stock_code', 'date'])
    kline_amt['log_amount_20d'] = kline_amt.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    
    factor_df = factor_df.merge(
        kline_amt[['date', 'stock_code', 'log_amount_20d']],
        on=['date', 'stock_code'], how='left'
    )
    
    def neutralize_date(group):
        y = group['factor_value'].values
        x = group['log_amount_20d'].values
        valid = ~(np.isnan(y) | np.isnan(x))
        if valid.sum() < 30:
            group['factor_neutral'] = np.nan
            return group
        y_v, x_v = y[valid], x[valid]
        X = np.column_stack([np.ones(len(x_v)), x_v])
        try:
            beta = np.linalg.lstsq(X, y_v, rcond=None)[0]
            resid = np.full(len(y), np.nan)
            resid[valid] = y_v - X @ beta
            group['factor_neutral'] = resid
        except:
            group['factor_neutral'] = np.nan
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(neutralize_date)
    
    # MAD winsorize + z-score
    def mad_zscore(group):
        vals = group['factor_neutral'].values
        valid = ~np.isnan(vals)
        if valid.sum() < 30:
            group['factor_final'] = np.nan
            return group
        v = vals[valid]
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        if mad < 1e-10:
            group['factor_final'] = 0.0
            return group
        upper = med + 5 * 1.4826 * mad
        lower = med - 5 * 1.4826 * mad
        v_clipped = np.clip(v, lower, upper)
        mean_v = v_clipped.mean()
        std_v = v_clipped.std()
        if std_v < 1e-10:
            group['factor_final'] = 0.0
            return group
        z = np.full(len(vals), np.nan)
        z[valid] = (v_clipped - mean_v) / std_v
        group['factor_final'] = z
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(mad_zscore)
    
    out = factor_df[['date', 'stock_code', 'factor_final']].rename(
        columns={'factor_final': 'factor_value'}
    ).dropna()
    
    out.to_csv(output_path, index=False)
    print(f"Saved: {output_path}, shape={out.shape}")

if __name__ == '__main__':
    base = Path(__file__).resolve().parent.parent
    
    # v2a: rank transform + D1 + 60d window
    print("=== v2a: D1 + rank + 60d ===")
    compute_delay_vectorized(
        str(base / 'data' / 'csi1000_returns.csv'),
        str(base / 'data' / 'csi1000_kline_raw.csv'),
        str(base / 'data' / 'factor_price_delay_v2a.csv'),
        window=60, n_lags=5, use_rank=True, delay_type='d1'
    )
