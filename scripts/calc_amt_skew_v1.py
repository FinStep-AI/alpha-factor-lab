"""
因子：成交额偏度 (Amount Skewness) v1
20日成交额分布偏度。用numpy向量化加速。
"""
import pandas as pd
import numpy as np
import os

def rolling_skew_fast(series, window=20, min_periods=15):
    """Vectorized rolling skewness using rolling moments."""
    arr = series.values.astype(float)
    n = len(arr)
    result = np.full(n, np.nan)
    
    for i in range(min_periods - 1, n):
        start = max(0, i - window + 1)
        w = arr[start:i+1]
        w = w[~np.isnan(w)]
        if len(w) < min_periods:
            continue
        m = np.mean(w)
        s = np.std(w, ddof=1)
        if s < 1e-10:
            result[i] = 0.0
            continue
        result[i] = np.mean(((w - m) / s) ** 3) * len(w) / (len(w)-1) * len(w) / max(len(w)-2, 1)
    return pd.Series(result, index=series.index)

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    print("Loading kline data...")
    df = pd.read_csv(os.path.join(data_dir, 'csi1000_kline_raw.csv'),
                     dtype={'stock_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    print(f"Loaded {len(df)} rows, {df['stock_code'].nunique()} stocks")
    
    # Use log(amount) for better distribution properties
    df['log_amount'] = np.log(df['amount'] + 1)
    
    # Compute 20d rolling skewness using pandas built-in (much faster than scipy apply)
    print("Computing 20d rolling amount skewness...")
    df['raw_factor'] = df.groupby('stock_code')['log_amount'].transform(
        lambda x: x.rolling(20, min_periods=15).skew()
    )
    
    # 20d rolling mean of amount for neutralization
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=15).mean() + 1)
    )
    
    df_valid = df.dropna(subset=['raw_factor', 'log_amount_20d']).copy()
    print(f"Valid rows: {len(df_valid)}")
    
    # Neutralize by amount (OLS per date)
    print("Neutralizing by amount...")
    result_parts = []
    for dt, grp in df_valid.groupby('date'):
        if len(grp) < 30:
            continue
        vals = grp['raw_factor'].values
        amt = grp['log_amount_20d'].values
        
        X = np.column_stack([np.ones(len(amt)), amt])
        try:
            beta = np.linalg.lstsq(X, vals, rcond=None)[0]
            resid = vals - X @ beta
        except:
            continue
        
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        if mad < 1e-10:
            continue
        resid = np.clip(resid, med - 5 * 1.4826 * mad, med + 5 * 1.4826 * mad)
        
        m = np.mean(resid)
        s = np.std(resid)
        if s < 1e-10:
            continue
        z = (resid - m) / s
        
        part = grp[['date', 'stock_code']].copy()
        part['factor_value'] = z
        result_parts.append(part)
    
    result = pd.concat(result_parts, ignore_index=True)
    result['date'] = result['date'].dt.strftime('%Y-%m-%d')
    
    out_path = os.path.join(data_dir, 'factor_amt_skew_v1.csv')
    result.to_csv(out_path, index=False)
    print(f"Saved to {out_path}, shape: {result.shape}")
    print(f"Date range: {result['date'].min()} ~ {result['date'].max()}")

if __name__ == '__main__':
    main()
