"""
因子：Close Location Value 自相关 (CLV Autocorrelation)
ID: clv_autocorr_v1

逻辑：
  CLV = (close - low) / (high - low), 反映日内收盘相对强度。
  计算过去20日CLV的Spearman自相关(滞后1阶)。
  
  高自相关 = 持续收在相同相对位置 = 日内买卖力量持续一致。
  可能反映知情交易者持续活动 / 趋势持续性。
  
  注意：高CLV自相关也可能是趋势股票的特征，所以加市值中性化。

Barra风格: MICRO (日内执行持续性)
计算：neutralize(rolling_spearman_acorr(clv, 20), log_amount)
中性化：成交额OLS + MAD + z-score
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def rolling_spearman_acorr(series, window=20, lag=1):
    """Rolling Spearman autocorrelation at given lag."""
    vals = series.values.astype(float)
    n = len(vals)
    out = np.full(n, np.nan, dtype=float)
    
    for i in range(window - 1, n):
        y = vals[max(0, i - window + 1):i + 1]
        key = np.isfinite(y)
        y_clean = y[key]
        if len(y_clean) < 12:
            continue
        try:
            if len(y_clean) > lag:
                r, _ = stats.spearmanr(y_clean[:-lag], y_clean[lag:])
                if np.isfinite(r):
                    out[i] = r
        except Exception:
            pass
    return out


def neutralize(values, control):
    """OLS neutralize, MAD winsorize, z-score."""
    combined = pd.DataFrame({'v': values, 'c': control})
    combined = combined.dropna()
    if len(combined) < 30:
        return np.full_like(values, np.nan)
    
    v = combined['v'].values
    x = combined['c'].values
    
    X = np.column_stack([np.ones(len(x)), x])
    try:
        beta = np.linalg.lstsq(X, v, rcond=None)[0]
        r = v - X @ beta
    except Exception:
        return np.full_like(values, np.nan)
    
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    if mad < 1e-10:
        return np.full_like(values, np.nan)
    r = np.clip(r, med - 5.2 * mad, med + 5.2 * mad)
    std = r.std()
    if std < 1e-10:
        return np.full_like(values, np.nan)
    
    result = np.full_like(values, np.nan)
    result[combined.index.values] = (r - np.median(r)) / std
    return result


def main():
    print("Loading data...")
    kline = pd.read_csv('data/csi1000_kline_raw.csv')
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # Compute CLV
    print("Computing CLV...")
    price_range = kline['high'] - kline['low']
    price_range = price_range.replace(0, np.nan)
    kline['clv'] = (kline['close'] - kline['low']) / price_range
    kline['clv'] = kline['clv'].clip(0, 1)  # CLV is always in [0,1]
    
    # Rolling autocorrelation per stock
    print("Computing 20-day rolling CLV autocorrelation...")
    
    results = []
    stock_count = 0
    for code, grp in kline.groupby('stock_code'):
        stock_count += 1
        if stock_count % 200 == 0:
            print(f"  Processed {stock_count}/1000 stocks...")
        
        grp = grp.sort_values('date').reset_index(drop=True)
        acorr = rolling_spearman_acorr(grp['clv'], window=20, lag=1)
        
        grp_out = grp[['stock_code', 'date', 'amount']].copy()
        grp_out['factor_raw'] = acorr
        results.append(grp_out)
    
    combined = pd.concat(results, ignore_index=True)
    print(f"Combined: {combined.shape}, raw non-null: {combined['factor_raw'].notna().sum()}")
    
    # Neutralize cross-sectionally per date
    print("Neutralizing cross-sectionally...")
    
    all_out = []
    date_count = 0
    for date, grp in combined.groupby('date'):
        date_count += 1
        if date_count % 100 == 0:
            print(f"  Date {date_count}/{combined['date'].nunique()}...")
        
        grp = grp.dropna(subset=['factor_raw']).copy()
        if len(grp) < 50:
            continue
        
        log_amount = np.log(grp['amount'].clip(lower=1))
        neutralized = neutralize(grp['factor_raw'].values, log_amount.values)
        
        grp = grp.copy()
        grp['factor_value'] = neutralized
        grp = grp.dropna(subset=['factor_value'])
        all_out.append(grp[['stock_code', 'date', 'factor_value']])
    
    result = pd.concat(all_out, ignore_index=True)
    print(f"\nFinal factor shape: {result.shape}")
    print(f"  Non-null: {result['factor_value'].notna().sum()}")
    print(f"  Mean: {result['factor_value'].mean():.4f}")
    print(f"  Std: {result['factor_value'].std():.4f}")
    print(f"  Dates: {result['date'].nunique()}")
    
    # Save
    output_path = 'data/factor_clv_autocorr_v1.csv'
    result.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    # Save a metadata summary
    print("\nDone!")


if __name__ == '__main__':
    main()
