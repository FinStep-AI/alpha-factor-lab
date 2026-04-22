"""
因子：日内方向自相关 (Intraday Direction Autocorrelation, IDA)
ID: intraday_direction_autocorr_v1

逻辑：日内方向 = (close - (high+low)/2) / (high - low)，衡量买家/卖方日内相对强度。
计算过去40日的自相关（相邻日方向是否连续相同）。高自相关 = 持续相同的订单流压力 = 
知情交易者持续介入信号。

假设：日内方向持续一致的股票（自相关高），反映知情交易者在持续建仓/出货，
后续应有 trend continuation alpha。

Barra风格: MICRO (订单流连续性)
计算：neutralize(autocorr(cti_ratio, 40), log_amount)
中性化：成交额OLS + MAD + zscore
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def compute_cti_ratio(df_kline):
    """Compute Close-to-Mid ratio = (close - (high+low)/2) / (high - low)."""
    mid = (df_kline['high'] + df_kline['low']) / 2.0
    price_range = df_kline['high'] - df_kline['low']
    price_range = price_range.replace(0, np.nan)
    cti = (df_kline['close'] - mid) / price_range
    # Clamp to [-0.5, 0.5] since close is bounded by high/low
    cti = cti.clip(-0.5, 0.5)
    return cti


def rolling_autocorr_40(series, window=40):
    """Scaled rolling autocorrelation at lag 1."""
    vals = series.values.astype(float)
    n = len(vals)
    out = np.full(n, np.nan)
    half_w = window // 2
    for i in range(half_w, n):
        y = vals[max(0, i-half_w):min(n, i+half_w)]
        if len(y) < 20:
            continue
        key = np.isfinite(y)
        y = y[key]
        if len(y) < 15:
            continue
        # Use robust rank autocorrelation (Spearman-like)
        if len(y) > 1:
            try:
                # Spearman autocorrelation equivalent: corr(y[:-1], y[1:])
                if len(y) > 3:
                    r, _ = stats.spearmanr(y[:-1], y[1:])
                    out[i] = r if np.isfinite(r) else np.nan
            except Exception:
                pass
    return out


def neutralize(values, control):
    """OLS neutralize, MAD winsorize, z-score."""
    mask = np.isfinite(values) & np.isfinite(control) & np.isfinite(control)
    if mask.sum() < 30:
        return np.full_like(values, np.nan)
    v = values[mask]
    x = control[mask]
    X = np.column_stack([np.ones(len(x)), x])
    try:
        beta = np.linalg.lstsq(X, v, rcond=None)[0]
        r = v - X @ beta
        med = np.median(r)
        mad = np.median(np.abs(r - med))
        if mad < 1e-10:
            return np.full_like(values, np.nan)
        r = np.clip(r, med - 5.2 * mad, med + 5.2 * mad)
        std = r.std()
        if std < 1e-10:
            return np.full_like(values, np.nan)
        result = np.full(mask.sum(), np.nan)
        valid = np.isfinite(r)
        result[valid] = (r[valid] - np.median(r[valid])) / std
        out = np.full_like(values, np.nan)
        out[mask] = result
        return out
    except Exception:
        return np.full_like(values, np.nan)


def main():
    # Step 1: Load kline data
    kline = pd.read_csv('data/csi1000_kline_raw.csv')
    print(f"Loaded kline: {kline.shape}, stocks: {kline['stock_code'].nunique()}")
    
    # Step 2: Compute CTI ratio per stock
    print("Computing CTI ratio...")
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # Calculate mid and range
    mid = (kline['high'] + kline['low']) / 2.0
    price_range = kline['high'] - kline['low']
    price_range = price_range.replace(0, np.nan)
    kline['cti_ratio'] = (kline['close'] - mid) / price_range
    kline['cti_ratio'] = kline['cti_ratio'].clip(-0.5, 0.5)
    
    # Step 3: Rolling autocorrelation of CTI ratio
    print("Computing 40-day rolling autocorrelation of CTI ratio...")
    
    results = []
    for code, grp in kline.groupby('stock_code'):
        grp = grp.sort_values('date').reset_index(drop=True)
        cti = grp['cti_ratio']
        
        # Rolling Spearman autocorr (window=40)
        window = 40
        acorr_vals = np.full(len(grp), np.nan)
        
        for i in range(window-1, len(grp)):
            window_vals = cti.iloc[i-window+1:i+1].values
            key = np.isfinite(window_vals)
            if key.sum() < 25:
                continue
            
            clean = window_vals[key]
            try:
                # Spearman correlation between adjacent values
                if len(clean) > 5:
                    r, _ = stats.spearmanr(clean[:-1], clean[1:])
                    if np.isfinite(r):
                        acorr_vals[i] = r
            except Exception:
                pass
        
        grp = grp.copy()
        grp['factor_raw'] = acorr_vals
        results.append(grp[['stock_code', 'date', 'factor_raw']])
    
    combined = pd.concat(results, ignore_index=True)
    print(f"Raw factor computed: {combined.shape}, non-null: {combined['factor_raw'].notna().sum()}")
    
    # Step 4: Neutralize per date (cross-sectional)
    print("Neutralizing per date...")
    
    neutralized = []
    for date, grp in combined.groupby('date'):
        grp = grp.dropna(subset=['factor_raw']).copy()
        if len(grp) < 50:
            continue
        
        # Use log_amount as neutralize variable
        # Need to merge amount
        kline_date = kline[kline['date'] == date][['stock_code', 'amount']].copy()
        grp = grp.merge(kline_date, on='stock_code', how='left')
        
        # Prepare control variable
        log_amount = np.log(grp['amount'].clip(lower=1))
        
        vals = grp['factor_raw'].values
        mask = np.isfinite(vals) & np.isfinite(log_amount.values)
        if mask.sum() < 30:
            continue
        
        v = vals[mask]
        x = log_amount.values[mask]
        
        # OLS neutralize
        X = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(X, v, rcond=None)[0]
            r = v - X @ beta
        except:
            continue
        
        # MAD winsorize
        med = np.median(r)
        mad = np.median(np.abs(r - med))
        if mad < 1e-10:
            continue
        r = np.clip(r, med - 5.2*mad, med + 5.2*mad)
        
        # z-score
        std = r.std()
        if std < 1e-10:
            continue
        z = (r - np.median(r)) / std
        
        out = grp.iloc[np.where(mask)[0]][['stock_code', 'date']].copy()
        out['factor_value'] = z
        neutralized.append(out)
    
    result = pd.concat(neutralized, ignore_index=True)
    print(f"Final factor: {result.shape}")
    print(result.tail(10))
    
    # Save
    result.to_csv('data/factor_intraday_direction_autocorr_v1.csv', index=False)
    print("Saved to data/factor_intraday_direction_autocorr_v1.csv")
    
    # Quick stats
    print(f"\nFactor stats:")
    print(f"  Non-null: {result['factor_value'].notna().sum()}")
    print(f"  Mean: {result['factor_value'].mean():.4f}")
    print(f"  Std: {result['factor_value'].std():.4f}")
    print(f"  Dates: {result['date'].nunique()}")


if __name__ == '__main__':
    main()
