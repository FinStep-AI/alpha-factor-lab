"""
因子：ROE趋势斜率 (ROE Trend Slope)
ID: roe_trend_v1

逻辑：
  每只股票过去8个季度的ROE做线性回归，取斜率。
  斜率>0 → ROE持续改善 → 盈利能力在增长 → Growth/Quality

中性化：成交额OLS中性化 + MAD + z-score
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def rolling_slope(series, window=8):
    """Vectorized rolling linear regression slope."""
    from numpy.lib.stride_tricks import sliding_window_view
    vals = series.values.astype(float)
    x = np.arange(window, dtype=float)
    x_c = x - x.mean()
    x_d = np.sum(x_c ** 2)
    n = len(vals)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        y = vals[i - window + 1:i + 1]
        if np.isnan(y).sum() <= window * 0.3:
            mask = ~np.isnan(y)
            if mask.sum() >= window * 0.7:
                x_m = x_c[mask]
                y_m = y[mask] - np.nanmean(y)
                out[i] = np.sum(x_m * y_m) / x_d
    return out


def neutralize_vectorized(values, control):
    """OLS neutralize values by control, then MAD + z-score. Returns numpy array."""
    mask = np.isfinite(values) & np.isfinite(control)
    if mask.sum() < 30:
        return np.full_like(values, np.nan, dtype=float)
    v = values[mask]
    x = control[mask]
    X = np.column_stack([np.ones(len(x)), x])
    try:
        beta = np.linalg.lstsq(X, v, rcond=None)[0]
        r = v - X @ beta
        med = np.median(r)
        mad = np.median(np.abs(r - med))
        if mad < 1e-10:
            return np.full_like(values, np.nan, dtype=float)
        r_clipped = np.clip(r, med - 5.2 * mad, med + 5.2 * mad)
        std = r_clipped.std()
        if std < 1e-10:
            return np.full_like(values, np.nan, dtype=float)
        z = np.full(mask.sum(), np.nan, dtype=float)
        z[~np.isnan(r_clipped)] = (r_clipped[~np.isnan(r_clipped)] - np.median(r_clipped)) / std
        result = np.full_like(values, np.nan, dtype=float)
        result[mask] = z
        return result
    except:
        return np.full_like(values, np.nan, dtype=float)


def main():
    print("Loading fundamental data...")
    fund = pd.read_csv('data/csi1000_fundamental_cache.csv')
    fund['report_date'] = pd.to_datetime(fund['report_date'])

    q1, q99 = fund['roe'].quantile(0.01), fund['roe'].quantile(0.99)
    fund['roe'] = fund['roe'].clip(q1, q99)

    print(f"Fund: {len(fund)} rows, {fund['stock_code'].nunique()} stocks")

    fund = fund.sort_values(['stock_code', 'report_date']).reset_index(drop=True)
    print("Computing 8Q rolling ROE slope...")

    fund['raw_factor'] = fund.groupby('stock_code')['roe'].transform(
        lambda s: pd.Series(rolling_slope(s, 8), index=s.index)
    )

    slope_df = fund[['stock_code', 'report_date', 'raw_factor']].dropna(subset=['raw_factor']).copy()
    print(f"Slope rows: {len(slope_df)}")
    slope_df = slope_df.rename(columns={'report_date': 'date'})

    if slope_df.empty:
        print("No data!")
        return

    # Load kline for dates and market cap proxy
    print("Loading kline...")
    kline = pd.read_csv('data/csi1000_kline_raw.csv')
    kline['date'] = pd.to_datetime(kline['date'])
    trading_dates_array = np.array(sorted(kline['date'].unique()))
    date_to_idx = {d: i for i, d in enumerate(trading_dates_array)}

    kline['log_amount_20d'] = kline.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean().clip(lower=1))
    )

    # Vectorized FF: build date index array for all factor records
    print("Aligning to daily dates...")
    slope_df['date_idx'] = slope_df['date'].map(date_to_idx)
    slope_df = slope_df.dropna(subset=['date_idx'])
    slope_df['date_idx'] = slope_df['date_idx'].astype(int)

    stocks = sorted(slope_df['stock_code'].unique())
    ndates = len(trading_dates_array)

    # Use merge + groupby+ffill approach (vectorized forward fill)
    all_daily_dates = pd.DataFrame({'date': trading_dates_array})
    all_daily_dates['date'] = pd.to_datetime(all_daily_dates['date'])

    results = []
    print(f"FF over {len(stocks)} stocks...")
    for si, stock in enumerate(stocks):
        if si % 200 == 0:
            print(f"  {si}/{len(stocks)}")
        sf = slope_df[slope_df['stock_code'] == stock][['date', 'raw_factor']].copy()
        if sf.empty:
            continue
        sf = sf.set_index('date').sort_index()
        sf['raw_factor'] = sf['raw_factor'].reindex(trading_dates_array, method='ffill', limit=60)
        sf['stock_code'] = stock
        sf = sf.dropna(subset=['raw_factor'])
        results.append(sf.reset_index())

    if not results:
        print("No results!")
        return

    factor = pd.concat(results, ignore_index=True)
    print(f"After FF: {len(factor)}")

    # Merge with kline data for neutralization
    factor = factor.merge(
        kline[['date', 'stock_code', 'log_amount_20d']],
        on=['date', 'stock_code'], how='inner'
    )
    factor = factor.dropna(subset=['raw_factor', 'log_amount_20d'])
    print(f"After merge: {len(factor)}")

    # Vectorized cross-sectional neutralization
    print("Neutralizing per date...")
    results_list = []

    for date, grp in factor.groupby('date'):
        vals = grp['raw_factor'].values.astype(np.float64)
        ctrl = grp['log_amount_20d'].values.astype(np.float64)
        if len(vals) < 30:
            continue
        neutralized = neutralize_vectorized(vals, ctrl)
        good = np.isfinite(neutralized)
        if good.sum() == 0:
            continue
        out = grp.iloc[:good.sum()].copy()
        out['factor_value'] = neutralized[good]
        results_list.append(out[['date', 'stock_code', 'factor_value']])

    if not results_list:
        print("No neutralized results!")
        return

    output = pd.concat(results_list, ignore_index=True)
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')

    # Write with float64 dtype explicitly
    output.to_csv('data/factor_roe_trend_v1.csv', index=False, float_format='%.6f')
    print(f"\nDone: {len(output)} rows")
    print(f"Dates: {output['date'].min()} - {output['date'].max()}")
    print(f"Stats:\n{output['factor_value'].describe()}")


if __name__ == '__main__':
    main()
