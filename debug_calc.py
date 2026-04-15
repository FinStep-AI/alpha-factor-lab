"""Debug version - print each step's row count."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def rolling_slope(series, window=8):
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

fund = pd.read_csv('data/csi1000_fundamental_cache.csv')
fund['report_date'] = pd.to_datetime(fund['report_date'])
fund = fund.sort_values(['stock_code', 'report_date']).reset_index(drop=True)
print(f"Fund shape: {fund.shape}")

fund['raw_factor'] = fund.groupby('stock_code')['roe'].transform(
    lambda s: pd.Series(rolling_slope(s, 8), index=s.index)
)

slope_df = fund[['stock_code', 'report_date', 'raw_factor']].dropna().copy()
slope_df = slope_df.rename(columns={'report_date': 'date'})
print(f"Slope rows: {len(slope_df)}")
print(f"Slope dates: {sorted(slope_df['date'].unique())}")

kline = pd.read_csv('data/csi1000_kline_raw.csv')
kline['date'] = pd.to_datetime(kline['date'])
trading_dates = sorted(kline['date'].unique())
print(f"Trading dates: {len(trading_dates)}")

date_to_idx = {d: i for i, d in enumerate(trading_dates)}
slope_df['date_idx'] = slope_df['date'].map(date_to_idx)
slope_df = slope_df.dropna(subset=['date_idx'])
slope_df['date_idx'] = slope_df['date_idx'].astype(int)
print(f"After date_idx filter: {len(slope_df)}")

all_daily = pd.DataFrame({'date': trading_dates})
stocks = sorted(slope_df['stock_code'].unique())
results = []

for si, stock in enumerate(stocks):
    sf = slope_df[slope_df['stock_code'] == stock][['date', 'raw_factor']].copy()
    sf = sf.set_index('date').sort_index()
    sf = sf[~sf.index.duplicated(keep='last')]
    sf_reindexed = sf.reindex(trading_dates, method='ffill', limit=60)
    n_nonnull = sf_reindexed['raw_factor'].notna().sum()
    sf_reindexed['stock_code'] = stock
    sf_clean = sf_reindexed.dropna(subset=['raw_factor'])
    if si < 1 or si % 200 == 0:
        print(f"Stock {stock}: FF={n_nonnull}, after_drop={len(sf_clean)}, date_range={sf_clean.index.min()}-{sf_clean.index.max()}")
    results.append(sf_clean.reset_index())

factor = pd.concat(results, ignore_index=True)
print(f"\nAfter FF: {len(factor)} rows, dates={sorted(factor['date'].unique())[:5]} to {sorted(factor['date'].unique())[-5:]}")

kline['log_amount_20d'] = kline.groupby('stock_code')['amount'].transform(
    lambda x: np.log(x.rolling(20, min_periods=10).mean().clip(lower=1))
)

factor = factor.merge(kline[['date', 'stock_code', 'log_amount_20d']],
                      on=['date', 'stock_code'], how='inner')
factor = factor.dropna(subset=['raw_factor', 'log_amount_20d'])
print(f"After kline merge: {len(factor)} rows")
print(f"Dates in merged: {sorted(factor['date'].unique())[:5]} to {sorted(factor['date'].unique())[-5:]}")
print(f"Unique dates: {factor['date'].nunique()}")

def neutralize(g):
    v = g['raw_factor'].values.astype(np.float64)
    x = g['log_amount_20d'].values.astype(np.float64)
    if len(v) < 30:
        print(f"  SKIP {g['date'].iloc[0]}: only {len(v)} stocks")
        g['factor_value'] = np.nan
        return g
    X = np.column_stack([np.ones(len(x)), x])
    try:
        b = np.linalg.lstsq(X, v, rcond=None)[0]
        r = v - X @ b
        med = np.median(r)
        mad = np.median(np.abs(r - med))
        if mad < 1e-10:
            g['factor_value'] = np.nan
            return g
        r = np.clip(r, med - 5.2*mad, med + 5.2*mad)
        r = r - np.median(r)
        std = r.std()
        if std < 1e-10:
            g['factor_value'] = np.nan
            return g
        z = r / std
        g['factor_value'] = z
        return g
    except Exception as e:
        print(f"  ERROR {g['date'].iloc[0]}: {e}")
        g['factor_value'] = np.nan
        return g

date_counts_before = factor.groupby('date').size()
print(f"\nDates before neutralize: {len(date_counts_before)}, min stocks={date_counts_before.min()}, max={date_counts_before.max()}")

factor = factor.groupby('date', group_keys=False).apply(neutralize)
date_counts_after = factor.groupby('date').size()
print(f"Dates after neutralize: {len(date_counts_after)}, min stocks={date_counts_after.min()}, max={date_counts_after.max()}")
print(f"Date list: {sorted(factor['date'].unique())}")

out = factor[['date', 'stock_code', 'factor_value']].dropna().copy()
out['date'] = out['date'].dt.strftime('%Y-%m-%d')
out.to_csv('data/factor_roe_trend_v1.csv', index=False, float_format='%.6f')
print(f"\nSaved {len(out)} rows")
