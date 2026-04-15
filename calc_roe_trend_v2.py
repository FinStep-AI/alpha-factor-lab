"""
因子：ROE趋势斜率 sign版 v2
ID: roe_trend_v2

基于v1发现：负ROE斜率值(ROE下降)的股票反而收益更高
(This is counter-intuitive but is the actual alpha signal in CSI1000)

改进：
- 缩短回看至6Q（增加数据窗口，从300天→~540天）
- 保持方向信息（用斜率符号，不用绝对值）

中性化：成交额OLS中性化 + MAD + z-score
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def rolling_slope_signed(series, window=6):
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
                out[i] = np.sum(x_m * y_m) / x_d  # signed (not abs)
    return out

def neutralize_vectorized(values, control):
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
        r_clipped = np.clip(r, med - 5.2*mad, med + 5.2*mad)
        std = r_clipped.std()
        if std < 1e-10:
            return np.full_like(values, np.nan, dtype=float)
        z = np.full(mask.sum(), np.nan, dtype=float)
        valid_r = r_clipped[~np.isnan(r_clipped)]
        if len(valid_r) > 0:
            z[~np.isnan(r_clipped)] = (valid_r - np.median(valid_r)) / std
        result = np.full_like(values, np.nan, dtype=float)
        result[mask] = z
        return result
    except:
        return np.full_like(values, np.nan, dtype=float)

def main():
    print("Loading data...")
    fund = pd.read_csv('data/csi1000_fundamental_cache.csv')
    fund['report_date'] = pd.to_datetime(fund['report_date'])
    q1, q99 = fund['roe'].quantile(0.01), fund['roe'].quantile(0.99)
    fund['roe'] = fund['roe'].clip(q1, q99)
    fund = fund.sort_values(['stock_code', 'report_date']).reset_index(drop=True)

    print("Computing 6Q signed ROC slope...")
    fund['raw_factor'] = fund.groupby('stock_code')['roe'].transform(
        lambda s: pd.Series(rolling_slope_signed(s, 6), index=s.index)
    )

    slope_df = fund[['stock_code', 'report_date', 'raw_factor']].dropna().copy()
    slope_df = slope_df.rename(columns={'report_date': 'date'})
    print(f"Slope rows: {len(slope_df)}, dates: {sorted(slope_df['date'].unique())[:4]}...{sorted(slope_df['date'].unique())[-1]}, stocks: {slope_df['stock_code'].nunique()}")

    kline = pd.read_csv('data/csi1000_kline_raw.csv')
    kline['date'] = pd.to_datetime(kline['date'])
    trading_dates = sorted(kline['date'].unique())
    all_daily = pd.DataFrame({'date': trading_dates})

    print("FF to daily...")
    stocks = sorted(slope_df['stock_code'].unique())
    results = []
    for si, stock in enumerate(stocks):
        if si % 200 == 0:
            print(f"  {si}/{len(stocks)}...")
        sf = slope_df[slope_df['stock_code'] == stock][['date', 'raw_factor']].copy()
        sf = sf.set_index('date').sort_index()
        sf = sf[~sf.index.duplicated(keep='last')]
        sf_reindexed = sf.reindex(trading_dates, method='ffill', limit=60)
        sf_reindexed['stock_code'] = stock
        sf = sf_reindexed.dropna(subset=['raw_factor']).reset_index()
        results.append(sf)

    factor = pd.concat(results, ignore_index=True)
    print(f"After FF: {len(factor)} rows")

    kline['log_amount_20d'] = kline.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean().clip(lower=1))
    )
    factor = factor.merge(kline[['date', 'stock_code', 'log_amount_20d']],
                          on=['date', 'stock_code'], how='inner')
    factor = factor.dropna(subset=['raw_factor', 'log_amount_20d'])
    print(f"After merge: {len(factor)}")

    print("Neutralizing...")
    out_list = []
    for date, grp in factor.groupby('date'):
        vals = grp['raw_factor'].values.astype(np.float64)
        ctrl = grp['log_amount_20d'].values.astype(np.float64)
        if len(vals) < 30:
            continue
        neutralized = neutralize_vectorized(vals, ctrl)
        good = np.isfinite(neutralized)
        if good.sum() == 0:
            continue
        sub = grp.iloc[:good.sum()].copy()
        sub['factor_value'] = neutralized[good]
        out_list.append(sub[['date', 'stock_code', 'factor_value']])

    if not out_list:
        print("No results!")
        return

    output = pd.concat(out_list, ignore_index=True)
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')
    output.to_csv('data/factor_roe_trend_v2.csv', index=False, float_format='%.6f')
    print(f"\nDone: {len(output)} rows, dates: {output['date'].min()} to {output['date'].max()}")
    print(f"Stats: mean={output['factor_value'].mean():.4f}, std={output['factor_value'].std():.4f}")
    print(output.groupby('date').size().describe())

if __name__ == '__main__':
    main()
