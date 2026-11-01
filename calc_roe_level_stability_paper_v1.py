import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path('.')
fund_path = ROOT / 'data/csi1000_fundamental_cache.csv'
kline_path = ROOT / 'data/csi1000_kline_raw.csv'
out_path = ROOT / 'data/factor_roe_level_stability_paper_v1.csv'

fund = pd.read_csv(fund_path)
fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
fund['report_date'] = pd.to_datetime(fund['report_date'])
fund = fund.sort_values(['stock_code', 'report_date']).copy()

g = fund.groupby('stock_code')
fund['roe_lag1'] = g['roe'].shift(1)
fund['roe_lag2'] = g['roe'].shift(2)
fund['roe_lag3'] = g['roe'].shift(3)
fund['roe_lag4'] = g['roe'].shift(4)
fund['roe_yoy'] = fund['roe'] - fund['roe_lag4']
fund['roe_ma4'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).mean())
fund['roe_std4'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).std())
fund['roe_smooth'] = (fund['roe'] + fund['roe_lag1'] + fund['roe_lag2'] + fund['roe_lag3']) / 4.0
fund['roe_vol_pen'] = fund['roe_std4'].clip(lower=0)

# paper spirit proxy: profitability persistence / smooth profitability
# reward high and smooth ROE, slight confirmation from YoY improvement, punish volatility
fund['raw_factor'] = (
    0.62 * np.tanh(fund['roe_ma4'] / 7.0)
    + 0.18 * np.tanh(fund['roe_smooth'] / 7.0)
    + 0.12 * np.tanh(fund['roe_yoy'].fillna(0) / 4.0)
    - 0.34 * np.tanh(fund['roe_vol_pen'].fillna(0) / 2.8)
)

fund = fund.dropna(subset=['raw_factor'])

kline = pd.read_csv(kline_path, usecols=['date', 'stock_code', 'close', 'amount', 'turnover'])
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline['date'] = pd.to_datetime(kline['date'])
kline['mkt_proxy'] = kline['close'].clip(lower=0.01) * kline['amount'].clip(lower=1) / (kline['turnover'].replace(0, np.nan) + 1e-6)
kline['log_mkt_proxy'] = np.log(kline['mkt_proxy'].clip(lower=1e6))
trade = kline[['date', 'stock_code', 'log_mkt_proxy']].sort_values(['stock_code', 'date'])

fund_map = fund[['stock_code', 'report_date', 'raw_factor']].sort_values(['stock_code', 'report_date'])
merged = pd.merge_asof(
    trade.sort_values('date'),
    fund_map.sort_values('report_date'),
    left_on='date',
    right_on='report_date',
    by='stock_code',
    direction='backward'
)
merged = merged.dropna(subset=['raw_factor', 'log_mkt_proxy']).copy()

res = []
for dt, gdf in merged.groupby('date', sort=True):
    x = gdf['log_mkt_proxy'].values.astype(float)
    y = gdf['raw_factor'].values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 50 or np.nanstd(x[mask]) < 1e-8 or np.nanstd(y[mask]) < 1e-8:
        continue
    X = np.column_stack([np.ones(mask.sum()), x[mask]])
    beta = np.linalg.lstsq(X, y[mask], rcond=None)[0]
    resid = y[mask] - X @ beta
    med = np.nanmedian(resid)
    mad = np.nanmedian(np.abs(resid - med))
    if not np.isfinite(mad) or mad < 1e-8:
        continue
    lo = med - 3 * 1.4826 * mad
    hi = med + 3 * 1.4826 * mad
    resid = np.clip(resid, lo, hi)
    std = np.nanstd(resid)
    if std < 1e-8:
        continue
    z = (resid - np.nanmean(resid)) / std
    tmp = gdf.loc[mask, ['date', 'stock_code']].copy()
    tmp['factor'] = z
    res.append(tmp)

out = pd.concat(res, ignore_index=True)
out['date'] = out['date'].dt.strftime('%Y-%m-%d')
out.to_csv(out_path, index=False)
print(f'saved {out_path} rows={len(out)} dates={out.date.min()}~{out.date.max()} stocks={out.stock_code.nunique()}')
