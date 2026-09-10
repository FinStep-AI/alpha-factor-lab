import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
FACTOR_ID = 'asset_turnover_inflect_guard_v1'
OUT = BASE / f'data/factor_{FACTOR_ID}.csv'

fund = pd.read_csv(BASE / 'data/csi1000_fundamental_cache.csv')
kline = pd.read_csv(BASE / 'data/csi1000_kline_raw.csv', usecols=['date', 'stock_code', 'close', 'amount', 'turnover'])

fund['report_date'] = pd.to_datetime(fund['report_date'])
fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
for col in ['roe', 'bps']:
    fund[col] = pd.to_numeric(fund[col], errors='coerce')
fund = fund.dropna(subset=['roe', 'bps']).sort_values(['stock_code', 'report_date']).drop_duplicates(['stock_code', 'report_date'])

for col in ['roe', 'bps']:
    q01, q99 = fund[col].quantile([0.01, 0.99])
    fund[col] = fund[col].clip(q01, q99)

fund['at'] = fund['roe'] / fund['bps'].clip(lower=0.5)
g = fund.groupby('stock_code')
fund['at_l1'] = g['at'].shift(1)
fund['at_l4'] = g['at'].shift(4)
fund['at_l8'] = g['at'].shift(8)
fund['roe_l1'] = g['roe'].shift(1)
fund['roe_l4'] = g['roe'].shift(4)
fund['bps_l1'] = g['bps'].shift(1)
fund['bps_l4'] = g['bps'].shift(4)

fund['at_yoy'] = fund['at'] - fund['at_l4']
fund['at_qoq'] = fund['at'] - fund['at_l1']
fund['at_prev_yoy'] = fund['at_l4'] - fund['at_l8']
fund['at_inflect'] = fund['at_yoy'] - 0.65 * fund['at_prev_yoy']
fund['roe_yoy'] = fund['roe'] - fund['roe_l4']
fund['roe_qoq'] = fund['roe'] - fund['roe_l1']
fund['bps_yoy'] = fund['bps'] / fund['bps_l4'].replace(0, np.nan) - 1
fund['bps_qoq'] = fund['bps'] / fund['bps_l1'].replace(0, np.nan) - 1
fund['roe_std4'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).std())
fund['at_std4'] = g['at'].transform(lambda s: s.rolling(4, min_periods=3).std())

# robust clipping on noisy accounting proxy
for col in ['at', 'at_yoy', 'at_qoq', 'at_prev_yoy', 'at_inflect', 'roe_yoy', 'roe_qoq', 'bps_yoy', 'bps_qoq']:
    s = fund[col]
    q01, q99 = s.quantile([0.01, 0.99])
    fund[col] = s.clip(q01, q99)

inflect_core = np.tanh(1.8 * fund['at_inflect'].fillna(0))
trend_core = np.tanh(1.2 * fund['at_yoy'].fillna(0))
qoq_confirm = np.tanh(1.6 * fund['at_qoq'].fillna(0))
profit_confirm = np.tanh((0.75 * fund['roe_yoy'].fillna(0) + 0.25 * fund['roe_qoq'].fillna(0)) / 5.5)
capital_guard = 1.0 / (1.0 + 1.8 * fund['bps_yoy'].clip(lower=0).fillna(0).abs() + 0.6 * fund['bps_qoq'].clip(lower=0).fillna(0).abs())
stability = 1.0 / (1.0 + 0.20 * fund['roe_std4'].abs().fillna(0) + 0.75 * fund['at_std4'].abs().fillna(0))
low_roe_guard = 0.55 + 0.45 * np.tanh(fund['roe'].fillna(0) / 8.0)

fund['raw_factor'] = (
    0.36 * inflect_core +
    0.24 * trend_core +
    0.18 * qoq_confirm +
    0.22 * profit_confirm
) * capital_guard * stability * low_roe_guard

fund = fund.replace([np.inf, -np.inf], np.nan).dropna(subset=['raw_factor'])
fund['avail_date'] = fund['report_date'] + pd.Timedelta(days=45)
factor_q = fund[['stock_code', 'avail_date', 'raw_factor']].rename(columns={'avail_date': 'date'})

kline['date'] = pd.to_datetime(kline['date'])
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline = kline.sort_values(['stock_code', 'date']).drop_duplicates(['date', 'stock_code'])
kline['mktcap_proxy'] = kline['close'].clip(lower=0.01) * kline['amount'].clip(lower=1) / (kline['turnover'].replace(0, np.nan) + 1e-6)
kline['log_mktcap'] = np.log(kline['mktcap_proxy'].clip(lower=1))
trade_dates = pd.Index(sorted(kline['date'].unique()))

parts = []
for stock, grp in factor_q.groupby('stock_code'):
    sf = grp[['date', 'raw_factor']].drop_duplicates('date', keep='last').set_index('date').sort_index()
    sf = sf.reindex(trade_dates, method='ffill', limit=80)
    sf['stock_code'] = stock
    sf = sf.dropna(subset=['raw_factor']).reset_index().rename(columns={'index': 'date'})
    parts.append(sf)

factor = pd.concat(parts, ignore_index=True)
factor = factor.merge(kline[['date', 'stock_code', 'log_mktcap']], on=['date', 'stock_code'], how='inner').dropna()

def neutralize(vals, ctrl):
    mask = np.isfinite(vals) & np.isfinite(ctrl)
    if mask.sum() < 30:
        return np.full(len(vals), np.nan)
    y = vals[mask]
    x = ctrl[mask]
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    if mad < 1e-12:
        return np.full(len(vals), np.nan)
    resid = np.clip(resid, med - 5.0 * mad, med + 5.0 * mad)
    std = resid.std()
    if std < 1e-12:
        return np.full(len(vals), np.nan)
    z = (resid - np.median(resid)) / std
    out = np.full(len(vals), np.nan)
    out[np.where(mask)[0]] = z
    return out

outs = []
for date, grp in factor.groupby('date'):
    nz = neutralize(grp['raw_factor'].to_numpy(float), grp['log_mktcap'].to_numpy(float))
    good = np.isfinite(nz)
    if good.any():
        sub = grp.loc[good, ['date', 'stock_code']].copy()
        sub['factor'] = nz[good]
        outs.append(sub)

result = pd.concat(outs, ignore_index=True)
result['date'] = pd.to_datetime(result['date']).dt.strftime('%Y-%m-%d')
result.to_csv(OUT, index=False, float_format='%.6f')
print(f'saved {OUT} rows={len(result)} dates={result["date"].nunique()} stocks={result["stock_code"].nunique()}')
