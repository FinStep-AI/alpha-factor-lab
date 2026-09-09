import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
FACTOR_ID = 'roe_bps_compound_smooth_quality_v1'
OUT = BASE / f'data/factor_{FACTOR_ID}.csv'

fund = pd.read_csv(BASE / 'data/csi1000_fundamental_cache.csv')
kline = pd.read_csv(BASE / 'data/csi1000_kline_raw.csv', usecols=['date', 'stock_code', 'close', 'amount', 'turnover'])

fund['report_date'] = pd.to_datetime(fund['report_date'])
fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
for col in ['roe', 'bps']:
    fund[col] = pd.to_numeric(fund[col], errors='coerce')
fund = fund.dropna(subset=['roe', 'bps']).sort_values(['stock_code', 'report_date']).drop_duplicates(['stock_code', 'report_date'], keep='last')

for col in ['roe', 'bps']:
    q01, q99 = fund[col].quantile([0.01, 0.99])
    fund[col] = fund[col].clip(q01, q99)

g = fund.groupby('stock_code')
fund['roe_l1'] = g['roe'].shift(1)
fund['roe_l2'] = g['roe'].shift(2)
fund['roe_l4'] = g['roe'].shift(4)
fund['bps_l1'] = g['bps'].shift(1)
fund['bps_l2'] = g['bps'].shift(2)
fund['bps_l4'] = g['bps'].shift(4)

fund['roe_yoy'] = fund['roe'] - fund['roe_l4']
fund['roe_qoq'] = fund['roe'] - fund['roe_l1']
fund['roe_accel'] = fund['roe_qoq'] - (fund['roe_l1'] - fund['roe_l2'])
fund['roe_mean4'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).mean())
fund['roe_std4'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).std())
fund['roe_smooth_penalty'] = (
    (fund['roe'] - 2 * fund['roe_l1'] + fund['roe_l2']).abs() +
    0.7 * fund['roe_accel'].abs()
)

fund['bps_yoy'] = fund['bps'] / fund['bps_l4'] - 1.0
fund['bps_qoq'] = fund['bps'] / fund['bps_l1'] - 1.0
fund['bps_roll_mean2'] = g['bps_yoy'].transform(lambda s: s.rolling(2, min_periods=2).mean())
fund['bps_roll_std3'] = g['bps_yoy'].transform(lambda s: s.rolling(3, min_periods=2).std())
fund['bps_compound_proxy'] = (1.0 + fund['bps_qoq'].clip(lower=-0.25, upper=0.35)).pow(4) - 1.0

roe_trend = np.tanh((0.75 * fund['roe_yoy'].fillna(0) + 0.25 * fund['roe_qoq'].fillna(0)) / 4.8)
roe_level = np.tanh(fund['roe_mean4'].fillna(0) / 8.5)
roe_stability = 1.0 / (1.0 + 0.50 * fund['roe_std4'].abs().fillna(0) + 0.22 * fund['roe_smooth_penalty'].fillna(0))
roe_accel_guard = 1.0 / (1.0 + 0.22 * fund['roe_accel'].clip(lower=0).fillna(0).abs())

bps_growth = np.tanh(1.6 * (0.65 * fund['bps_yoy'].clip(lower=-0.35, upper=1.2).fillna(0) + 0.35 * fund['bps_qoq'].clip(lower=-0.15, upper=0.30).fillna(0)))
bps_compound = np.tanh(1.2 * fund['bps_compound_proxy'].clip(lower=-0.35, upper=1.0).fillna(0))
bps_stability = 1.0 / (1.0 + 3.0 * fund['bps_roll_std3'].abs().fillna(0))
over_expand_penalty = 1.0 / (1.0 + 2.8 * fund['bps_roll_mean2'].clip(lower=0.30).sub(0.30).clip(lower=0).fillna(0))

fund['raw_factor'] = (
    (0.38 * roe_trend + 0.14 * roe_level + 0.18 * bps_growth + 0.16 * bps_compound) *
    roe_stability * bps_stability * roe_accel_guard * over_expand_penalty
)

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
factor = factor.merge(kline[['date', 'stock_code', 'log_mktcap']], on=['date', 'stock_code'], how='inner').dropna(subset=['raw_factor', 'log_mktcap'])

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
    out = np.full(len(vals), np.nan)
    out[np.where(mask)[0]] = (resid - np.median(resid)) / std
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
print(f'saved {OUT} rows={len(result)} dates={result.date.nunique()} stocks={result.stock_code.nunique()}')
