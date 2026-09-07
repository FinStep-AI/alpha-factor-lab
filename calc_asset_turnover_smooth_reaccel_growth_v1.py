import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
FACTOR_ID = 'asset_turnover_smooth_reaccel_growth_v1'
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

g = fund.groupby('stock_code')
fund['asset_turnover_proxy'] = fund['roe'] / fund['bps'].abs().clip(lower=0.5)
fund['at_l1'] = g['asset_turnover_proxy'].shift(1)
fund['at_l2'] = g['asset_turnover_proxy'].shift(2)
fund['at_l4'] = g['asset_turnover_proxy'].shift(4)
fund['at_l5'] = g['asset_turnover_proxy'].shift(5)
fund['at_l8'] = g['asset_turnover_proxy'].shift(8)
fund['roe_l1'] = g['roe'].shift(1)
fund['roe_l4'] = g['roe'].shift(4)
fund['bps_l4'] = g['bps'].shift(4)

fund['at_yoy'] = fund['asset_turnover_proxy'] - fund['at_l4']
fund['at_prev_yoy'] = fund['at_l1'] - fund['at_l5']
fund['at_reaccel'] = fund['at_yoy'] - fund['at_prev_yoy']
fund['at_qoq'] = fund['asset_turnover_proxy'] - fund['at_l1']
fund['at_smooth'] = -(
    (fund['asset_turnover_proxy'] - 2 * fund['at_l1'] + fund['at_l2']).abs() +
    0.6 * fund['at_qoq'].sub(fund['at_l1'] - fund['at_l2']).abs()
)
fund['roe_yoy'] = fund['roe'] - fund['roe_l4']
fund['bps_yoy'] = fund['bps'] - fund['bps_l4']
fund['at_vol4'] = g['asset_turnover_proxy'].transform(lambda s: s.rolling(4, min_periods=3).std())
fund['roe_vol4'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).std())

improve = np.tanh(18.0 * fund['at_yoy'].fillna(0))
reaccel = np.tanh(22.0 * fund['at_reaccel'].fillna(0))
smooth = np.tanh(10.0 * fund['at_smooth'].fillna(0))
qoq = np.tanh(14.0 * fund['at_qoq'].fillna(0))
profit_confirm = np.tanh((0.7 * fund['roe_yoy'].fillna(0) + 0.3 * fund['bps_yoy'].fillna(0)) / 5.0)
stability = 1.0 / (1.0 + 5.0 * fund['at_vol4'].abs().fillna(0) + 0.12 * fund['roe_vol4'].abs().fillna(0))
low_level_penalty = 0.6 + 0.4 * np.tanh(fund['asset_turnover_proxy'].fillna(0) / 1.5)

fund['raw_factor'] = (
    0.32 * improve +
    0.24 * reaccel +
    0.18 * smooth +
    0.10 * qoq +
    0.16 * profit_confirm
) * stability * low_level_penalty

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
print(f'saved {OUT} rows={len(result)} dates={result.date.nunique()} stocks={result.stock_code.nunique()}')
