import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
FACTOR_ID = 'asset_turnover_reaccel_expansion_gap_growth_v1'
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
fund['at_proxy'] = fund['roe'] / fund['bps'].abs().clip(lower=0.5)
fund['at_l1'] = g['at_proxy'].shift(1)
fund['at_l2'] = g['at_proxy'].shift(2)
fund['at_l4'] = g['at_proxy'].shift(4)
fund['at_l5'] = g['at_proxy'].shift(5)
fund['roe_l1'] = g['roe'].shift(1)
fund['roe_l4'] = g['roe'].shift(4)
fund['bps_l1'] = g['bps'].shift(1)
fund['bps_l4'] = g['bps'].shift(4)

fund['at_yoy'] = fund['at_proxy'] - fund['at_l4']
fund['at_prev_yoy'] = fund['at_l1'] - fund['at_l5']
fund['at_reaccel'] = fund['at_yoy'] - fund['at_prev_yoy']
fund['at_qoq'] = fund['at_proxy'] - fund['at_l1']
fund['at_qoq_inflect'] = fund['at_qoq'] - (fund['at_l1'] - fund['at_l2'])
fund['roe_yoy'] = fund['roe'] - fund['roe_l4']
fund['roe_qoq'] = fund['roe'] - fund['roe_l1']
fund['bps_yoy'] = fund['bps'] / fund['bps_l4'].replace(0, np.nan) - 1.0
fund['bps_qoq'] = fund['bps'] / fund['bps_l1'].replace(0, np.nan) - 1.0
fund['at_std4'] = g['at_proxy'].transform(lambda s: s.rolling(4, min_periods=3).std())
fund['roe_std4'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).std())
fund['bps_growth_smooth'] = g['bps_yoy'].transform(lambda s: s.rolling(2, min_periods=2).mean())

reaccel = np.tanh(28.0 * fund['at_reaccel'].fillna(0))
level_improve = np.tanh(20.0 * fund['at_yoy'].fillna(0))
inflect = np.tanh(18.0 * fund['at_qoq_inflect'].fillna(0))
profit_confirm = np.tanh((0.7 * fund['roe_yoy'].fillna(0) + 0.3 * fund['roe_qoq'].fillna(0)) / 4.0)
expansion_penalty = 1.0 / (1.0 + 10.0 * fund['bps_yoy'].clip(lower=0).fillna(0).abs() + 5.0 * fund['bps_qoq'].clip(lower=0).fillna(0).abs())
controlled_growth_bonus = 1.0 + 0.18 * np.tanh((fund['bps_growth_smooth'].fillna(0) - 0.03) / 0.08)
stability = 1.0 / (1.0 + 6.0 * fund['at_std4'].abs().fillna(0) + 0.18 * fund['roe_std4'].abs().fillna(0))
quality_gate = 0.55 + 0.45 * np.tanh(fund['roe'].fillna(0) / 7.0)

fund['raw_factor'] = (
    0.34 * reaccel +
    0.24 * level_improve +
    0.18 * inflect +
    0.24 * profit_confirm
) * expansion_penalty * controlled_growth_bonus * stability * quality_gate

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
print(f'saved {OUT} rows={len(result)} dates={result.date.min()}~{result.date.max()} stocks={result.stock_code.nunique()}')
