import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
FACTOR_ID = 'roe_confirm_capdisc_growth_v2'
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
fund['roe_lag1'] = g['roe'].shift(1)
fund['roe_lag4'] = g['roe'].shift(4)
fund['roe_lag5'] = g['roe'].shift(5)
fund['bps_lag1'] = g['bps'].shift(1)
fund['bps_lag4'] = g['bps'].shift(4)

fund['roe_yoy'] = fund['roe'] - fund['roe_lag4']
fund['roe_yoy_prev'] = fund['roe_lag1'] - fund['roe_lag5']
fund['roe_yoy_accel'] = fund['roe_yoy'] - fund['roe_yoy_prev']
fund['roe_qoq'] = fund['roe'] - fund['roe_lag1']
fund['bps_yoy'] = fund['bps'] / fund['bps_lag4'] - 1.0
fund['bps_qoq'] = fund['bps'] / fund['bps_lag1'] - 1.0
fund['roe_mean4'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).mean())
fund['roe_std4'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).std())
fund['roe_yoy_std4'] = g['roe_yoy'].transform(lambda s: s.rolling(4, min_periods=3).std())

# 更强调“同比改善已出现 + 环比仍在确认”，并惩罚资本膨胀快于ROE改善
improve = np.tanh(fund['roe_yoy'].fillna(0) / 4.0)
confirm = np.tanh(fund['roe_qoq'].fillna(0) / 2.0)
accel = np.tanh(fund['roe_yoy_accel'].fillna(0) / 2.5)
level = np.tanh(fund['roe_mean4'].fillna(0) / 7.5)
stability = 1.0 / (1.0 + 0.7 * fund['roe_std4'].abs().fillna(0) + 0.3 * fund['roe_yoy_std4'].abs().fillna(0))
capital_burst = np.maximum(fund['bps_yoy'].fillna(0), fund['bps_qoq'].fillna(0)).clip(lower=0)
capital_guard = 1.0 - np.tanh(2.8 * capital_burst)
match_guard = 1.0 - np.tanh(np.maximum(capital_burst - (fund['roe_yoy'].fillna(0) / 20.0), 0))

fund['raw_factor'] = (0.38 * improve + 0.26 * confirm + 0.18 * accel + 0.10 * level) * stability * capital_guard * match_guard
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
factor = factor.merge(kline[['date', 'stock_code', 'log_mktcap']], on=['date', 'stock_code'], how='inner')
factor = factor.dropna(subset=['raw_factor', 'log_mktcap'])

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
    clipped = np.clip(resid, med - 5.2 * mad, med + 5.2 * mad)
    std = clipped.std()
    if std < 1e-12:
        return np.full(len(vals), np.nan)
    z = (clipped - np.median(clipped)) / std
    out = np.full(len(vals), np.nan)
    out[np.where(mask)[0]] = z
    return out

out = []
for date, grp in factor.groupby('date'):
    nz = neutralize(grp['raw_factor'].values.astype(float), grp['log_mktcap'].values.astype(float))
    good = np.isfinite(nz)
    if good.sum() == 0:
        continue
    sub = grp.loc[good, ['date', 'stock_code']].copy()
    sub['factor'] = nz[good]
    out.append(sub)

result = pd.concat(out, ignore_index=True)
result['date'] = pd.to_datetime(result['date']).dt.strftime('%Y-%m-%d')
result.to_csv(OUT, index=False, float_format='%.6f')
print(f'saved {OUT} rows={len(result)} dates={result.date.min()}~{result.date.max()}')
