import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
FACTOR_ID = 'roe_improve_capital_efficiency_v1'
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
fund['bps_lag1'] = g['bps'].shift(1)
fund['bps_lag4'] = g['bps'].shift(4)
fund['roe_mean4'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).mean())
fund['roe_std4'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).std())
fund['bps_yoy_mean2'] = g['bps'].transform(lambda s: (s / s.shift(4) - 1.0).rolling(2, min_periods=2).mean())

fund['roe_yoy'] = fund['roe'] - fund['roe_lag4']
fund['roe_qoq'] = fund['roe'] - fund['roe_lag1']
fund['bps_yoy'] = fund['bps'] / fund['bps_lag4'] - 1.0
fund['bps_qoq'] = fund['bps'] / fund['bps_lag1'] - 1.0

# 核心：盈利改善效率 = ROE改善 / 权益扩张；
# 偏好 ROE 改善由经营效率驱动，而不是单纯扩表。
capital_base = 1.0 + 2.5 * fund['bps_yoy'].clip(lower=0).fillna(0) + 1.2 * fund['bps_qoq'].clip(lower=0).fillna(0)
roe_efficiency = fund['roe_yoy'].fillna(0) / capital_base
trend_confirm = np.tanh((0.65 * fund['roe_yoy'].fillna(0) + 0.35 * fund['roe_qoq'].fillna(0)) / 5.5)
level_gate = np.tanh(fund['roe_mean4'].fillna(0) / 8.0)
stability_gate = 1.0 / (1.0 + 0.9 * fund['roe_std4'].abs().fillna(0))
capital_discipline = 1.0 / (1.0 + 3.0 * fund['bps_yoy_mean2'].clip(lower=0).abs().fillna(0))

fund['raw_factor'] = np.tanh(roe_efficiency / 4.0) * (0.55 + 0.45 * level_gate) * stability_gate + 0.25 * trend_confirm * capital_discipline
fund = fund.replace([np.inf, -np.inf], np.nan).dropna(subset=['raw_factor'])
fund['avail_date'] = fund['report_date'] + pd.Timedelta(days=45)
factor_q = fund[['stock_code', 'avail_date', 'raw_factor']].rename(columns={'avail_date': 'date'})

kline['date'] = pd.to_datetime(kline['date'])
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline = kline.sort_values(['stock_code', 'date']).drop_duplicates(['date', 'stock_code'])
kline['mktcap_proxy'] = kline['close'].clip(lower=0.01) * kline['amount'].clip(lower=1) / (kline['turnover'].replace(0, np.nan) + 1e-6)
kline['log_mktcap'] = np.log(kline['mktcap_proxy'].clip(lower=1))
trade_dates = pd.Index(sorted(kline['date'].unique()))

res = []
for stock, grp in factor_q.groupby('stock_code'):
    sf = grp[['date', 'raw_factor']].drop_duplicates('date', keep='last').set_index('date').sort_index()
    sf = sf.reindex(trade_dates, method='ffill', limit=80)
    sf['stock_code'] = stock
    sf = sf.dropna(subset=['raw_factor']).reset_index().rename(columns={'index': 'date'})
    res.append(sf)

factor = pd.concat(res, ignore_index=True)
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
