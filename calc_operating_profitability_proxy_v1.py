#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""operating_profitability_proxy_v1

论文语义：Fama-French 5-factor / profitability-investment 主线。
在当前 A 股数据仅有 ROE / BPS 的限制下，构造一个“高盈利水平 + 盈利改善 - 盈利波动 - 净资产扩张”的本土化代理。

raw = 0.55*tanh(MA4(ROE)/8)
    + 0.35*tanh((ROE_t-ROE_t-4)/6)
    - 0.25*tanh(STD4(ROE)/4)
    - 0.20*tanh((BPS_t/BPS_t-4 - 1)/0.25)

映射：report_date + 45 天
中性化：对 20 日 log(amount) 做截面 OLS 中性化
标准化：MAD 缩尾 + z-score
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
FACTOR_ID = 'operating_profitability_proxy_v1'
OUT = BASE / f'data/factor_{FACTOR_ID}.csv'

fund = pd.read_csv(BASE / 'data/csi1000_fundamental_cache.csv')
k = pd.read_csv(BASE / 'data/csi1000_kline_raw.csv', usecols=['date', 'stock_code', 'amount'])

fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
fund['report_date'] = pd.to_datetime(fund['report_date'])
for col in ['roe', 'bps']:
    fund[col] = pd.to_numeric(fund[col], errors='coerce')
fund = fund.dropna(subset=['roe', 'bps']).sort_values(['stock_code', 'report_date']).drop_duplicates(['stock_code', 'report_date'])

for col in ['roe', 'bps']:
    q01, q99 = fund[col].quantile([0.01, 0.99])
    fund[col] = fund[col].clip(q01, q99)

g = fund.groupby('stock_code')
fund['roe_ma4'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).mean())
fund['roe_std4'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).std())
fund['roe_lag4'] = g['roe'].shift(4)
fund['bps_lag4'] = g['bps'].shift(4)
fund['roe_yoy'] = fund['roe'] - fund['roe_lag4']
fund['bps_yoy'] = fund['bps'] / fund['bps_lag4'] - 1

fund['raw_factor'] = (
    0.55 * np.tanh(fund['roe_ma4'] / 8.0)
    + 0.35 * np.tanh(fund['roe_yoy'] / 6.0)
    - 0.25 * np.tanh(fund['roe_std4'] / 4.0)
    - 0.20 * np.tanh(fund['bps_yoy'] / 0.25)
)
fund = fund.replace([np.inf, -np.inf], np.nan).dropna(subset=['raw_factor'])
fund['avail_date'] = fund['report_date'] + pd.Timedelta(days=45)
qf = fund[['stock_code', 'avail_date', 'raw_factor']].rename(columns={'avail_date': 'date'})

k['date'] = pd.to_datetime(k['date'])
k['stock_code'] = k['stock_code'].astype(str).str.zfill(6)
k = k.sort_values(['stock_code', 'date']).drop_duplicates(['date', 'stock_code'])
k['log_amount_20d'] = k.groupby('stock_code')['amount'].transform(lambda s: np.log(s.rolling(20, min_periods=10).mean().clip(lower=1)))
trade_dates = pd.Index(sorted(k['date'].unique()))

res = []
for stock, grp in qf.groupby('stock_code'):
    sf = grp[['date', 'raw_factor']].drop_duplicates('date', keep='last').set_index('date').sort_index()
    sf = sf.reindex(trade_dates, method='ffill', limit=70)
    sf['stock_code'] = stock
    sf = sf.dropna(subset=['raw_factor']).reset_index().rename(columns={'index': 'date'})
    res.append(sf)

factor = pd.concat(res, ignore_index=True)
factor = factor.merge(k[['date', 'stock_code', 'log_amount_20d']], on=['date', 'stock_code'], how='inner')
factor = factor.dropna(subset=['raw_factor', 'log_amount_20d'])

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
    nz = neutralize(grp['raw_factor'].values.astype(float), grp['log_amount_20d'].values.astype(float))
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
