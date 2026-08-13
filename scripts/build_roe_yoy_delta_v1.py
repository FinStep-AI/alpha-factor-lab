#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
kline = pd.read_csv(BASE/'data'/'csi1000_kline_raw.csv')
fund = pd.read_csv(BASE/'data'/'csi1000_fundamental_cache.csv')

kline['date'] = pd.to_datetime(kline['date'])
fund['report_date'] = pd.to_datetime(fund['report_date'])

# use amount as market-cap proxy for required size neutralization
mkt = kline[['date','stock_code','amount']].copy()
mkt['stock_code'] = mkt['stock_code'].astype(str)
mkt['log_amount'] = np.log(mkt['amount'].clip(lower=1))
trade_dates = pd.DataFrame({'date': sorted(kline['date'].unique())})

fund['stock_code'] = fund['stock_code'].astype(str)
fund = fund.sort_values(['stock_code','report_date']).copy()
fund['roe_lag4'] = fund.groupby('stock_code')['roe'].shift(4)
fund['roe_yoy_delta'] = fund['roe'] - fund['roe_lag4']
fund['info_date'] = fund['report_date'] + pd.Timedelta(days=30)
fund = fund.dropna(subset=['roe_yoy_delta']).copy()

all_parts = []
for sc, g in fund.groupby('stock_code'):
    td = trade_dates.copy()
    mapped = pd.merge_asof(td.sort_values('date'),
                           g[['info_date','roe_yoy_delta']].sort_values('info_date'),
                           left_on='date', right_on='info_date',
                           direction='backward')
    mapped['stock_code'] = sc
    all_parts.append(mapped[['date','stock_code','roe_yoy_delta']])

panel = pd.concat(all_parts, ignore_index=True)
panel = panel.merge(mkt[['date','stock_code','log_amount']], on=['date','stock_code'], how='left')
panel = panel.dropna(subset=['roe_yoy_delta','log_amount']).copy()

# cross-sectional winsorize + size neutralize + zscore
res_list = []
for dt, g in panel.groupby('date', sort=True):
    x = g['roe_yoy_delta'].astype(float).copy()
    med = x.median()
    mad = (x-med).abs().median()
    if pd.notna(mad) and mad > 0:
        bound = 3.0 * 1.4826 * mad
        x = x.clip(med-bound, med+bound)
    y = x.values
    X = np.column_stack([np.ones(len(g)), g['log_amount'].astype(float).values])
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    fac = np.full(len(g), np.nan)
    if mask.sum() >= 20:
        beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        resid = np.full(len(g), np.nan)
        resid[mask] = y[mask] - X[mask] @ beta
        mu = np.nanmean(resid)
        sd = np.nanstd(resid)
        if np.isfinite(sd) and sd > 1e-12:
            fac = (resid - mu) / sd
        else:
            fac = resid
    out = g[['date','stock_code']].copy()
    out['factor_value'] = fac
    out['roe_yoy_delta'] = g['roe_yoy_delta'].values
    res_list.append(out)

factor = pd.concat(res_list, ignore_index=True).dropna(subset=['factor_value']).copy()
factor['date'] = pd.to_datetime(factor['date']).dt.strftime('%Y-%m-%d')
factor.to_csv(BASE/'data'/'factor_roe_yoy_delta_v1.csv', index=False)
print('saved', BASE/'data'/'factor_roe_yoy_delta_v1.csv', 'rows', len(factor), 'dates', factor['date'].nunique())
print(factor.head().to_string())
