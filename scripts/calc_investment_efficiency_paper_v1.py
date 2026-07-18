#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investment efficiency paper-style factor v1

Paper anchor:
- q-factor investment idea (low investment better)
- localized with profitability gate using available A-share data

factor_raw = -bps_yoy / (abs(roe) + 5)
Interpretation:
- punish aggressive equity/book expansion
- reward low investment conditional on profitability support

45-day lag + neutralize vs log(amount) + MAD winsorize + zscore
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data'
OUT = DATA / 'factor_investment_efficiency_paper_v1.csv'


def mad_winsorize(x, n=5.0):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-12:
        return x
    s = 1.4826 * mad
    return np.clip(x, med - n * s, med + n * s)


def neutralize_cs(df):
    g = df[['raw', 'log_amount']].replace([np.inf, -np.inf], np.nan).dropna().copy()
    out = pd.Series(np.nan, index=df.index)
    if len(g) < 20:
        return out
    X = np.column_stack([np.ones(len(g)), g['log_amount'].values])
    y = g['raw'].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    resid = mad_winsorize(resid, 5.0)
    sd = resid.std()
    z = (resid - resid.mean()) / sd if sd > 1e-12 else resid - resid.mean()
    out.loc[g.index] = z
    return out


def main():
    kline = pd.read_csv(DATA / 'csi1000_kline_raw.csv', parse_dates=['date'])
    fund = pd.read_csv(DATA / 'csi1000_fundamental_cache.csv', parse_dates=['report_date'])
    kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
    fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
    fund = fund.dropna(subset=['bps', 'roe']).drop_duplicates(['stock_code', 'report_date']).sort_values(['stock_code', 'report_date'])

    fund['bps_lag4'] = fund.groupby('stock_code')['bps'].shift(4)
    fund['bps_yoy'] = fund['bps'] / fund['bps_lag4'] - 1
    fund.loc[(fund['bps_lag4'] <= 0) | (~np.isfinite(fund['bps_yoy'])), 'bps_yoy'] = np.nan
    fund['raw'] = -fund['bps_yoy'] / (fund['roe'].abs() + 5.0)
    fund['avail_date'] = (fund['report_date'] + pd.Timedelta(days=45)).dt.normalize()

    signal = fund[['stock_code', 'avail_date', 'raw']].dropna().sort_values(['stock_code', 'avail_date'])
    panel = kline[['date', 'stock_code', 'amount']].copy()
    panel['date'] = pd.to_datetime(panel['date']).dt.normalize()
    panel['log_amount'] = np.log(panel['amount'].clip(lower=1.0))
    panel = panel.sort_values(['stock_code', 'date'])

    merged = pd.merge_asof(
        panel.sort_values('date'),
        signal.rename(columns={'avail_date': 'date'}).sort_values('date'),
        by='stock_code', on='date', direction='backward'
    )
    merged = merged.dropna(subset=['raw'])

    out = []
    for dt, g in merged.groupby('date'):
        g = g.copy()
        g['factor_value'] = neutralize_cs(g)
        g = g.dropna(subset=['factor_value'])
        if len(g) < 50:
            continue
        out.append(g[['date', 'stock_code', 'factor_value']])

    res = pd.concat(out, ignore_index=True)
    res.to_csv(OUT, index=False)
    print(f'Saved {OUT} rows={len(res)} dates={res.date.nunique()} stocks={res.stock_code.nunique()}')
    print(res['factor_value'].describe())


if __name__ == '__main__':
    main()
