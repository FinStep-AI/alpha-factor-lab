#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd


def robust_z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad < 1e-12:
        std = s.std()
        if pd.isna(std) or std < 1e-12:
            return pd.Series(np.nan, index=s.index)
        return (s - s.mean()) / std
    z = (s - med) / (1.4826 * mad)
    return z.clip(-5, 5)


def neutralize_cs(df: pd.DataFrame, raw_col: str, cap_col: str) -> pd.Series:
    vals = df[raw_col].astype(float).values
    ctrl = df[cap_col].astype(float).values
    mask = np.isfinite(vals) & np.isfinite(ctrl)
    out = np.full(len(df), np.nan)
    if mask.sum() < 30:
        return pd.Series(out, index=df.index)
    y = vals[mask]
    x = ctrl[mask]
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    if mad < 1e-12:
        return pd.Series(out, index=df.index)
    clipped = np.clip(resid, med - 5.2 * mad, med + 5.2 * mad)
    std = clipped.std()
    if std < 1e-12:
        return pd.Series(out, index=df.index)
    z = (clipped - np.median(clipped)) / std
    out[np.where(mask)[0]] = z
    return pd.Series(out, index=df.index)


def compute_factor(fund_file, kline_file, output_file):
    fund = pd.read_csv(fund_file)
    fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
    fund['report_date'] = pd.to_datetime(fund['report_date'])
    for c in ['roe', 'bps']:
        fund[c] = pd.to_numeric(fund[c], errors='coerce')
    fund = fund.dropna(subset=['stock_code', 'report_date', 'roe', 'bps'])
    fund = fund.sort_values(['stock_code', 'report_date']).drop_duplicates(['stock_code', 'report_date'], keep='last')

    g = fund.groupby('stock_code')
    fund['roe_delta_1q'] = g['roe'].diff(1)
    fund['roe_delta_2q'] = g['roe'].diff(2)
    fund['roe_accel'] = fund['roe_delta_1q'] - g['roe_delta_1q'].shift(1)
    fund['roe_std4'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).std())
    fund['bps_yoy'] = g['bps'].pct_change(4) * 100
    fund['bps_yoy_std4'] = g['bps_yoy'].transform(lambda s: s.rolling(4, min_periods=3).std())
    fund['roe_level4'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).mean())

    def build_raw(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.copy()
        grp['z_roe_accel'] = robust_z(grp['roe_accel'])
        grp['z_roe_delta_2q'] = robust_z(grp['roe_delta_2q'])
        grp['z_roe_level4'] = robust_z(grp['roe_level4'])
        grp['z_roe_std4'] = robust_z(grp['roe_std4'])
        grp['z_bps_yoy'] = robust_z(grp['bps_yoy'])
        grp['z_bps_yoy_std4'] = robust_z(grp['bps_yoy_std4'])
        grp['raw_factor'] = (
            0.40 * grp['z_roe_accel'] +
            0.25 * grp['z_roe_delta_2q'] +
            0.20 * grp['z_roe_level4'] +
            0.20 * grp['z_bps_yoy'] -
            0.20 * grp['z_roe_std4'] -
            0.15 * grp['z_bps_yoy_std4']
        )
        return grp

    fund = fund.groupby('report_date', group_keys=False).apply(build_raw)
    fund = fund.replace([np.inf, -np.inf], np.nan).dropna(subset=['raw_factor'])
    fund['avail_date'] = fund['report_date'] + pd.Timedelta(days=45)
    factor_q = fund[['stock_code', 'avail_date', 'raw_factor']].rename(columns={'avail_date': 'date'})

    kline = pd.read_csv(kline_file, usecols=['date', 'stock_code', 'close', 'amount', 'turnover'])
    kline['date'] = pd.to_datetime(kline['date'])
    kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
    kline = kline.sort_values(['stock_code', 'date']).drop_duplicates(['date', 'stock_code'])
    kline['mktcap_proxy'] = kline['close'].clip(lower=0.01) * kline['amount'].clip(lower=1) / (kline['turnover'].replace(0, np.nan) + 1e-6)
    kline['log_mktcap'] = np.log(kline['mktcap_proxy'].clip(lower=1))
    trade_dates = pd.Index(sorted(kline['date'].unique()))

    aligned = []
    for stock, grp in factor_q.groupby('stock_code'):
        sf = grp[['date', 'raw_factor']].drop_duplicates('date', keep='last').set_index('date').sort_index()
        sf = sf.reindex(trade_dates, method='ffill', limit=80)
        sf['stock_code'] = stock
        sf = sf.dropna(subset=['raw_factor']).reset_index().rename(columns={'index': 'date'})
        aligned.append(sf)
    factor = pd.concat(aligned, ignore_index=True)
    factor = factor.merge(kline[['date', 'stock_code', 'log_mktcap']], on=['date', 'stock_code'], how='inner')
    factor = factor.dropna(subset=['raw_factor', 'log_mktcap'])

    out = []
    for date, grp in factor.groupby('date'):
        nz = neutralize_cs(grp, 'raw_factor', 'log_mktcap')
        good = nz.notna()
        if good.sum() < 30:
            continue
        sub = grp.loc[good, ['date', 'stock_code']].copy()
        sub['factor'] = nz.loc[good].values
        out.append(sub)

    result = pd.concat(out, ignore_index=True)
    result['date'] = pd.to_datetime(result['date']).dt.strftime('%Y-%m-%d')
    result.to_csv(output_file, index=False, float_format='%.6f')
    print(f'saved {output_file} rows={len(result)} dates={result.date.min()}~{result.date.max()} stocks={result.stock_code.nunique()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fundamental', default='data/csi1000_fundamental_cache.csv')
    parser.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    parser.add_argument('--output', default='data/factor_roe_reaccel_stability_v2.csv')
    args = parser.parse_args()
    compute_factor(args.fundamental, args.kline, args.output)
