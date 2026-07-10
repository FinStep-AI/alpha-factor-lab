#!/usr/bin/env python3
import pandas as pd, numpy as np, argparse, warnings
warnings.filterwarnings('ignore')


def robust_z(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-12:
        m = np.nanmean(x); s = np.nanstd(x)
        return (x - m) / (s + 1e-12)
    z = (x - med) / (1.4826 * mad)
    z = np.clip(z, -5, 5)
    m = np.nanmean(z); s = np.nanstd(z)
    return (z - m) / (s + 1e-12)


def neutralize_cs(df, ycol='factor_raw', xcol='log_mktcap'):
    out = []
    for dt, g in df.groupby('date'):
        tmp = g[[ycol, xcol, 'stock_code']].copy()
        tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
        if len(tmp) < 30:
            continue
        y = tmp[ycol].values.astype(float)
        x = tmp[xcol].values.astype(float)
        X = np.column_stack([np.ones(len(x)), x])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        z = robust_z(resid)
        out.append(pd.DataFrame({'date': dt, 'stock_code': tmp['stock_code'].values, 'factor_value': z}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=['date','stock_code','factor_value'])


def main(fund_file, kline_file, output_file):
    fund = pd.read_csv(fund_file)
    fund['report_date'] = pd.to_datetime(fund['report_date'])
    fund = fund.dropna(subset=['roe','bps']).copy()
    fund = fund.sort_values(['stock_code','report_date']).drop_duplicates(['stock_code','report_date'], keep='last')

    # quarterly dynamics
    g = fund.groupby('stock_code')
    fund['roe_ma2'] = g['roe'].transform(lambda s: s.rolling(2, min_periods=2).mean())
    fund['roe_prev_ma2'] = g['roe_ma2'].shift(1)
    fund['roe_recovery'] = fund['roe_ma2'] - fund['roe_prev_ma2']

    fund['bps_g'] = g['bps'].pct_change()
    fund['bps_g_mean2'] = g['bps_g'].transform(lambda s: s.rolling(2, min_periods=2).mean())
    fund['bps_g_std4'] = g['bps_g'].transform(lambda s: s.rolling(4, min_periods=3).std())
    fund['bps_stability'] = fund['bps_g_mean2'] / (fund['bps_g_std4'].abs() + 1e-6)

    # cross-sectional combine on report_date
    fund['roe_rec_z'] = fund.groupby('report_date')['roe_recovery'].transform(lambda s: pd.Series(robust_z(s.values), index=s.index))
    fund['bps_sta_z'] = fund.groupby('report_date')['bps_stability'].transform(lambda s: pd.Series(robust_z(s.values), index=s.index))
    # favor names with improving ROE and stable positive bps accumulation
    fund['factor_raw'] = 0.55 * fund['roe_rec_z'] + 0.45 * fund['bps_sta_z']

    fund = fund.dropna(subset=['factor_raw']).copy()
    fund['avail_date'] = fund['report_date'] + pd.Timedelta(days=45)

    k = pd.read_csv(kline_file)
    k['date'] = pd.to_datetime(k['date'])
    k = k.sort_values(['stock_code','date']).copy()
    k['mktcap_proxy'] = k['close'].clip(lower=0.01) * k['amount'].clip(lower=1) / (k['turnover'].replace(0, np.nan) + 1e-6)
    k['log_mktcap'] = np.log(k['mktcap_proxy'].clip(lower=1))

    panels = []
    for sc, kg in k[['date','stock_code','log_mktcap']].groupby('stock_code'):
        fg = fund.loc[fund['stock_code']==sc, ['avail_date','factor_raw']].sort_values('avail_date')
        if fg.empty:
            continue
        merged = pd.merge_asof(kg.sort_values('date'), fg, left_on='date', right_on='avail_date', direction='backward')
        merged = merged[['date','stock_code','log_mktcap','factor_raw']]
        panels.append(merged)
    daily = pd.concat(panels, ignore_index=True)
    daily = daily.dropna(subset=['factor_raw','log_mktcap'])

    out = neutralize_cs(daily, 'factor_raw', 'log_mktcap')
    out['stock_code'] = out['stock_code'].astype(str)
    out.to_csv(output_file, index=False)
    print(f'saved {output_file} rows={len(out)} dates={out.date.nunique()} stocks={out.stock_code.nunique()}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fundamental', default='data/csi1000_fundamental_cache.csv')
    ap.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    ap.add_argument('--output', default='data/factor_roe_bps_recovery_stability_v1.csv')
    a = ap.parse_args()
    main(a.fundamental, a.kline, a.output)
