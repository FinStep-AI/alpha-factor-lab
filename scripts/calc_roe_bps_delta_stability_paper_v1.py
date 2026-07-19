
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path

WD = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
FACTOR_ID = 'roe_bps_delta_stability_paper_v1'
OUT = WD / 'data' / f'factor_{FACTOR_ID}.csv'


def winsorize_mad(s, n=5.0):
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return s
    scale = 1.4826 * mad
    return s.clip(med - n * scale, med + n * scale)


def zscore(s):
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def neutralize(df, y_col='raw_factor', x_col='log_amount'):
    out = []
    for d, g in df.groupby('date', sort=False):
        g = g.copy()
        m = g[y_col].notna() & g[x_col].notna() & np.isfinite(g[y_col]) & np.isfinite(g[x_col])
        if m.sum() < 20:
            g['factor'] = np.nan
            out.append(g)
            continue
        y = g.loc[m, y_col].values.astype(float)
        x = g.loc[m, x_col].values.astype(float)
        X = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        g.loc[m, 'factor'] = resid
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    res['factor'] = res.groupby('date')['factor'].transform(lambda s: zscore(winsorize_mad(s, 5.0)))
    return res


def main():
    fund = pd.read_csv(WD / 'data' / 'csi1000_fundamental_cache.csv', parse_dates=['report_date'])
    k = pd.read_csv(WD / 'data' / 'csi1000_kline_raw.csv', parse_dates=['date'], usecols=['date','stock_code','amount'])

    fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
    k['stock_code'] = k['stock_code'].astype(str).str.zfill(6)

    fund = fund.sort_values(['stock_code','report_date']).copy()
    g = fund.groupby('stock_code', group_keys=False)

    # 论文语义：盈利持续性/成长改善；本土化成可实现代理
    fund['roe_delta_yoy'] = g['roe'].diff(4)
    fund['bps_growth_yoy'] = g['bps'].pct_change(4, fill_method=None)
    fund['roe_vol_4q'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).std())
    fund['roe_mean_4q'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).mean())

    # 高ROE改善 + 高BPS增长 + 低ROE波动；保留正盈利门控但不过度平滑
    fund['raw_factor'] = (
        (fund['roe_delta_yoy'] / (fund['roe_vol_4q'].abs() + 1.0)) *
        np.tanh(fund['roe_mean_4q'] / 10.0) +
        0.5 * fund['bps_growth_yoy'].clip(-1, 3)
    )

    fund['available_date'] = fund['report_date'] + pd.Timedelta(days=45)
    daily = k.sort_values(['stock_code','date']).copy()
    daily['amt20'] = daily.groupby('stock_code')['amount'].transform(lambda s: s.rolling(20, min_periods=5).mean())
    daily['log_amount'] = np.log(daily['amt20'].clip(lower=1))

    merged_parts = []
    right = fund[['stock_code','available_date','raw_factor']].sort_values(['stock_code','available_date'])
    for code, left_g in daily.groupby('stock_code', sort=False):
        rg = right[right['stock_code'] == code]
        if rg.empty:
            continue
        part = pd.merge_asof(
            left_g.sort_values('date'),
            rg,
            left_on='date', right_on='available_date',
            direction='backward'
        )
        merged_parts.append(part)
    df = pd.concat(merged_parts, ignore_index=True)
    if 'stock_code_x' in df.columns:
        df = df.rename(columns={'stock_code_x': 'stock_code'})
    if 'stock_code_y' in df.columns:
        df = df.drop(columns=['stock_code_y'])
    df = df.dropna(subset=['raw_factor','log_amount']).copy()
    df = neutralize(df, 'raw_factor', 'log_amount')

    out = df[['date','stock_code','factor']].dropna().copy()
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    out.to_csv(OUT, index=False)
    print(f'saved {OUT} rows={len(out)} dates={out.date.nunique()}')

if __name__ == '__main__':
    main()
