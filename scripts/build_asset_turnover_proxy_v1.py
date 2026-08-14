#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path

WINDOW_Q = 4
MIN_Q = 3


def winsorize(s, lower=0.025, upper=0.975):
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def neutralize(group):
    g = group.dropna(subset=['factor_zscore', 'log_mktcap'])
    if len(g) < 10:
        g = g.copy()
        g['factor_neutral'] = np.nan
        return g[['factor_neutral']]
    x = g['log_mktcap'].values
    y = g['factor_zscore'].values
    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)
    b = np.nansum((x - x_mean) * (y - y_mean)) / (np.nansum((x - x_mean) ** 2) + 1e-10)
    a = y_mean - b * x_mean
    g = g.copy()
    g['factor_neutral'] = y - (a + b * x)
    return g[['factor_neutral']]


def main():
    data_dir = Path('data')
    out_path = data_dir / 'factor_asset_turnover_proxy_v1.csv'

    fund = pd.read_csv(data_dir / 'csi1000_fundamental_cache.csv')
    kline = pd.read_csv(data_dir / 'csi1000_kline_raw.csv', usecols=['date','stock_code','amount','turnover'])

    fund['stock_code'] = fund['stock_code'].astype(str)
    kline['stock_code'] = kline['stock_code'].astype(str)
    fund['report_date'] = pd.to_datetime(fund['report_date'])
    kline['date'] = pd.to_datetime(kline['date'])

    fund = fund.sort_values(['stock_code','report_date']).copy()
    kline = kline.sort_values(['stock_code','date']).copy()

    # proxy: ROE = NetIncome / Equity ; BPS ∝ Equity/share
    # asset-turnover-like improvement proxy = growth(net_income_proxy) - growth(equity_proxy)
    # with smoothing on quarterly changes
    fund['roe_dec'] = fund['roe'] / 100.0
    fund['ni_proxy'] = fund['roe_dec'] * fund['bps']
    fund['equity_proxy'] = fund['bps']

    for col in ['ni_proxy', 'equity_proxy']:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    # yoy growth and short-term acceleration blend
    for col in ['ni_proxy','equity_proxy','roe_dec']:
        fund[f'{col}_yoy'] = fund.groupby('stock_code')[col].pct_change(4, fill_method=None)
        fund[f'{col}_qoq'] = fund.groupby('stock_code')[col].pct_change(1, fill_method=None)
        fund[f'{col}_slope4'] = fund.groupby('stock_code')[col].transform(
            lambda s: s.rolling(WINDOW_Q, min_periods=MIN_Q).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if np.isfinite(x).sum() >= MIN_Q else np.nan,
                raw=False
            )
        )

    # core signal: NI growth outpaces equity growth => operating efficiency / turnover improving
    fund['raw_factor'] = (
        0.50 * (fund['ni_proxy_yoy'] - fund['equity_proxy_yoy']) +
        0.25 * (fund['ni_proxy_qoq'] - fund['equity_proxy_qoq']) +
        0.25 * fund['roe_dec_slope4']
    )

    fund['raw_factor'] = fund['raw_factor'].replace([np.inf, -np.inf], np.nan)
    fund = fund.dropna(subset=['raw_factor'])

    # align quarterly fundamental to daily panel via asof merge
    daily = kline[['date','stock_code','amount','turnover']].copy()
    daily['mktcap_proxy'] = daily['amount'] / daily['turnover'].replace(0, np.nan)
    daily['log_mktcap'] = np.log(daily['mktcap_proxy'].replace(0, np.nan))

    daily = daily.sort_values(['date','stock_code']).reset_index(drop=True)
    fund_asof = fund[['stock_code','report_date','raw_factor']].sort_values(['report_date','stock_code']).reset_index(drop=True)

    merged = pd.merge_asof(
        daily,
        fund_asof,
        left_on='date',
        right_on='report_date',
        by='stock_code',
        direction='backward'
    )

    result = merged[['date','stock_code','raw_factor','log_mktcap']].dropna().copy()
    result['raw_factor'] = result.groupby('date')['raw_factor'].transform(winsorize)
    result['factor_zscore'] = result.groupby('date')['raw_factor'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    ).clip(-3,3)

    neutralized = result.groupby('date', group_keys=False).apply(neutralize)
    result['factor_neutral'] = neutralized['factor_neutral'].values
    result['factor_value'] = result.groupby('date')['factor_neutral'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    ).clip(-3,3)

    output = result[['date','stock_code','factor_value']].dropna().copy()
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')
    output.to_csv(out_path, index=False)

    print(f'Saved to {out_path}')
    print(output.head())
    print(output['factor_value'].describe())
    print('rows', len(output), 'dates', output['date'].min(), output['date'].max())

if __name__ == '__main__':
    main()
