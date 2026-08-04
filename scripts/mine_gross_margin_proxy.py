
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

kline = pd.read_csv(BASE / 'data' / 'csi1000_kline_raw.csv')
fund = pd.read_csv(BASE / 'data' / 'csi1000_fundamental_cache.csv')

kline['date'] = pd.to_datetime(kline['date'])
fund['report_date'] = pd.to_datetime(fund['report_date'])

# stock_code standardize
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)

# proxy variables from available data
fund = fund.sort_values(['report_date','stock_code']).copy()
fund['roe_yoy'] = fund.groupby('stock_code')['roe'].pct_change(4, fill_method=None)
fund['bps_yoy'] = fund.groupby('stock_code')['bps'].pct_change(4, fill_method=None)
fund['roe_qoq'] = fund.groupby('stock_code')['roe'].diff(1)
fund['bps_qoq'] = fund.groupby('stock_code')['bps'].pct_change(1, fill_method=None)

# 毛利率趋势代理：ROE同比改善 + 每股净资产同比扩张，惩罚高波动/高振幅
fund['gross_margin_trend_proxy'] = (
    0.65 * fund['roe_yoy'] +
    0.35 * fund['bps_yoy'] -
    0.10 * fund['roe_qoq'].abs()
)

fund = fund[['stock_code','report_date','gross_margin_trend_proxy']].dropna()

# asof merge fundamentals to daily bars
kline = kline.sort_values(['date','stock_code']).copy()
fund = fund.sort_values(['report_date','stock_code']).copy()
merged = pd.merge_asof(
    kline,
    fund,
    left_on='date',
    right_on='report_date',
    by='stock_code',
    direction='backward'
)

# market cap proxy from amount / turnover
turn = merged['turnover'].replace(0, np.nan) / 100.0
merged['mcap_proxy'] = merged['amount'] / turn
merged['log_mcap'] = np.log(merged['mcap_proxy'].replace(0, np.nan))
merged['amp20'] = merged.groupby('stock_code')['amplitude'].transform(lambda s: s.rolling(20, min_periods=10).mean())
merged['vol20'] = merged.groupby('stock_code')['pct_change'].transform(lambda s: s.rolling(20, min_periods=10).std())

# raw factor: fundamental trend minus short-term risk/noise
merged['raw_factor'] = (
    merged['gross_margin_trend_proxy']
    - 0.15 * np.log1p(merged['amp20'].clip(lower=0))
    - 0.10 * merged['vol20']
)

# cross-sectional market-cap neutralization by date
out = []
for dt, g in merged.groupby('date'):
    x = g[['stock_code','raw_factor','log_mcap']].copy()
    x = x.replace([np.inf,-np.inf], np.nan).dropna()
    if len(x) < 30:
        continue
    y = x['raw_factor'].values
    X = np.column_stack([np.ones(len(x)), x['log_mcap'].values])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    med = np.nanmedian(resid)
    mad = np.nanmedian(np.abs(resid - med))
    if mad > 1e-12:
        resid = np.clip(resid, med - 3*1.4826*mad, med + 3*1.4826*mad)
    std = np.nanstd(resid)
    if std > 1e-12:
        resid = (resid - np.nanmean(resid)) / std
    else:
        resid = resid - np.nanmean(resid)
    tmp = pd.DataFrame({'date': dt, 'stock_code': x['stock_code'].values, 'factor': resid})
    out.append(tmp)

out = pd.concat(out, ignore_index=True)
out['date'] = out['date'].dt.strftime('%Y-%m-%d')
out.to_csv(BASE / 'data' / 'factor_gross_margin_trend_proxy_v1.csv', index=False)
print('saved', len(out), 'rows to data/factor_gross_margin_trend_proxy_v1.csv')
print(out.head().to_string())
