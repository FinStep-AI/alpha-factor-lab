
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
FUND_PATH = BASE / 'data' / 'csi1000_fundamental_cache.csv'
KLINE_PATH = BASE / 'data' / 'csi1000_kline_raw.csv'
OUT_PATH = BASE / 'data' / 'factor_grossmargin_proxy_trend_v1.csv'


def mad_clip(s, n=3.0):
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return s
    lo = med - 1.4826 * mad * n
    hi = med + 1.4826 * mad * n
    return s.clip(lo, hi)


def zscore(s):
    std = s.std()
    if pd.isna(std) or std == 0:
        return s * 0
    return (s - s.mean()) / std


def neutralize(df, y_col='raw', x_col='log_mktcap'):
    x = df[x_col].astype(float)
    y = df[y_col].astype(float)
    mask = x.notna() & y.notna()
    out = pd.Series(np.nan, index=df.index)
    if mask.sum() < 10:
        return out
    X = np.column_stack([np.ones(mask.sum()), x[mask].values])
    beta = np.linalg.lstsq(X, y[mask].values, rcond=None)[0]
    resid = y[mask].values - X @ beta
    out.loc[mask] = resid
    return out


def map_report_to_trade_date(report_date):
    return pd.to_datetime(report_date) + pd.Timedelta(days=45)


fund = pd.read_csv(FUND_PATH)
fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
fund['report_date'] = pd.to_datetime(fund['report_date'])
fund = fund.sort_values(['stock_code', 'report_date']).copy()

# 仅有 ROE/BPS，用 BPS 同比扩张 + ROE 改善 代理“毛利率/经营质量趋势”
fund['roe_lag4'] = fund.groupby('stock_code')['roe'].shift(4)
fund['bps_lag4'] = fund.groupby('stock_code')['bps'].shift(4)
fund['roe_yoy'] = fund['roe'] - fund['roe_lag4']
fund['bps_yoy'] = fund['bps'] / fund['bps_lag4'] - 1
fund['roe_qoq'] = fund.groupby('stock_code')['roe'].diff(1)
fund['roe_ma4'] = fund.groupby('stock_code')['roe'].transform(lambda s: s.rolling(4, min_periods=3).mean())
fund['roe_std4'] = fund.groupby('stock_code')['roe'].transform(lambda s: s.rolling(4, min_periods=3).std())
fund['bps_yoy_std4'] = fund.groupby('stock_code')['bps_yoy'].transform(lambda s: s.rolling(4, min_periods=3).std())

# 趋势质量代理：盈利改善 + 净资产扩张克制 + 波动低
fund['raw_fund'] = (
    0.50 * np.tanh(fund['roe_yoy'] / 4.0)
    + 0.25 * np.tanh((fund['roe'] - fund['roe_ma4']) / 2.0)
    - 0.15 * np.tanh((fund['bps_yoy'].fillna(0)) / 0.25)
    - 0.10 * np.tanh(fund['roe_std4'].fillna(0) / 3.0)
)
fund['trade_date'] = map_report_to_trade_date(fund['report_date'])
fund = fund[['stock_code', 'trade_date', 'raw_fund']].dropna()

k = pd.read_csv(KLINE_PATH)
k['stock_code'] = k['stock_code'].astype(str).str.zfill(6)
k['date'] = pd.to_datetime(k['date'])
k = k.sort_values(['stock_code', 'date']).copy()
k['mktcap_proxy'] = k['amount'] / (k['turnover'].replace(0, np.nan) / 100.0)
k['log_mktcap'] = np.log(k['mktcap_proxy'].replace(0, np.nan))
base = k[['date', 'stock_code', 'log_mktcap']].copy().sort_values(['date', 'stock_code'])

merged = pd.merge_asof(
    base,
    fund.sort_values(['trade_date', 'stock_code']),
    left_on='date',
    right_on='trade_date',
    by='stock_code',
    direction='backward'
)
merged = merged.dropna(subset=['raw_fund', 'log_mktcap']).copy()

parts = []
for d, g in merged.groupby('date'):
    if len(g) < 50:
        continue
    g = g.copy()
    g['raw_clip'] = mad_clip(g['raw_fund'])
    g['neutral'] = neutralize(g.rename(columns={'raw_clip':'raw'}), y_col='raw', x_col='log_mktcap')
    g['factor'] = zscore(mad_clip(g['neutral'].dropna().reindex(g.index)))
    parts.append(g[['date', 'stock_code', 'factor']])

out = pd.concat(parts, ignore_index=True).dropna()
out.to_csv(OUT_PATH, index=False)
print('saved', OUT_PATH, 'rows', len(out), 'dates', out['date'].min(), out['date'].max())
