import numpy as np
import pandas as pd
from pathlib import Path

WD = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')

kline = pd.read_csv(WD / 'data/csi1000_kline_raw.csv')
fund = pd.read_csv(WD / 'data/csi1000_fundamental_cache.csv')

kline['date'] = pd.to_datetime(kline['date'])
fund['report_date'] = pd.to_datetime(fund['report_date'])

kline['mktcap_proxy'] = (kline['close'].abs() * kline['volume'].astype(float)).clip(lower=1e5)
kline['log_mktcap'] = np.log(kline['mktcap_proxy'])
kline = kline.sort_values(['stock_code', 'date']).copy()
kline['amount_ma20'] = kline.groupby('stock_code')['amount'].transform(lambda s: s.rolling(20, min_periods=10).mean())
kline['turnover_ma20'] = kline.groupby('stock_code')['turnover'].transform(lambda s: s.rolling(20, min_periods=10).mean())
kline['ret_vol20'] = kline.groupby('stock_code')['pct_change'].transform(lambda s: s.rolling(20, min_periods=10).std())
kline['amplitude_ma20'] = kline.groupby('stock_code')['amplitude'].transform(lambda s: s.rolling(20, min_periods=10).mean())

fund = fund.sort_values(['stock_code', 'report_date']).copy()
fund['roe_lag4'] = fund.groupby('stock_code')['roe'].shift(4)
fund['bps_lag4'] = fund.groupby('stock_code')['bps'].shift(4)
fund['bps_lag1'] = fund.groupby('stock_code')['bps'].shift(1)
fund['roe_mean4'] = fund.groupby('stock_code')['roe'].transform(lambda s: s.rolling(4, min_periods=4).mean())
fund['roe_std4'] = fund.groupby('stock_code')['roe'].transform(lambda s: s.rolling(4, min_periods=4).std())
fund['bps_yoy'] = fund['bps'] / fund['bps_lag4'] - 1
fund['bps_qoq'] = fund['bps'] / fund['bps_lag1'] - 1
fund['raw_quarter'] = -(
    0.55 * np.tanh(fund['bps_yoy'] / 0.35)
    + 0.20 * np.tanh(fund['roe_mean4'] / 8.0)
    - 0.15 * np.tanh(fund['roe_std4'].fillna(0) / 4.0)
    - 0.10 * np.tanh(fund['bps_qoq'] / 0.20)
)

trade_dates = np.array(sorted(kline['date'].unique()), dtype='datetime64[ns]')

def next_trade_day(dt):
    target = np.datetime64(dt + pd.Timedelta(days=45))
    idx = trade_dates.searchsorted(target)
    if idx >= len(trade_dates):
        return pd.NaT
    return pd.Timestamp(trade_dates[idx])

fund['date'] = fund['report_date'].map(next_trade_day)
fund = fund.dropna(subset=['date', 'raw_quarter'])

kline_sub = kline[['date','stock_code','log_mktcap','amount_ma20','turnover_ma20','ret_vol20','amplitude_ma20']].sort_values(['date','stock_code']).copy()
merged = pd.merge_asof(
    kline_sub.sort_values('date'),
    fund[['stock_code','date','raw_quarter']].sort_values('date'),
    on='date', by='stock_code', direction='backward'
)
merged['liq_penalty'] = (
    0.45 * np.tanh(np.log1p(merged['amount_ma20']) / 22.0)
    + 0.35 * np.tanh(merged['turnover_ma20'] / 8.0)
    + 0.20 * np.tanh(merged['ret_vol20'] / 4.0)
)
merged['raw_factor'] = merged['raw_quarter'] - 0.35 * merged['liq_penalty'] - 0.08 * np.tanh(merged['amplitude_ma20'] / 6.0)

def cs_resid(df):
    sub = df[['raw_factor','log_mktcap']].replace([np.inf,-np.inf], np.nan).dropna()
    if len(sub) < 20:
        return pd.Series(index=df.index, dtype=float)
    y = sub['raw_factor'].values
    x = sub['log_mktcap'].values
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X.dot(beta)
    out = pd.Series(index=sub.index, data=resid)
    med = np.nanmedian(resid)
    mad = np.nanmedian(np.abs(resid - med)) + 1e-8
    out = out.clip(med - 5*mad, med + 5*mad)
    out = (out - out.mean()) / (out.std(ddof=0) + 1e-8)
    return out

merged['factor'] = merged.groupby('date', group_keys=False).apply(cs_resid)
out = merged[['date','stock_code','factor']].dropna().copy()
out.to_csv(WD/'data/factor_bps_turnover_efficiency_v1.csv', index=False)
print('saved', len(out))
