import numpy as np
import pandas as pd
from pathlib import Path

WD = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
fund = pd.read_csv(WD/'data/csi1000_fundamental_cache.csv')
kline = pd.read_csv(WD/'data/csi1000_kline_raw.csv')

fund['report_date'] = pd.to_datetime(fund['report_date'])
kline['date'] = pd.to_datetime(kline['date'])
fund = fund.sort_values(['stock_code','report_date']).copy()

# quarterly features
for col in ['roe','bps']:
    fund[f'{col}_lag1'] = fund.groupby('stock_code')[col].shift(1)
    fund[f'{col}_lag2'] = fund.groupby('stock_code')[col].shift(2)
    fund[f'{col}_lag4'] = fund.groupby('stock_code')[col].shift(4)

fund['roe_qoq'] = fund['roe'] - fund['roe_lag1']
fund['roe_accel'] = (fund['roe'] - fund['roe_lag1']) - (fund['roe_lag1'] - fund['roe_lag2'])
fund['roe_yoy'] = fund['roe'] - fund['roe_lag4']
fund['bps_yoy'] = fund['bps'] / fund['bps_lag4'] - 1
fund['bps_qoq'] = fund['bps'] / fund['bps_lag1'] - 1
fund['roe_ma4'] = fund.groupby('stock_code')['roe'].transform(lambda s: s.rolling(4, min_periods=3).mean())
fund['roe_std4'] = fund.groupby('stock_code')['roe'].transform(lambda s: s.rolling(4, min_periods=3).std())
fund['roe_gap'] = fund['roe'] - fund['roe_ma4']

# disclosure lag: 45d for quarterlies
fund['effective_date'] = fund['report_date'] + pd.Timedelta(days=45)
features = ['roe_qoq','roe_accel','roe_yoy','bps_yoy','bps_qoq','roe_std4','roe_gap']
map_df = fund[['stock_code','effective_date'] + features].dropna(subset=['roe_yoy']).copy()
map_df = map_df.sort_values(['stock_code','effective_date'])

# merge to daily using asof by stock
all_parts=[]
for code, px in kline[['date','stock_code','amount','close','turnover']].sort_values(['stock_code','date']).groupby('stock_code'):
    ff = map_df[map_df.stock_code==code].sort_values('effective_date')
    if ff.empty:
        continue
    merged = pd.merge_asof(px.sort_values('date'),
                           ff.drop(columns=['stock_code']).rename(columns={'effective_date':'date'}),
                           on='date', direction='backward')
    all_parts.append(merged)

df = pd.concat(all_parts, ignore_index=True)
# market cap proxy
turn = df['turnover'].replace(0, np.nan) / 100.0
mktcap = df['amount'] / turn
log_mktcap = np.log(mktcap.replace([np.inf,-np.inf], np.nan))

# gated quality-growth signal
# idea: reward improving ROE when capital expansion is restrained; suppress overheated / highly volatile profitability
core = (
    0.42*np.tanh(df['roe_accel']/2.5) +
    0.28*np.tanh(df['roe_yoy']/6.0) +
    0.18*np.tanh((-df['roe_gap'])/3.0) -
    0.22*np.tanh(df['bps_yoy']/0.20) -
    0.10*np.tanh(df['bps_qoq']/0.06) -
    0.16*np.tanh(df['roe_std4']/3.0)
)

gate = (
    (df['roe_yoy'] > 0).astype(float) * 0.55 +
    (df['roe_accel'] > 0).astype(float) * 0.25 +
    (df['bps_yoy'] < 0.18).astype(float) * 0.20
)
raw = core * gate

# cross-sectional neutralization by date against log mktcap
out=[]
for d, g in df.assign(raw=raw, log_mktcap=log_mktcap)[['date','stock_code','raw','log_mktcap']].groupby('date'):
    gg = g.dropna().copy()
    if len(gg) < 30:
        continue
    x = gg['log_mktcap'].values
    y = gg['raw'].values
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    std = np.nanstd(resid)
    if std > 1e-12:
        resid = (resid - np.nanmean(resid)) / std
    else:
        resid = resid * np.nan
    tmp = pd.DataFrame({'date': d, 'stock_code': gg['stock_code'].values, 'factor': resid})
    out.append(tmp)

factor = pd.concat(out, ignore_index=True)
outfile = WD/'data/factor_roe_inflect_capital_light_v1.csv'
factor.to_csv(outfile, index=False)
print('saved', outfile, 'rows', len(factor), 'dates', factor.date.nunique(), 'stocks', factor.stock_code.nunique())
