import pandas as pd, numpy as np

fund = pd.read_csv('data/csi1000_fundamental_cache.csv')
price = pd.read_csv('data/csi1000_kline_raw.csv', usecols=['date','stock_code','amount','turnover'])
fund['report_date'] = pd.to_datetime(fund['report_date'])
price['date'] = pd.to_datetime(price['date'])
price['mcap_proxy'] = price['amount'] / price['turnover'].replace(0, np.nan)
price['log_mcap_proxy'] = np.log(price['mcap_proxy'].clip(lower=1.0))

f = fund.sort_values(['stock_code', 'report_date']).copy()
g = f.groupby('stock_code')

# 基本面代理：盈利改善 + 改善加速 + 净资产温和扩张 - ROE波动惩罚
f['roe_yoy'] = g['roe'].pct_change(4, fill_method=None)
f['roe_accel'] = g['roe_yoy'].diff(1)
f['bps_yoy'] = g['bps'].pct_change(4, fill_method=None)
f['bps_accel'] = g['bps_yoy'].diff(1)
f['roe_stability'] = g['roe'].rolling(4, min_periods=3).std().reset_index(level=0, drop=True)

f['raw'] = (
    0.45 * np.tanh(f['roe_yoy'] / 0.60) +
    0.25 * np.tanh(f['roe_accel'] / 0.40) +
    0.20 * np.tanh(f['bps_yoy'] / 0.35) -
    0.15 * np.tanh((-f['bps_accel']).clip(lower=0) / 0.25) -
    0.20 * np.tanh(f['roe_stability'].fillna(f['roe_stability'].median()) / 3.0)
)

merged = price.merge(f[['stock_code', 'report_date', 'raw']], on='stock_code', how='left')
merged = merged[merged['report_date'] <= merged['date']].sort_values(['stock_code', 'date', 'report_date'])
merged = merged.groupby(['stock_code', 'date']).tail(1).copy()

outs = []
for d, grp in merged.groupby('date'):
    gg = grp[['stock_code', 'raw', 'log_mcap_proxy']].dropna().copy()
    if len(gg) < 20:
        continue

    y = gg['raw'].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(gg)), gg['log_mcap_proxy'].to_numpy(dtype=float)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    s = pd.Series(resid, index=gg.index)

    med = s.median()
    mad = (s - med).abs().median()
    if pd.notna(mad) and mad > 0:
        band = 3.0 * 1.4826 * mad
        s = s.clip(med - band, med + band)

    std = s.std()
    if pd.notna(std) and std > 0:
        s = (s - s.mean()) / std

    tmp = gg[['stock_code']].copy()
    tmp['date'] = d
    tmp['factor'] = s.values
    outs.append(tmp)

result = pd.concat(outs, ignore_index=True)
result = result[['date', 'stock_code', 'factor']].sort_values(['date', 'stock_code'])
result.to_csv('data/factor_roe_accel_bps_sustainable_growth_v1.csv', index=False)
print('saved', len(result), 'rows to data/factor_roe_accel_bps_sustainable_growth_v1.csv')
