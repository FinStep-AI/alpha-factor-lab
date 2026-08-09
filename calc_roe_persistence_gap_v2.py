
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
fund_path = BASE / 'data' / 'csi1000_fundamental_cache.csv'
kline_path = BASE / 'data' / 'csi1000_kline_raw.csv'
out_path = BASE / 'data' / 'factor_roe_persistence_gap_v2.csv'

fund = pd.read_csv(fund_path)
fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
fund['report_date'] = pd.to_datetime(fund['report_date'])
fund = fund.sort_values(['stock_code','report_date']).copy()

# 论文代理思路：盈利持续性越差（高位回落/高改善难持续）未来收益越弱；
# 在当前字段下，用“ROE水平 - ROE同比变化”近似 persistence gap。
# 高当前ROE但同比改善已见顶/转弱 => 更可持续；高同比冲高但ROE底子弱 => 不可持续。
# 为避免重复前面单纯 improve / smoothing 版本，这里做 level-vs-change gap。
fund['roe_lag4'] = fund.groupby('stock_code')['roe'].shift(4)
fund['roe_yoy_delta'] = fund['roe'] - fund['roe_lag4']
fund['roe_mean4'] = fund.groupby('stock_code')['roe'].transform(lambda s: s.rolling(4, min_periods=2).mean())
fund['bps_lag4'] = fund.groupby('stock_code')['bps'].shift(4)
fund['bps_yoy'] = fund['bps'] / fund['bps_lag4'] - 1

# Winsor-ish smooth transform
lvl = np.tanh(fund['roe_mean4'] / 8.0)
chg = np.tanh(fund['roe_yoy_delta'] / 6.0)
inv = np.tanh(fund['bps_yoy'] / 0.25)
fund['raw_factor'] = 0.65 * lvl - 0.55 * chg - 0.15 * inv

fund = fund[['stock_code','report_date','raw_factor']].dropna().copy()
fund['effective_date'] = fund['report_date'] + pd.Timedelta(days=45)

k = pd.read_csv(kline_path, usecols=['date','stock_code','amount'])
k['date'] = pd.to_datetime(k['date'])
k['stock_code'] = k['stock_code'].astype(str).str.zfill(6)
k = k.sort_values(['stock_code','date'])
k['log_mktcap'] = np.log(k['amount'].clip(lower=1.0))

# point-in-time mapping: backward merge latest available report to each trade date
merged = pd.merge_asof(
    k.sort_values('date'),
    fund.sort_values('effective_date'),
    left_on='date',
    right_on='effective_date',
    by='stock_code',
    direction='backward'
)
merged = merged[['date','stock_code','raw_factor','log_mktcap']].dropna().copy()

# cross-sectional neutralize by size proxy

def neutralize(group):
    g = group.dropna(subset=['raw_factor','log_mktcap']).copy()
    if len(g) < 10:
        group['factor'] = np.nan
        return group[['date','stock_code','factor']]
    x = g['log_mktcap'].values
    y = g['raw_factor'].values
    X = np.column_stack([np.ones(len(g)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    # winsor + zscore
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    if mad > 0:
        lim = 3 * 1.4826 * mad
        resid = np.clip(resid, med - lim, med + lim)
    std = resid.std()
    g['factor'] = (resid - resid.mean()) / std if std > 1e-12 else np.nan
    return g[['date','stock_code','factor']]

out = merged.groupby('date', group_keys=False).apply(neutralize)
out['date'] = pd.to_datetime(out['date']).dt.strftime('%Y-%m-%d')
out.to_csv(out_path, index=False)
print('saved', out_path, 'rows', len(out), 'dates', out['date'].nunique(), 'stocks', out['stock_code'].nunique())
print(out.head())
