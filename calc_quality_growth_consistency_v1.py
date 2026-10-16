import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path('.')
fund_path = ROOT / 'data/csi1000_fundamental_cache.csv'
kline_path = ROOT / 'data/csi1000_kline_raw.csv'
out_path = ROOT / 'data/factor_quality_growth_consistency_v1.csv'

fund = pd.read_csv(fund_path)
fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
fund['report_date'] = pd.to_datetime(fund['report_date'])
fund = fund.sort_values(['stock_code', 'report_date']).copy()

for col in ['roe', 'bps']:
    g = fund.groupby('stock_code')[col]
    fund[f'{col}_lag4'] = g.shift(4)
    fund[f'{col}_lag8'] = g.shift(8)
    fund[f'{col}_yoy'] = fund[col] - fund[f'{col}_lag4']
    fund[f'{col}_yoy_prev'] = fund[f'{col}_lag4'] - fund[f'{col}_lag8']
    fund[f'{col}_reaccel'] = fund[f'{col}_yoy'] - fund[f'{col}_yoy_prev']

fund['roe_yoy_clip'] = fund['roe_yoy'].clip(-30, 30)
fund['bps_yoy_clip'] = fund['bps_yoy'].clip(-10, 10)
fund['consistency'] = - (fund['roe_reaccel'].abs().clip(0, 20) * 0.6 + fund['bps_reaccel'].abs().clip(0, 10) * 0.4)
fund['raw_factor'] = (
    0.50 * fund['roe_yoy_clip'] +
    0.30 * fund['bps_yoy_clip'] +
    0.20 * fund['consistency']
)

# 将财报值向后映射到交易日，避免未来函数：每个交易日使用最近已披露 report_date<=date 的值
kline = pd.read_csv(kline_path, usecols=['date', 'stock_code', 'close'])
kline['date'] = pd.to_datetime(kline['date'])
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
# 用close近似市值暴露代理（仅用于中性化）
trade = kline.sort_values(['stock_code', 'date'])[['date', 'stock_code', 'close']].copy()

fund_map = fund[['stock_code', 'report_date', 'raw_factor']].dropna().sort_values(['stock_code', 'report_date'])
merged = pd.merge_asof(
    trade.sort_values('date'),
    fund_map.sort_values('report_date'),
    left_on='date',
    right_on='report_date',
    by='stock_code',
    direction='backward'
)
merged = merged.dropna(subset=['raw_factor', 'close']).copy()
merged['log_mkt_proxy'] = np.log(merged['close'].clip(lower=0.5))

# 截面市值中性化 + zscore
res = []
for dt, g in merged.groupby('date', sort=True):
    x = g['log_mkt_proxy'].values
    y = g['raw_factor'].values
    if len(g) < 20 or np.nanstd(x) < 1e-8 or np.nanstd(y) < 1e-8:
        continue
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    std = resid.std()
    if std < 1e-8:
        continue
    z = (resid - resid.mean()) / std
    tmp = g[['date', 'stock_code']].copy()
    tmp['factor'] = z
    res.append(tmp)

out = pd.concat(res, ignore_index=True)
out['date'] = out['date'].dt.strftime('%Y-%m-%d')
out.to_csv(out_path, index=False)
print(f'saved {out_path} rows={len(out)} dates={out.date.nunique()} stocks={out.stock_code.nunique()}')
