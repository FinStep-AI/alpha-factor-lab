import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path('.')
fund_path = ROOT / 'data/csi1000_fundamental_cache.csv'
kline_path = ROOT / 'data/csi1000_kline_raw.csv'
out_path = ROOT / 'data/factor_profit_persistence_discipline_v3.csv'

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
    fund[f'{col}_accel'] = fund[f'{col}_yoy'] - fund[f'{col}_yoy_prev']

fund['roe_yoy_c'] = fund['roe_yoy'].clip(-25, 25)
fund['roe_accel_c'] = fund['roe_accel'].clip(-20, 20)
fund['bps_yoy_c'] = fund['bps_yoy'].clip(-8, 8)

# 利润持续性 / 成长纪律：
# 1) ROE同比改善为主信号
# 2) ROE改善继续加速时放大
# 3) BPS扩张提供成长确认，但过猛扩张反而扣分
roe_gate = ((fund['roe_yoy'] > 0).astype(float) * 0.7 +
            (fund['roe_accel'] > 0).astype(float) * 0.3)
bps_penalty = np.maximum(fund['bps_yoy_c'] - 1.5, 0)

fund['raw_factor'] = (
    0.55 * np.tanh(fund['roe_yoy_c'] / 6.0)
    + 0.30 * np.tanh(fund['roe_accel_c'] / 4.0)
    + 0.20 * np.tanh(fund['bps_yoy_c'] / 1.2)
    - 0.25 * np.tanh(bps_penalty / 1.5)
) * roe_gate

kline = pd.read_csv(kline_path, usecols=['date', 'stock_code', 'amount', 'turnover'])
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline['date'] = pd.to_datetime(kline['date'])
kline['mkt_proxy'] = kline['amount'] / kline['turnover'].replace(0, np.nan)
kline['log_mkt_proxy'] = np.log(kline['mkt_proxy'].clip(lower=1e6))
trade = kline[['date', 'stock_code', 'log_mkt_proxy']].sort_values(['stock_code', 'date'])

fund_map = fund[['stock_code', 'report_date', 'raw_factor']].dropna().sort_values(['stock_code', 'report_date'])
merged = pd.merge_asof(
    trade.sort_values('date'),
    fund_map.sort_values('report_date'),
    left_on='date',
    right_on='report_date',
    by='stock_code',
    direction='backward'
)
merged = merged.dropna(subset=['raw_factor', 'log_mkt_proxy']).copy()

res = []
for dt, g in merged.groupby('date', sort=True):
    x = g['log_mkt_proxy'].values
    y = g['raw_factor'].values
    if len(g) < 50 or np.nanstd(x) < 1e-8 or np.nanstd(y) < 1e-8:
        continue
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta

    med = np.nanmedian(resid)
    mad = np.nanmedian(np.abs(resid - med))
    if not np.isfinite(mad) or mad < 1e-8:
        continue
    lo = med - 3 * 1.4826 * mad
    hi = med + 3 * 1.4826 * mad
    resid = np.clip(resid, lo, hi)

    std = np.nanstd(resid)
    if std < 1e-8:
        continue
    z = (resid - np.nanmean(resid)) / std

    tmp = g[['date', 'stock_code']].copy()
    tmp['factor'] = z
    res.append(tmp)

out = pd.concat(res, ignore_index=True)
out['date'] = out['date'].dt.strftime('%Y-%m-%d')
out.to_csv(out_path, index=False)
print(f'saved {out_path} rows={len(out)} dates={out.date.nunique()} stocks={out.stock_code.nunique()}')
