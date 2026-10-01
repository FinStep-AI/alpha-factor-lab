import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
FACTOR_ID = 'operating_asset_expansion_pure_v1'
OUT = BASE / f'data/factor_{FACTOR_ID}.csv'

fund = pd.read_csv(
    BASE/'data/csi1000_revenue_yoy_raw.csv',
    usecols=['stock_code','end_date','info_publ_date','account_receivable_operate_revenue','inventory_current_asset',
             'total_current_asset','end_period_cash_equivalent','debt_assets_ratio']
)
kline = pd.read_csv(BASE/'data/csi1000_kline_raw.csv', usecols=['date','stock_code','close','amount','turnover'])

fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
fund['report_date'] = pd.to_datetime(fund['end_date'])
fund['avail_date'] = pd.to_datetime(fund['info_publ_date'])
for c in ['account_receivable_operate_revenue','inventory_current_asset','total_current_asset','end_period_cash_equivalent','debt_assets_ratio']:
    fund[c] = pd.to_numeric(fund[c], errors='coerce')

fund = fund.sort_values(['stock_code','report_date']).drop_duplicates(['stock_code','report_date'], keep='last')
g = fund.groupby('stock_code')
fund['ar_rev_delta'] = fund['account_receivable_operate_revenue'] - g['account_receivable_operate_revenue'].shift(1)
fund['inv_share_delta'] = fund['inventory_current_asset'] - g['inventory_current_asset'].shift(1)
fund['cash_share'] = fund['end_period_cash_equivalent'] / fund['total_current_asset'].replace(0, np.nan)
fund['cash_share_delta'] = fund['cash_share'] - g['cash_share'].shift(1)
fund['debt_ratio_delta'] = fund['debt_assets_ratio'] - g['debt_assets_ratio'].shift(1)

for c in ['ar_rev_delta','inv_share_delta','cash_share_delta','debt_ratio_delta']:
    s = fund[c]
    q01, q99 = s.quantile([0.01, 0.99])
    fund[c] = s.clip(q01, q99)

# Pure operating-asset expansion penalty; higher factor = less aggressive balance-sheet build-up.
fund['raw_factor'] = -(
    0.42 * np.tanh(fund['ar_rev_delta'].fillna(0) / 4.0) +
    0.30 * np.tanh(fund['inv_share_delta'].fillna(0) / 6.0) -
    0.18 * np.tanh(fund['cash_share_delta'].fillna(0) / 0.08) +
    0.24 * np.tanh(fund['debt_ratio_delta'].fillna(0) / 4.0)
)
fund = fund.replace([np.inf, -np.inf], np.nan).dropna(subset=['raw_factor', 'avail_date'])
factor_q = fund[['stock_code','avail_date','raw_factor']].rename(columns={'avail_date':'date'})

kline['date'] = pd.to_datetime(kline['date'])
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline = kline.sort_values(['stock_code','date']).drop_duplicates(['date','stock_code'])
kline['mktcap_proxy'] = kline['close'].clip(lower=0.01) * kline['amount'].clip(lower=1) / (kline['turnover'].replace(0, np.nan) + 1e-6)
kline['log_mktcap'] = np.log(kline['mktcap_proxy'].clip(lower=1))
trade_dates = pd.Index(sorted(kline['date'].unique()))

frames = []
for stock, grp in factor_q.groupby('stock_code'):
    sf = grp[['date','raw_factor']].drop_duplicates('date', keep='last').set_index('date').sort_index()
    sf = sf.reindex(trade_dates, method='ffill', limit=80)
    sf['stock_code'] = stock
    sf = sf.dropna(subset=['raw_factor']).reset_index().rename(columns={'index':'date'})
    frames.append(sf)

factor = pd.concat(frames, ignore_index=True)
factor = factor.merge(kline[['date','stock_code','log_mktcap']], on=['date','stock_code'], how='inner')
factor = factor.dropna(subset=['raw_factor','log_mktcap'])

def neutralize(vals, ctrl):
    mask = np.isfinite(vals) & np.isfinite(ctrl)
    if mask.sum() < 30:
        return np.full(len(vals), np.nan)
    y = vals[mask]
    x = ctrl[mask]
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    if mad < 1e-12:
        return np.full(len(vals), np.nan)
    resid = np.clip(resid, med - 5.2 * mad, med + 5.2 * mad)
    std = resid.std()
    if std < 1e-12:
        return np.full(len(vals), np.nan)
    z = (resid - np.median(resid)) / std
    out = np.full(len(vals), np.nan)
    out[np.where(mask)[0]] = z
    return out

out = []
for date, grp in factor.groupby('date'):
    nz = neutralize(grp['raw_factor'].values.astype(float), grp['log_mktcap'].values.astype(float))
    good = np.isfinite(nz)
    if good.sum() == 0:
        continue
    sub = grp.loc[good, ['date','stock_code']].copy()
    sub['factor'] = nz[good]
    out.append(sub)

result = pd.concat(out, ignore_index=True)
result['date'] = pd.to_datetime(result['date']).dt.strftime('%Y-%m-%d')
result.to_csv(OUT, index=False, float_format='%.6f')
print(f'saved {OUT} rows={len(result)} dates={result.date.min()}~{result.date.max()} stocks={result.stock_code.nunique()}')
