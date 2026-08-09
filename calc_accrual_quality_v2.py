import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
FACTOR_ID = 'accrual_quality_v2'
OUT = BASE / f'data/factor_{FACTOR_ID}.csv'

fund = pd.read_csv(
    BASE/'data/csi1000_revenue_yoy_raw.csv',
    usecols=['stock_code','end_date','info_publ_date','net_profit_yoy','operate_revenue_yoy',
             'account_receivable_yoy','inventory_current_asset','inventory_turnover_rate',
             'account_receivable_operate_revenue','roe','gross_income_ratio']
)
kline = pd.read_csv(BASE/'data/csi1000_kline_raw.csv', usecols=['date','stock_code','close','amount','turnover'])

fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
fund['report_date'] = pd.to_datetime(fund['end_date'])
fund['avail_date'] = pd.to_datetime(fund['info_publ_date'])
for c in ['net_profit_yoy','operate_revenue_yoy','account_receivable_yoy','inventory_current_asset',
          'inventory_turnover_rate','account_receivable_operate_revenue','roe','gross_income_ratio']:
    fund[c] = pd.to_numeric(fund[c], errors='coerce')

fund = fund.sort_values(['stock_code','report_date']).drop_duplicates(['stock_code','report_date'], keep='last')
g = fund.groupby('stock_code')

fund['rev_yoy_ma2'] = g['operate_revenue_yoy'].transform(lambda s: s.rolling(2, min_periods=2).mean())
fund['profit_yoy_ma2'] = g['net_profit_yoy'].transform(lambda s: s.rolling(2, min_periods=2).mean())
fund['profit_rev_gap'] = fund['profit_yoy_ma2'] - fund['rev_yoy_ma2']
fund['ar_rev_ratio_delta'] = fund['account_receivable_operate_revenue'] - g['account_receivable_operate_revenue'].shift(1)
fund['inv_share_delta'] = fund['inventory_current_asset'] - g['inventory_current_asset'].shift(1)
fund['inv_turn_delta'] = fund['inventory_turnover_rate'] - g['inventory_turnover_rate'].shift(1)
fund['roe_ma4'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).mean())
fund['gm_ma4'] = g['gross_income_ratio'].transform(lambda s: s.rolling(4, min_periods=3).mean())

for c in ['profit_rev_gap','account_receivable_yoy','ar_rev_ratio_delta','inv_share_delta','inv_turn_delta','roe_ma4','gm_ma4']:
    s = fund[c]
    q01, q99 = s.quantile([0.01, 0.99])
    fund[c] = s.clip(q01, q99)

# Sloan-style accrual/earnings-quality intuition, localized with available A-share fields:
# earnings growth not backed by revenue, receivables build-up, and inventory swelling imply lower quality.
# Better inventory turnover and decent profitability act as quality anchors.
profit_gap_penalty = np.tanh(fund['profit_rev_gap'].fillna(0) / 25.0)
receivable_penalty = 0.7 * np.tanh(fund['account_receivable_yoy'].fillna(0) / 30.0) + 0.6 * np.tanh(fund['ar_rev_ratio_delta'].fillna(0) / 6.0)
inventory_penalty = 0.5 * np.tanh(fund['inv_share_delta'].fillna(0) / 8.0) - 0.5 * np.tanh(fund['inv_turn_delta'].fillna(0) / 1.2)
profitability_anchor = 0.35 * np.tanh(fund['roe_ma4'].fillna(0) / 8.0) + 0.25 * np.tanh(fund['gm_ma4'].fillna(0) / 20.0)

fund['raw_factor'] = profitability_anchor - profit_gap_penalty - receivable_penalty - inventory_penalty
fund = fund.replace([np.inf, -np.inf], np.nan).dropna(subset=['raw_factor','avail_date'])
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
print(f'saved {OUT} rows={len(result)} dates={result.date.min()}~{result.date.max()}')
