#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
FACTOR_ID = 'quality_minus_investment_proxy_v1'
OUT = BASE / f'data/factor_{FACTOR_ID}.csv'

fund = pd.read_csv(BASE / 'data/csi1000_revenue_yoy_raw.csv')
kline = pd.read_csv(BASE / 'data/csi1000_kline_raw.csv', usecols=['date', 'stock_code', 'close', 'amount', 'turnover'])

fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
fund['end_date'] = pd.to_datetime(fund['end_date'])
fund['info_publ_date'] = pd.to_datetime(fund['info_publ_date'])
for c in ['roe','gross_income_ratio','operate_revenue_yoy','net_profit_yoy','debt_assets_ratio','current_ratio','quick_ratio','account_receivable_operate_revenue','inventory_turnover_rate']:
    if c in fund.columns:
        fund[c] = pd.to_numeric(fund[c], errors='coerce')

keep_cols = ['stock_code','end_date','info_publ_date','roe','gross_income_ratio','operate_revenue_yoy','net_profit_yoy','debt_assets_ratio','current_ratio','quick_ratio','account_receivable_operate_revenue','inventory_turnover_rate']
fund = fund[keep_cols].drop_duplicates(['stock_code','end_date']).sort_values(['stock_code','end_date'])

# clip quarterly outliers
for c in ['roe','gross_income_ratio','operate_revenue_yoy','net_profit_yoy','debt_assets_ratio','current_ratio','quick_ratio','account_receivable_operate_revenue','inventory_turnover_rate']:
    s = fund[c]
    q01, q99 = s.quantile([0.01, 0.99])
    fund[c] = s.clip(q01, q99)

g = fund.groupby('stock_code')
fund['roe_mean4'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).mean())
fund['roe_std4'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).std())
fund['gm_mean4'] = g['gross_income_ratio'].transform(lambda s: s.rolling(4, min_periods=3).mean())
fund['gm_std4'] = g['gross_income_ratio'].transform(lambda s: s.rolling(4, min_periods=3).std())
fund['rev_yoy_mean2'] = g['operate_revenue_yoy'].transform(lambda s: s.rolling(2, min_periods=2).mean())
fund['np_yoy_mean2'] = g['net_profit_yoy'].transform(lambda s: s.rolling(2, min_periods=2).mean())
fund['inv_turn_mean4'] = g['inventory_turnover_rate'].transform(lambda s: s.rolling(4, min_periods=3).mean())
fund['ar_ratio_mean4'] = g['account_receivable_operate_revenue'].transform(lambda s: s.rolling(4, min_periods=3).mean())
fund['debt_mean4'] = g['debt_assets_ratio'].transform(lambda s: s.rolling(4, min_periods=3).mean())
fund['liq_mean4'] = ((fund['current_ratio'] + fund['quick_ratio']) / 2.0)
fund['liq_mean4'] = g['liq_mean4'].transform(lambda s: s.rolling(4, min_periods=3).mean())

# Paper-inspired proxy: robust profitability minus aggressive/low-quality expansion.
profitability = 0.45*np.tanh(fund['roe_mean4']/8.0) + 0.35*np.tanh(fund['gm_mean4']/20.0) + 0.20*np.tanh(fund['np_yoy_mean2']/40.0)
growth = 0.60*np.tanh(fund['rev_yoy_mean2']/40.0) + 0.40*np.tanh(fund['inv_turn_mean4']/8.0)
safety = 0.40*np.tanh((2.0 - fund['debt_mean4'])/20.0) + 0.30*np.tanh(fund['liq_mean4']-1.0) - 0.30*np.tanh(fund['ar_ratio_mean4']/20.0)
stability_penalty = 0.50*np.tanh(fund['roe_std4'].abs()/6.0) + 0.50*np.tanh(fund['gm_std4'].abs()/10.0)

fund['raw_factor'] = profitability + 0.35*growth + 0.20*safety - 0.55*stability_penalty
fund = fund.replace([np.inf, -np.inf], np.nan).dropna(subset=['raw_factor','info_publ_date'])

# lag after publication to avoid lookahead
fund['avail_date'] = fund['info_publ_date'] + pd.Timedelta(days=5)
factor_q = fund[['stock_code','avail_date','raw_factor']].rename(columns={'avail_date':'date'})

kline['date'] = pd.to_datetime(kline['date'])
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline = kline.sort_values(['stock_code','date']).drop_duplicates(['date','stock_code'])
kline['mktcap_proxy'] = kline['close'].clip(lower=0.01) * kline['amount'].clip(lower=1) / (kline['turnover'].replace(0, np.nan) + 1e-6)
kline['log_mktcap'] = np.log(kline['mktcap_proxy'].clip(lower=1))
trade_dates = pd.Index(sorted(kline['date'].unique()))

res = []
for stock, grp in factor_q.groupby('stock_code'):
    sf = grp[['date','raw_factor']].drop_duplicates('date', keep='last').set_index('date').sort_index()
    sf = sf.reindex(trade_dates, method='ffill', limit=80)
    sf['stock_code'] = stock
    sf = sf.dropna(subset=['raw_factor']).reset_index().rename(columns={'index':'date'})
    res.append(sf)

factor = pd.concat(res, ignore_index=True)
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
    clipped = np.clip(resid, med - 5.2*mad, med + 5.2*mad)
    std = clipped.std()
    if std < 1e-12:
        return np.full(len(vals), np.nan)
    z = (clipped - np.median(clipped))/std
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
