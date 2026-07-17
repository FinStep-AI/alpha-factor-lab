
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
FACTOR_ID = 'gross_margin_revenue_quality_v1'
OUT = BASE / f'data/factor_{FACTOR_ID}.csv'

fund = pd.read_csv(
    BASE/'data/csi1000_revenue_yoy_raw.csv',
    usecols=[
        'stock_code','end_date','info_publ_date','gross_income_ratio','gross_income_ratio_yoy',
        'operate_revenue_yoy','inventory_turnover_rate','account_receivable_operate_revenue',
        'debt_assets_ratio','current_ratio','quick_ratio'
    ]
)
kline = pd.read_csv(BASE/'data/csi1000_kline_raw.csv', usecols=['date','stock_code','close','amount','turnover'])

fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
fund['report_date'] = pd.to_datetime(fund['end_date'])
fund['avail_date'] = pd.to_datetime(fund['info_publ_date'])
for c in [
    'gross_income_ratio','gross_income_ratio_yoy','operate_revenue_yoy','inventory_turnover_rate',
    'account_receivable_operate_revenue','debt_assets_ratio','current_ratio','quick_ratio'
]:
    fund[c] = pd.to_numeric(fund[c], errors='coerce')

fund = fund.sort_values(['stock_code','report_date']).drop_duplicates(['stock_code','report_date'], keep='last')

g = fund.groupby('stock_code')
fund['gm_yoy_lag1'] = g['gross_income_ratio_yoy'].shift(1)
fund['rev_yoy_lag1'] = g['operate_revenue_yoy'].shift(1)
fund['inv_turn_lag1'] = g['inventory_turnover_rate'].shift(1)
fund['ar_ratio_lag1'] = g['account_receivable_operate_revenue'].shift(1)
fund['debt_lag1'] = g['debt_assets_ratio'].shift(1)

fund['gm_trend2'] = g['gross_income_ratio_yoy'].transform(lambda s: s.rolling(2, min_periods=2).mean())
fund['rev_trend2'] = g['operate_revenue_yoy'].transform(lambda s: s.rolling(2, min_periods=2).mean())
fund['gm_vol4'] = g['gross_income_ratio_yoy'].transform(lambda s: s.rolling(4, min_periods=3).std())
fund['rev_vol4'] = g['operate_revenue_yoy'].transform(lambda s: s.rolling(4, min_periods=3).std())

fund['inv_turn_improve'] = fund['inventory_turnover_rate'] - fund['inv_turn_lag1']
fund['ar_ratio_change'] = fund['account_receivable_operate_revenue'] - fund['ar_ratio_lag1']
fund['debt_change'] = fund['debt_assets_ratio'] - fund['debt_lag1']
fund['liq_buffer'] = 0.5*fund['current_ratio'] + 0.5*fund['quick_ratio']

# winsorize raw inputs
for c in [
    'gross_income_ratio_yoy','operate_revenue_yoy','gm_trend2','rev_trend2','gm_vol4','rev_vol4',
    'inv_turn_improve','ar_ratio_change','debt_change','liq_buffer'
]:
    s = fund[c]
    q1, q99 = s.quantile([0.01,0.99])
    fund[c] = s.clip(q1, q99)

# factor intuition:
# 1) prefer firms with improving gross margin and revenue growth trend
# 2) penalize unstable growth
# 3) reward inventory turnover improvement
# 4) penalize receivables ratio rise and debt rise
# 5) require some liquidity buffer
margin_growth = np.tanh(fund['gm_trend2'] / 20.0)
revenue_growth = np.tanh(fund['rev_trend2'] / 30.0)
stability = 1.0 / (1.0 + 0.08*fund['gm_vol4'].abs().fillna(0) + 0.05*fund['rev_vol4'].abs().fillna(0))
efficiency = np.tanh(fund['inv_turn_improve'].fillna(0) / 1.5) - 0.7*np.tanh(fund['ar_ratio_change'].fillna(0) / 8.0)
bs_gate = np.tanh((fund['liq_buffer'].fillna(0) - 1.0) / 1.0) - 0.6*np.tanh(fund['debt_change'].fillna(0) / 10.0)
level_anchor = np.tanh(fund['gross_income_ratio'].fillna(0) / 25.0)

fund['raw_factor'] = (0.30*margin_growth + 0.30*revenue_growth + 0.20*efficiency + 0.20*bs_gate) * stability * (0.7 + 0.3*level_anchor)
fund = fund.replace([np.inf, -np.inf], np.nan).dropna(subset=['raw_factor','avail_date'])

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
    resid = np.clip(resid, med - 5*mad, med + 5*mad)
    std = resid.std()
    if std < 1e-12:
        return np.full(len(vals), np.nan)
    z = (resid - np.median(resid)) / std
    out = np.full(len(vals), np.nan)
    out[np.where(mask)[0]] = z
    return out

out=[]
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
print(f'saved {OUT} rows={len(result)} unique_dates={result.date.nunique()}')
print(result.head().to_string())
