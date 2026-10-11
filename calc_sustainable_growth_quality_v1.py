
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
FACTOR_ID = 'sustainable_growth_quality_v1'
OUT = BASE / f'data/factor_{FACTOR_ID}.csv'

fund = pd.read_csv(
    BASE/'data/csi1000_revenue_yoy_raw.csv',
    usecols=[
        'stock_code','end_date','info_publ_date','roe','roa','gross_income_ratio','operate_revenue_yoy',
        'inventory_turnover_rate','account_receivable_operate_revenue','debt_assets_ratio',
        'current_ratio','quick_ratio','retained_profit','end_period_cash_equivalent'
    ]
)
kline = pd.read_csv(BASE/'data/csi1000_kline_raw.csv', usecols=['date','stock_code','close','amount','turnover'])

fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
fund['report_date'] = pd.to_datetime(fund['end_date'])
fund['avail_date'] = pd.to_datetime(fund['info_publ_date'])
for c in ['roe','roa','gross_income_ratio','operate_revenue_yoy','inventory_turnover_rate',
          'account_receivable_operate_revenue','debt_assets_ratio','current_ratio','quick_ratio',
          'retained_profit','end_period_cash_equivalent']:
    fund[c] = pd.to_numeric(fund[c], errors='coerce')

fund = fund.sort_values(['stock_code','report_date']).drop_duplicates(['stock_code','report_date'], keep='last')
g = fund.groupby('stock_code')

fund['retained_log'] = np.log1p(fund['retained_profit'].clip(lower=0))
fund['cash_log'] = np.log1p(fund['end_period_cash_equivalent'].clip(lower=0))
fund['retained_growth'] = fund['retained_log'] - g['retained_log'].shift(4)
fund['cash_growth'] = fund['cash_log'] - g['cash_log'].shift(4)
fund['roe_accel'] = fund['roe'] - g['roe'].shift(1)
fund['rev_accel'] = fund['operate_revenue_yoy'] - g['operate_revenue_yoy'].shift(1)
fund['roa_trend'] = fund['roa'] - g['roa'].shift(4)
fund['gm_trend'] = fund['gross_income_ratio'] - g['gross_income_ratio'].shift(4)
fund['inv_turn_change'] = fund['inventory_turnover_rate'] - g['inventory_turnover_rate'].shift(4)
fund['ar_load_change'] = fund['account_receivable_operate_revenue'] - g['account_receivable_operate_revenue'].shift(4)
fund['debt_change'] = fund['debt_assets_ratio'] - g['debt_assets_ratio'].shift(4)
fund['liquidity_gap'] = fund['quick_ratio'] - fund['current_ratio']
fund['liquidity_change'] = fund['liquidity_gap'] - g['liquidity_gap'].shift(4)

for c in ['retained_growth','cash_growth','roe_accel','rev_accel','roa_trend','gm_trend',
          'inv_turn_change','ar_load_change','debt_change','liquidity_change']:
    s = fund[c]
    q01, q99 = s.quantile([0.01, 0.99])
    fund[c] = s.clip(q01, q99)

# Paper spirit: sustainable growth should come from internally financed, profitable growth,
# not balance-sheet stretching. We proxy Higgins-style sustainable growth with retained-profit
# accumulation + profitability trends, then penalize leverage build-up and receivable loading.
sustainable_core = (
    0.26 * np.tanh(fund['retained_growth'].fillna(0) / 0.22) +
    0.16 * np.tanh(fund['cash_growth'].fillna(0) / 0.20) +
    0.20 * np.tanh(fund['roe_accel'].fillna(0) / 1.8) +
    0.14 * np.tanh(fund['roa_trend'].fillna(0) / 1.2) +
    0.14 * np.tanh(fund['gm_trend'].fillna(0) / 4.5) +
    0.10 * np.tanh(fund['rev_accel'].fillna(0) / 12.0)
)
quality_guard = (
    0.10 * np.tanh(fund['inv_turn_change'].fillna(0) / 1.6) -
    0.16 * np.tanh(fund['ar_load_change'].fillna(0) / 4.0) -
    0.14 * np.tanh(fund['debt_change'].fillna(0) / 4.0) +
    0.08 * np.tanh((-fund['liquidity_change']).fillna(0) / 0.12)
)

fund['raw_factor'] = sustainable_core + quality_guard
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
