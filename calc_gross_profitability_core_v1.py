import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
FACTOR_ID = 'gross_profitability_core_v1'
OUT = BASE / f'data/factor_{FACTOR_ID}.csv'

fund = pd.read_csv(
    BASE/'data/csi1000_revenue_yoy_raw.csv',
    usecols=[
        'stock_code','end_date','info_publ_date','gross_income_ratio','gross_income_ratio_yoy',
        'operate_revenue_yoy','account_receivable_operate_revenue','inventory_turnover_rate'
    ]
)
kline = pd.read_csv(BASE/'data/csi1000_kline_raw.csv', usecols=['date','stock_code','close','amount','turnover'])

fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
fund['report_date'] = pd.to_datetime(fund['end_date'])
fund['avail_date'] = pd.to_datetime(fund['info_publ_date'])
for c in ['gross_income_ratio','gross_income_ratio_yoy','operate_revenue_yoy','account_receivable_operate_revenue','inventory_turnover_rate']:
    fund[c] = pd.to_numeric(fund[c], errors='coerce')
fund = fund.sort_values(['stock_code','report_date']).drop_duplicates(['stock_code','report_date'], keep='last')
g = fund.groupby('stock_code')

fund['gp_ma4'] = g['gross_income_ratio'].transform(lambda s: s.rolling(4, min_periods=3).mean())
fund['gp_yoy_ma2'] = g['gross_income_ratio_yoy'].transform(lambda s: s.rolling(2, min_periods=2).mean())
fund['rev_yoy_ma2'] = g['operate_revenue_yoy'].transform(lambda s: s.rolling(2, min_periods=2).mean())
fund['ar_ratio_ma2'] = g['account_receivable_operate_revenue'].transform(lambda s: s.rolling(2, min_periods=2).mean())
fund['inv_turn_ma2'] = g['inventory_turnover_rate'].transform(lambda s: s.rolling(2, min_periods=2).mean())
fund['inv_turn_delta'] = fund['inventory_turnover_rate'] - g['inventory_turnover_rate'].shift(1)

for c in ['gp_ma4','gp_yoy_ma2','rev_yoy_ma2','ar_ratio_ma2','inv_turn_ma2','inv_turn_delta']:
    s = fund[c]
    q01, q99 = s.quantile([0.01, 0.99])
    fund[c] = s.clip(q01, q99)

# 更贴近 gross profitability 主线：高毛利 + 毛利改善 + 收入增长确认；
# 对应收高占比给予惩罚，用存货周转做轻微确认。
fund['raw_factor'] = (
    0.50 * np.tanh(fund['gp_ma4'] / 18.0) +
    0.25 * np.tanh(fund['gp_yoy_ma2'] / 20.0) +
    0.20 * np.tanh(fund['rev_yoy_ma2'] / 30.0) -
    0.20 * np.tanh((fund['ar_ratio_ma2'].fillna(0) - 10.0) / 8.0) +
    0.10 * np.tanh(fund['inv_turn_delta'].fillna(0) / 1.5)
)

fund = fund.replace([np.inf, -np.inf], np.nan).dropna(subset=['raw_factor', 'avail_date'])
factor_q = fund[['stock_code','avail_date','raw_factor']].rename(columns={'avail_date':'date'})

kline['date'] = pd.to_datetime(kline['date'])
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline = kline.sort_values(['stock_code','date']).drop_duplicates(['date','stock_code'])
kline['mktcap_proxy'] = kline['close'].clip(lower=0.01) * kline['amount'].clip(lower=1) / (kline['turnover'].replace(0, np.nan) + 1e-6)
kline['log_mktcap'] = np.log(kline['mktcap_proxy'].clip(lower=1))
trade_dates = pd.Index(sorted(kline['date'].unique()))

parts = []
for stock, grp in factor_q.groupby('stock_code'):
    sf = grp[['date','raw_factor']].drop_duplicates('date', keep='last').set_index('date').sort_index()
    sf = sf.reindex(trade_dates, method='ffill', limit=80)
    sf['stock_code'] = stock
    sf = sf.dropna(subset=['raw_factor']).reset_index().rename(columns={'index':'date'})
    parts.append(sf)
factor = pd.concat(parts, ignore_index=True)
factor = factor.merge(kline[['date','stock_code','log_mktcap']], on=['date','stock_code'], how='inner').dropna(subset=['raw_factor','log_mktcap'])

out=[]
for date, grp in factor.groupby('date'):
    y = grp['raw_factor'].values.astype(float)
    x = grp['log_mktcap'].values.astype(float)
    mask = np.isfinite(y)&np.isfinite(x)
    if mask.sum() < 30:
        continue
    X = np.column_stack([np.ones(mask.sum()), x[mask]])
    beta = np.linalg.lstsq(X, y[mask], rcond=None)[0]
    resid = y[mask] - X @ beta
    med = np.median(resid)
    mad = np.median(np.abs(resid-med))
    if mad < 1e-12:
        continue
    resid = np.clip(resid, med-5*mad, med+5*mad)
    std = resid.std()
    if std < 1e-12:
        continue
    z = (resid - np.median(resid)) / std
    sub = grp.loc[mask, ['date','stock_code']].copy()
    sub['factor'] = z
    out.append(sub)

result = pd.concat(out, ignore_index=True)
result['date'] = pd.to_datetime(result['date']).dt.strftime('%Y-%m-%d')
result.to_csv(OUT, index=False, float_format='%.6f')
print(f'saved {OUT} rows={len(result)} unique_dates={result.date.nunique()}')
