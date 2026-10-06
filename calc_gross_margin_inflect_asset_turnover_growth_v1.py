import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
FACTOR_ID = 'gross_margin_inflect_asset_turnover_growth_v1'
OUT = BASE / f'data/factor_{FACTOR_ID}.csv'

fund = pd.read_csv(BASE / 'data/csi1000_fundamental_cache.csv')
kline = pd.read_csv(BASE / 'data/csi1000_kline_raw.csv', usecols=['date', 'stock_code', 'close', 'amount', 'turnover'])

fund['report_date'] = pd.to_datetime(fund['report_date'])
fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
for col in ['roe', 'bps']:
    fund[col] = pd.to_numeric(fund[col], errors='coerce')
fund = fund.dropna(subset=['roe', 'bps']).sort_values(['stock_code', 'report_date']).drop_duplicates(['stock_code', 'report_date'])

# 用 DuPont 近似拆解：roe ~= net_margin * asset_turnover * leverage
# 缺真实营收/总资产时，用 bps 的变化近似权益扩张，并以 roe/bps 近似资产周转代理；
# 由此构造“利润率改善拐点 × 周转协同”的成长因子。
for col in ['roe', 'bps']:
    q01, q99 = fund[col].quantile([0.01, 0.99])
    fund[col] = fund[col].clip(q01, q99)

g = fund.groupby('stock_code')
fund['at_proxy'] = fund['roe'] / fund['bps'].abs().clip(lower=0.8)
fund['eq_growth_qoq'] = fund['bps'] / fund['bps'].shift(1).replace(0, np.nan) - 1.0
fund['eq_growth_yoy'] = fund['bps'] / fund['bps'].shift(4).replace(0, np.nan) - 1.0
fund['margin_proxy'] = fund['roe'] / (fund['at_proxy'].replace(0, np.nan))
# 代数上约等于 bps，但加入 winsor/时序差分后可作为利润率-资本扩张约束的复合近似
fund['margin_proxy'] = np.log1p(fund['bps'].clip(lower=0))

fund['gm_l1'] = g['margin_proxy'].shift(1)
fund['gm_l2'] = g['margin_proxy'].shift(2)
fund['gm_l4'] = g['margin_proxy'].shift(4)
fund['at_l1'] = g['at_proxy'].shift(1)
fund['at_l2'] = g['at_proxy'].shift(2)
fund['at_l4'] = g['at_proxy'].shift(4)
fund['roe_l1'] = g['roe'].shift(1)
fund['roe_l4'] = g['roe'].shift(4)

fund['gm_yoy'] = fund['margin_proxy'] - fund['gm_l4']
fund['gm_qoq'] = fund['margin_proxy'] - fund['gm_l1']
fund['gm_inflect'] = fund['gm_qoq'] - (fund['gm_l1'] - fund['gm_l2'])
fund['at_yoy'] = fund['at_proxy'] - fund['at_l4']
fund['at_qoq'] = fund['at_proxy'] - fund['at_l1']
fund['at_inflect'] = fund['at_qoq'] - (fund['at_l1'] - fund['at_l2'])
fund['roe_yoy'] = fund['roe'] - fund['roe_l4']
fund['roe_qoq'] = fund['roe'] - fund['roe_l1']
fund['gm_std4'] = g['gm_yoy'].transform(lambda s: s.rolling(4, min_periods=3).std())
fund['at_std4'] = g['at_yoy'].transform(lambda s: s.rolling(4, min_periods=3).std())
fund['roe_std4'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).std())

for col in ['at_proxy','margin_proxy','eq_growth_qoq','eq_growth_yoy','gm_yoy','gm_qoq','gm_inflect','at_yoy','at_qoq','at_inflect','roe_yoy','roe_qoq','gm_std4','at_std4','roe_std4']:
    s = fund[col]
    q01, q99 = s.quantile([0.01, 0.99])
    fund[col] = s.clip(q01, q99)

gm_trend = np.tanh(6.0 * fund['gm_yoy'].fillna(0))
gm_inflect = np.tanh(8.0 * fund['gm_inflect'].fillna(0))
at_trend = np.tanh(4.0 * fund['at_yoy'].fillna(0))
at_inflect = np.tanh(6.0 * fund['at_inflect'].fillna(0))
synergy = np.tanh(10.0 * (fund['gm_yoy'].fillna(0) * fund['at_yoy'].fillna(0)))
profit_confirm = np.tanh((0.7 * fund['roe_yoy'].fillna(0) + 0.3 * fund['roe_qoq'].fillna(0)) / 5.0)
stability = 1.0 / (1.0 + 1.6 * fund['gm_std4'].abs().fillna(0) + 1.2 * fund['at_std4'].abs().fillna(0) + 0.08 * fund['roe_std4'].abs().fillna(0))
capital_guard = 1.0 / (1.0 + 4.0 * fund['eq_growth_yoy'].clip(lower=0.20).fillna(0) + 2.0 * fund['eq_growth_qoq'].clip(lower=0.08).fillna(0))
level_gate = 0.65 + 0.35 * np.tanh(fund['roe'].fillna(0) / 8.0)

fund['raw_factor'] = (
    0.24 * gm_trend +
    0.22 * gm_inflect +
    0.18 * at_trend +
    0.14 * at_inflect +
    0.12 * synergy +
    0.10 * profit_confirm
) * stability * capital_guard * level_gate

fund = fund.replace([np.inf, -np.inf], np.nan).dropna(subset=['raw_factor'])
fund['avail_date'] = fund['report_date'] + pd.Timedelta(days=45)
factor_q = fund[['stock_code', 'avail_date', 'raw_factor']].rename(columns={'avail_date': 'date'})

kline['date'] = pd.to_datetime(kline['date'])
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline = kline.sort_values(['stock_code', 'date']).drop_duplicates(['date', 'stock_code'])
kline['mktcap_proxy'] = kline['close'].clip(lower=0.01) * kline['amount'].clip(lower=1) / (kline['turnover'].replace(0, np.nan) + 1e-6)
kline['log_mktcap'] = np.log(kline['mktcap_proxy'].clip(lower=1))
trade_dates = pd.Index(sorted(kline['date'].unique()))

parts = []
for stock, grp in factor_q.groupby('stock_code'):
    sf = grp[['date', 'raw_factor']].drop_duplicates('date', keep='last').set_index('date').sort_index()
    sf = sf.reindex(trade_dates, method='ffill', limit=80)
    sf['stock_code'] = stock
    sf = sf.dropna(subset=['raw_factor']).reset_index().rename(columns={'index': 'date'})
    parts.append(sf)

factor = pd.concat(parts, ignore_index=True)
factor = factor.merge(kline[['date', 'stock_code', 'log_mktcap']], on=['date', 'stock_code'], how='inner').dropna(subset=['raw_factor', 'log_mktcap'])

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
    resid = np.clip(resid, med - 5.0 * mad, med + 5.0 * mad)
    std = resid.std()
    if std < 1e-12:
        return np.full(len(vals), np.nan)
    z = (resid - np.median(resid)) / std
    out = np.full(len(vals), np.nan)
    out[np.where(mask)[0]] = z
    return out

outs = []
for date, grp in factor.groupby('date'):
    nz = neutralize(grp['raw_factor'].to_numpy(float), grp['log_mktcap'].to_numpy(float))
    good = np.isfinite(nz)
    if good.any():
        sub = grp.loc[good, ['date', 'stock_code']].copy()
        sub['factor'] = nz[good]
        outs.append(sub)

result = pd.concat(outs, ignore_index=True)
result['date'] = pd.to_datetime(result['date']).dt.strftime('%Y-%m-%d')
result.to_csv(OUT, index=False, float_format='%.6f')
print(f'saved {OUT} rows={len(result)} dates={result.date.min()}~{result.date.max()} stocks={result.stock_code.nunique()}')
