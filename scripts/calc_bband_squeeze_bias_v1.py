#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KLINE = ROOT / 'data' / 'csi1000_kline_raw.csv'
OUT = ROOT / 'data' / 'factor_bband_squeeze_bias_v1.csv'

k = pd.read_csv(KLINE)
k['date'] = pd.to_datetime(k['date'])
k = k.sort_values(['stock_code','date']).copy()

# price/flow features
k['ret1'] = k.groupby('stock_code')['close'].pct_change()
k['log_mktcap_proxy'] = np.log((k['close'].clip(lower=0.01) * k['amount'].clip(lower=1.0)).replace(0, np.nan))

# rolling bollinger-style stats
by = k.groupby('stock_code', group_keys=False)
k['ma20'] = by['close'].transform(lambda s: s.rolling(20, min_periods=15).mean())
k['std20'] = by['close'].transform(lambda s: s.rolling(20, min_periods=15).std())
k['bandwidth'] = (4.0 * k['std20']) / (k['ma20'].abs() + 1e-6)
k['bw_pct60'] = by['bandwidth'].transform(lambda s: s.rolling(60, min_periods=30).rank(pct=True))
k['z_close_ma'] = (k['close'] - k['ma20']) / (k['std20'] + 1e-6)
k['turnover20'] = by['turnover'].transform(lambda s: s.rolling(20, min_periods=15).mean())
k['amount20'] = by['amount'].transform(lambda s: s.rolling(20, min_periods=15).mean())
k['ret5'] = by['close'].transform(lambda s: s.pct_change(5))
k['ret20'] = by['close'].transform(lambda s: s.pct_change(20))

# core idea: in squeeze state, directional bias + quiet participation predicts delayed continuation / release
compress = 1.0 - k['bw_pct60']
trend_align = np.tanh(k['ret20'] / 0.12)
bias = np.tanh(k['z_close_ma'] / 2.0)
turn_confirm = np.tanh(np.log1p(k['turnover20'].clip(lower=0)) / 2.5)
raw = compress * (0.55 * bias + 0.30 * trend_align + 0.15 * turn_confirm)

# winsorize by date
k['raw'] = raw.replace([np.inf, -np.inf], np.nan)
def winsor(s):
    if s.notna().sum() < 10:
        return s
    q1, q99 = s.quantile([0.01, 0.99])
    return s.clip(q1, q99)
k['raw_w'] = k.groupby('date')['raw'].transform(winsor)

# market-cap neutralization per date

def neutralize_date(df):
    x = df['log_mktcap_proxy']
    y = df['raw_w']
    valid = x.notna() & y.notna()
    out = pd.Series(np.nan, index=df.index)
    if valid.sum() < 20:
        return out
    xv = x[valid].values
    yv = y[valid].values
    X = np.column_stack([np.ones(len(xv)), xv])
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    resid = yv - X @ beta
    if resid.std() > 1e-8:
        resid = (resid - resid.mean()) / resid.std()
    else:
        resid = resid * 0
    out.loc[valid[valid].index] = resid
    return out

k['factor'] = k.groupby('date', group_keys=False).apply(neutralize_date).reset_index(level=0, drop=True)
res = k[['date','stock_code','factor']].dropna().copy()
res['date'] = res['date'].dt.strftime('%Y-%m-%d')
res.to_csv(OUT, index=False)
print(f'wrote {OUT} rows={len(res)}')
print(res.head())
