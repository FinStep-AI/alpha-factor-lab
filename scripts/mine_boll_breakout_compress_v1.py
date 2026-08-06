import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
DATA = ROOT / 'data'
OUT = DATA / 'factor_boll_breakout_compress_v1.csv'

k = pd.read_csv(DATA / 'csi1000_kline_raw.csv')
k['date'] = pd.to_datetime(k['date'])
k = k.sort_values(['stock_code','date']).copy()

# 价格/量能特征
k['typical_price'] = (k['high'] + k['low'] + k['close']) / 3
k['vwap_proxy'] = k['amount'] / k['volume'].replace(0, np.nan)
# amount单位未知，但与volume相除仍是价格量纲，异常值回退到typical_price
mask_bad = ~np.isfinite(k['vwap_proxy']) | (k['vwap_proxy'] <= 0)
k.loc[mask_bad, 'vwap_proxy'] = k.loc[mask_bad, 'typical_price']

# 用amount近似自由流通市值代理，做横截面中性化
k['log_mktcap_proxy'] = np.log(k['close'].clip(lower=0.1) * k['amount'].clip(lower=1))

# 布林压缩 + 突破确认
by = k.groupby('stock_code', group_keys=False)
k['ma20'] = by['close'].transform(lambda s: s.rolling(20, min_periods=15).mean())
k['std20'] = by['close'].transform(lambda s: s.rolling(20, min_periods=15).std())
k['boll_width'] = (4 * k['std20']) / k['ma20'].replace(0, np.nan)
k['bw_pct_60'] = by['boll_width'].transform(lambda s: s.rolling(60, min_periods=30).rank(pct=True))

# 收盘位置，越接近日内高点越像突破确认
k['close_pos'] = (k['close'] - k['low']) / (k['high'] - k['low']).replace(0, np.nan)
# VWAP 偏强：收盘高于成交重心，代表尾盘更强
k['vwap_strength'] = (k['close'] - k['vwap_proxy']) / k['vwap_proxy'].replace(0, np.nan)
# 量能温和放大，避免纯脉冲巨量
k['vol_ratio20'] = k['amount'] / by['amount'].transform(lambda s: s.rolling(20, min_periods=10).mean())
k['vol_score'] = -np.abs(np.log(k['vol_ratio20'].clip(lower=1e-6)))
# 最近5日小趋势确认
k['ret_5d'] = by['close'].transform(lambda s: s.pct_change(5))

raw = (
    -0.45 * k['bw_pct_60'] +
     0.25 * k['close_pos'] +
     0.20 * k['vwap_strength'] +
     0.15 * k['vol_score'] +
     0.15 * k['ret_5d']
)
k['raw_factor'] = raw.replace([np.inf,-np.inf], np.nan)

# 横截面 winsorize + 市值中性化 + zscore
parts = []
for date, g in k.groupby('date'):
    g = g[['date','stock_code','raw_factor','log_mktcap_proxy']].copy()
    x = g['raw_factor']
    med = x.median()
    mad = (x - med).abs().median()
    if pd.isna(mad) or mad == 0:
        g['factor'] = np.nan
        parts.append(g[['date','stock_code','factor']])
        continue
    lo, hi = med - 3*1.4826*mad, med + 3*1.4826*mad
    y = x.clip(lo, hi)
    cap = g['log_mktcap_proxy']
    valid = y.notna() & cap.notna() & np.isfinite(y) & np.isfinite(cap)
    resid = pd.Series(np.nan, index=g.index)
    if valid.sum() >= 20:
        X = np.column_stack([np.ones(valid.sum()), cap[valid].values])
        beta = np.linalg.lstsq(X, y[valid].values, rcond=None)[0]
        resid.loc[valid] = y[valid].values - X @ beta
    else:
        resid = y
    mu = resid.mean()
    sd = resid.std()
    g['factor'] = (resid - mu) / (sd if pd.notna(sd) and sd > 0 else np.nan)
    parts.append(g[['date','stock_code','factor']])

factor = pd.concat(parts, ignore_index=True).dropna()
factor.to_csv(OUT, index=False)
print(f'saved {OUT} rows={len(factor)} dates={factor.date.nunique()} stocks={factor.stock_code.nunique()}')
print(factor.head().to_string())
