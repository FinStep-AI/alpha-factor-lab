#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROE-BPS 持续性扩散反转因子 v1

论文语义锚点：盈利持续性 / 成长质量。
在仅有 roe + bps 字段下，构造“盈利持续性 × 账面扩张一致性”的反向代理：
- 高ROE持续性若同时伴随高BPS扩张，可能代表已被过度定价的‘稳定成长’；
- 在中证1000里尝试反向捕捉困境反转/预期修正。

公式（季度层面）:
    roe_persist = mean(ROE_8q) / (std(ROE_8q) + 1)
    bps_growth  = BPS_t / BPS_t-4 - 1
    bps_persist = mean(BPS_yoy_4q) / (std(BPS_yoy_4q) + 0.05)
    raw = -(0.65 * z(roe_persist) + 0.35 * z(bps_persist) + 0.20 * z(max(bps_growth,0)))

然后：
- report_date + 45d 映射到日频
- 截面用 log(amount) 做市值代理中性化
- MAD缩尾 + z-score
输出: data/factor_roe_bps_persistence_spread_v1.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
KLINE = BASE / 'data' / 'csi1000_kline_raw.csv'
FUND = BASE / 'data' / 'csi1000_fundamental_cache.csv'
OUT = BASE / 'data' / 'factor_roe_bps_persistence_spread_v1.csv'
LAG = 45


def mad_winsorize(s, n=3.0):
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return s
    scale = 1.4826 * mad
    return s.clip(med - n * scale, med + n * scale)


def zscore(s):
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return s * np.nan
    return (s - s.mean()) / std


def ols_resid(y, x):
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 10:
        return np.full_like(y, np.nan, dtype=float)
    X = np.column_stack([np.ones(mask.sum()), x[mask]])
    beta = np.linalg.lstsq(X, y[mask], rcond=None)[0]
    resid = np.full_like(y, np.nan, dtype=float)
    resid[mask] = y[mask] - X @ beta
    return resid

print('[1] load data')
k = pd.read_csv(KLINE)
k['date'] = pd.to_datetime(k['date'])
k = k.sort_values(['stock_code','date'])

f = pd.read_csv(FUND)
f['report_date'] = pd.to_datetime(f['report_date'])
f = f.sort_values(['stock_code','report_date']).copy()

print('[2] build quarterly features')
def build_quarter(g):
    g = g.sort_values('report_date').copy()
    g['roe_mean8'] = g['roe'].rolling(8, min_periods=6).mean()
    g['roe_std8'] = g['roe'].rolling(8, min_periods=6).std(ddof=0)
    g['roe_persist'] = g['roe_mean8'] / (g['roe_std8'] + 1.0)

    g['bps_yoy'] = g['bps'] / g['bps'].shift(4) - 1
    g['bps_yoy_mean4'] = g['bps_yoy'].rolling(4, min_periods=3).mean()
    g['bps_yoy_std4'] = g['bps_yoy'].rolling(4, min_periods=3).std(ddof=0)
    g['bps_persist'] = g['bps_yoy_mean4'] / (g['bps_yoy_std4'] + 0.05)
    g['bps_growth_pos'] = g['bps_yoy'].clip(lower=0)
    return g

q = f.groupby('stock_code', group_keys=False).apply(build_quarter)
q['info_date'] = q['report_date'] + pd.Timedelta(days=LAG)
q = q[['stock_code','info_date','roe_persist','bps_persist','bps_growth_pos']].dropna(subset=['roe_persist'])

print('[3] map to daily via merge_asof')
frames = []
for sc, gk in k.groupby('stock_code'):
    fq = q[q['stock_code']==sc].sort_values('info_date')
    if fq.empty:
        continue
    merged = pd.merge_asof(
        gk.sort_values('date'),
        fq.sort_values('info_date'),
        left_on='date', right_on='info_date',
        by='stock_code',
        direction='backward'
    )
    frames.append(merged)

df = pd.concat(frames, ignore_index=True)

print('[4] cross-sectional factor construction')
res = []
for date, g in df.groupby('date'):
    gg = g[['date','stock_code','amount','roe_persist','bps_persist','bps_growth_pos']].copy()
    if len(gg) < 50:
        continue
    gg['roe_z'] = zscore(mad_winsorize(gg['roe_persist']))
    gg['bpsp_z'] = zscore(mad_winsorize(gg['bps_persist']))
    gg['bg_z'] = zscore(mad_winsorize(gg['bps_growth_pos']))
    gg['raw'] = -(0.65*gg['roe_z'] + 0.35*gg['bpsp_z'].fillna(0) + 0.20*gg['bg_z'].fillna(0))
    x = np.log(gg['amount'].clip(lower=1).astype(float).values)
    y = gg['raw'].astype(float).values
    gg['resid'] = ols_resid(y, x)
    gg['factor'] = zscore(mad_winsorize(pd.Series(gg['resid'], index=gg.index)))
    res.append(gg[['date','stock_code','factor']])

out = pd.concat(res, ignore_index=True).dropna()
out.to_csv(OUT, index=False)
print('[done]', OUT, 'rows=', len(out), 'dates=', out['date'].nunique())
