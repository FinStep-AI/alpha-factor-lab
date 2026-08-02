#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""quality_reinvestment_gap_v1

论文语义来源：质量/保守投资(Quality Minus Junk, Asness et al. 2013) 的本土化极简代理。
在当前数据仅有 ROE/BPS 的限制下，用“盈利改善 - 权益扩张”刻画：
- ROE 同比改善 = 盈利质量/成长改善
- BPS 同比扩张 = 更激进的再投资/净资产扩张代理
因子含义：更高的盈利改善、配合更克制的权益扩张，更可能对应高质量成长。

公式：raw = tanh((ROE_t - ROE_t-4)/6) - 0.8 * tanh((BPS_t / BPS_t-4 - 1)/0.3)
映射：财报可用日 = report_date + 45 天
中性化：对 20 日对数成交额做横截面 OLS 中性化
标准化：残差截面 z-score
"""

import numpy as np
import pandas as pd
from pathlib import Path

WORKDIR = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
OUT = WORKDIR / 'data/factor_quality_reinvestment_gap_v1.csv'

k = pd.read_csv(WORKDIR / 'data/csi1000_kline_raw.csv')
k['date'] = pd.to_datetime(k['date'])
k['stock_code'] = k['stock_code'].astype(str)
k['log_amount_20d'] = k.groupby('stock_code')['amount'].transform(
    lambda s: np.log(s.rolling(20, min_periods=10).mean().clip(lower=1))
)
base = k[['date', 'stock_code', 'log_amount_20d']].copy()

f = pd.read_csv(WORKDIR / 'data/csi1000_fundamental_cache.csv')
f['stock_code'] = f['stock_code'].astype(str)
f['report_date'] = pd.to_datetime(f['report_date'])
f = f.sort_values(['stock_code', 'report_date']).copy()
g = f.groupby('stock_code')
f['roe_lag4'] = g['roe'].shift(4)
f['bps_lag4'] = g['bps'].shift(4)
f['roe_yoy'] = f['roe'] - f['roe_lag4']
f['bps_yoy'] = f['bps'] / f['bps_lag4'] - 1
f['raw'] = np.tanh(f['roe_yoy'] / 6) - 0.8 * np.tanh(f['bps_yoy'] / 0.3)
f['avail_date'] = f['report_date'] + pd.Timedelta(days=45)

mapped = pd.merge_asof(
    base.sort_values('date'),
    f[['stock_code', 'avail_date', 'raw']].sort_values('avail_date'),
    left_on='date', right_on='avail_date', by='stock_code', direction='backward'
)
daily = mapped[['date', 'stock_code', 'raw', 'log_amount_20d']].copy().dropna()

def neutralize(grp: pd.DataFrame) -> pd.Series:
    g = grp.dropna().copy()
    if len(g) < 20:
        return pd.Series(np.nan, index=grp.index)
    X = np.column_stack([np.ones(len(g)), g['log_amount_20d'].values])
    y = g['raw'].values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    out = pd.Series(np.nan, index=grp.index)
    out.loc[g.index] = resid
    return out

def zscore(grp: pd.DataFrame) -> pd.Series:
    s = grp['resid']
    if s.notna().sum() < 20 or s.std() == 0:
        return pd.Series(np.nan, index=grp.index)
    z = (s - s.mean()) / s.std()
    lo, hi = z.quantile(0.01), z.quantile(0.99)
    return z.clip(lo, hi)

daily['resid'] = daily.groupby('date', group_keys=False).apply(neutralize)
daily['factor'] = daily.groupby('date', group_keys=False).apply(zscore)
out = daily[['date', 'stock_code', 'factor']].dropna().copy()
out['date'] = out['date'].dt.strftime('%Y-%m-%d')
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)
print(f'saved {OUT} rows={len(out)}')
