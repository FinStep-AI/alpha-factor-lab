#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path('data/factor_roe_accel_bps_guard_growth_v2.csv')

k = pd.read_csv('data/csi1000_kline_raw.csv', usecols=['date','stock_code','close'])
k['date'] = pd.to_datetime(k['date'])
k['stock_code'] = k['stock_code'].astype(str).str.zfill(6)

f = pd.read_csv('data/csi1000_fundamental_cache.csv')
f['report_date'] = pd.to_datetime(f['report_date'])
f['stock_code'] = f['stock_code'].astype(str).str.zfill(6)
f = f.sort_values(['stock_code','report_date']).copy()

# 基本面派生：ROE一阶变化、二阶加速度、BPS扩张约束、稳定性惩罚
f['roe_qoq'] = f.groupby('stock_code')['roe'].diff(1)
f['roe_accel'] = f.groupby('stock_code')['roe_qoq'].diff(1)
f['bps_growth'] = f.groupby('stock_code')['bps'].pct_change(1, fill_method=None)
f['roe_vol_4q'] = f.groupby('stock_code')['roe_qoq'].rolling(4, min_periods=2).std().reset_index(level=0, drop=True)

# winsor helper

def robust_z(s):
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad == 0:
        std = s.std(ddof=0)
        return (s - s.mean()) / std if std and not pd.isna(std) else s * np.nan
    return (s - med) / (1.4826 * mad)

# Growth主信号：ROE加速度高、BPS温和扩张、ROE变化不太抖
f['raw_factor'] = (
    f['roe_accel'] * (1 + f['bps_growth'].clip(-0.2, 0.3)) / (1 + f['roe_vol_4q'].fillna(f['roe_vol_4q'].median()).abs())
)

# 用 close*bps 代理市值，按财报日向后对齐到交易日，再做横截面市值中性化
k2 = k.sort_values(['date','stock_code']).reset_index(drop=True)
f2 = f[['stock_code','report_date','bps','raw_factor']].sort_values(['report_date','stock_code']).reset_index(drop=True)
panel = pd.merge_asof(
    k2,
    f2,
    left_on='date', right_on='report_date', by='stock_code', direction='backward'
)
panel = panel[['date','stock_code','close','bps','raw_factor']].dropna(subset=['raw_factor','bps','close']).copy()
panel['mcap_proxy'] = (panel['close'].clip(lower=0.01) * panel['bps'].clip(lower=0.01)).replace([np.inf,-np.inf], np.nan)
panel = panel.dropna(subset=['mcap_proxy'])
panel['log_mcap'] = np.log(panel['mcap_proxy'])


def neutralize_day(g):
    x = g['raw_factor'].astype(float)
    m = g['log_mcap'].astype(float)
    mask = x.notna() & m.notna()
    if mask.sum() < 20:
        return pd.Series(np.nan, index=g.index)
    xv = x[mask]
    mv = m[mask]
    # winsor
    lo, hi = xv.quantile(0.01), xv.quantile(0.99)
    xv = xv.clip(lo, hi)
    A = np.column_stack([np.ones(len(mv)), mv.values])
    beta = np.linalg.lstsq(A, xv.values, rcond=None)[0]
    resid = xv.values - A @ beta
    out = pd.Series(np.nan, index=g.index)
    out.loc[mask] = resid
    # cross-sectional z-score
    rr = out.loc[mask]
    std = rr.std(ddof=0)
    if std and not np.isnan(std):
        out.loc[mask] = (rr - rr.mean()) / std
    return out

panel['factor_value'] = panel.groupby('date', group_keys=False).apply(neutralize_day)
res = panel[['date','stock_code','factor_value']].dropna().copy()
res['date'] = res['date'].dt.strftime('%Y-%m-%d')
OUT.parent.mkdir(parents=True, exist_ok=True)
res.to_csv(OUT, index=False)
print('saved', OUT, 'rows', len(res), 'dates', res['date'].nunique(), 'stocks', res['stock_code'].nunique())
print(res.head())
