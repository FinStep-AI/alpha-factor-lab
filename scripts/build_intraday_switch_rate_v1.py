#!/usr/bin/env python3
"""
Intraday Trend Continuity Rate v1
构造：20日内收益率符号改变频率 lows = 趋势连续性强
信号：低切换率 = 信号聚合 = 有秩序的价格演化 → 次日反转？

逻辑：
  sign(close_t/open_t - 1) vs sign(close_{t-1}/open_{t-1} - 1)
  连续一致 → 信息线性积累 → 次日延续
  频繁切换 → 信息模糊 → 次日反转
  先测原始方向（高连续率=高收益），不达标再反转
"""

import numpy as np, pandas as pd

print("Building intraday_switch_rate_v1 (sign change freq)...")

df = pd.read_csv('data/csi1000_kline_raw.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.dropna(subset=['close','open','amount'])
df['close'] = df['close'].clip(lower=0.01)
df['open']  = df['open'].clip(lower=0.01)
df['amount'] = df['amount'].clip(lower=0)
amt_q99 = df['amount'].quantile(0.99)
df['log_amount'] = np.log(df['amount'].clip(1, amt_q99))

# intraday return sign (continuity vs flip)
df['intra_sig'] = (df['close'] - df['open']).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
# shift by 1 day within each stock
df['intra_sig_prev'] = df.groupby('stock_code')['intra_sig'].shift(1)
# flip = (current != prev), 0 if same, 1 if different
df['flip'] = (df['intra_sig'] != df['intra_sig_prev']).astype(float)

# 20d flip rate (lower = more continuous)
df['factor_raw'] = df.groupby('stock_code')['flip'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

df = df.dropna(subset=['factor_raw','log_amount'])

# Cross-section OLS log_amount + MAD + z
rows = []
for dt, sg in df.groupby('date', sort=True):
    x = sg['log_amount'].values.reshape(-1,1)
    y = sg['factor_raw'].values
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    if valid.sum() < 30: continue
    try:
        coef = np.linalg.lstsq(x[valid], y[valid], rcond=None)[0][0]
    except Exception: continue
    resid = y[valid] - x[valid].flatten()*coef
    med = np.nanmedian(resid)
    mad = np.nanmedian(np.abs(resid - med))
    if mad < 1e-10: continue
    u, lo = med + 3*1.4826*mad, med - 3*1.4826*mad
    rc = np.clip(resid, lo, u)
    mu, sigma = np.nanmean(rc), np.nanstd(rc)
    if sigma < 1e-10: continue
    z = (rc - mu) / sigma
    idxs = sg.index[valid]
    for pos, idx in enumerate(idxs):
        rows.append({'date': dt, 'stock_code': sg.at[idx,'stock_code'], 'factor': float(z[pos])})

out = pd.DataFrame(rows)
out.to_csv('data/factor_intraday_switch_rate_v1.csv', index=False)
print(f"Saved {len(out)} rows")
if len(out) > 0:
    print(out['factor'].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]))
