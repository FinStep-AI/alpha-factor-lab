#!/usr/bin/env python3
"""Close High Location v1"""
import numpy as np, pandas as pd

df = pd.read_csv('data/csi1000_kline_raw.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.dropna(subset=['close','high','low','amount'])

# Basic cleaning
df['close'] = df['close'].clip(lower=0.01)
df['high']  = df['high'].clip(lower=0.01)
df['low']   = df['low'].clip(lower=0.01)
df['amount'] = df['amount'].clip(lower=0)

# log amount
amt_q99 = df['amount'].quantile(0.99)
df['log_amount'] = np.log(df['amount'].clip(1, amt_q99))

# Raw factor: (close-low)/(high-low) 20d MA
df['cl_loc'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
# Clip extreme raw values
df['cl_loc'] = df['cl_loc'].clip(-1.0, 2.0)

df['factor_raw'] = df.groupby('stock_code')['cl_loc'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

# Remove NaN factor_raw
df = df.dropna(subset=['factor_raw','log_amount'])

# Cross-section neutralization (OLS log_amount → MAD winsorize → z-score)
out_rows = []
for dt, sg in df.groupby('date', sort=True):
    n = len(sg)
    if n < 30: continue
    log_a = sg['log_amount'].values
    raw_f = sg['factor_raw'].values
    # Remove rows with any inf/nan
    valid = np.isfinite(log_a) & np.isfinite(raw_f) & (np.abs(log_a) < 100) & (np.abs(raw_f) < 100)
    if valid.sum() < 30: continue
    x = log_a[valid].reshape(-1,1)
    y = raw_f[valid]
    try:
        coef = np.linalg.lstsq(x, y, rcond=None)[0][0]
    except Exception:
        continue
    resid = y - x.flatten()*coef
    med = np.nanmedian(resid)
    mad = np.nanmedian(np.abs(resid - med))
    if mad < 1e-10: continue
    u, lo = med + 3*1.4826*mad, med - 3*1.4826*mad
    rc = np.clip(resid, lo, u)
    mu = np.nanmean(rc); sigma = np.nanstd(rc)
    if sigma < 1e-10: continue
    z = (rc - mu) / sigma
    idx_list = sg.index[valid].tolist()
    for pos, idx in enumerate(idx_list):
        out_rows.append({'date': dt, 'stock_code': sg.at[idx,'stock_code'], 'factor': float(z[pos])})

out = pd.DataFrame(out_rows)
out.to_csv('data/factor_close_high_v1.csv', index=False)
print(f"Saved {len(out)} rows")
if len(out) > 0:
    print(out['factor'].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]))
