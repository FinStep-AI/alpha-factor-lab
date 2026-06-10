
import warnings, os, numpy as np, pandas as pd, sys
warnings.filterwarnings('ignore')
BASE = os.getcwd()

fund = pd.read_csv(os.path.join(BASE, 'data', 'csi1000_fundamental_cache.csv'))
fund['report_date'] = pd.to_datetime(fund['report_date'])
fund['stock_code']  = fund['stock_code'].astype(str).str.zfill(6)
fund = fund.sort_values(['stock_code', 'report_date']).drop_duplicates(['stock_code', 'report_date'], keep='last')

kline = pd.read_csv(os.path.join(BASE, 'data', 'csi1000_kline_raw.csv'), usecols=['date', 'stock_code', 'amount'])
kline['date']       = pd.to_datetime(kline['date'])
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
td_idx = pd.DatetimeIndex(sorted(kline['date'].unique()))

rpt2k  = {rd: td_idx[td_idx >= rd][0] for rd in sorted(fund['report_date'].unique()) if len(td_idx[td_idx >= rd]) > 0}
fund['kline_date'] = fund['report_date'].map(rpt2k)
kdates = sorted(fund['kline_date'].unique())

amt_day      = kline.groupby('date')['amount'].median()
day_amt_lmap = {}   # date -> dict(sc -> log_amount)

recs = []
for i, cd in enumerate(kdates):
    sub = fund[fund['kline_date'] == cd].dropna(subset=['roe'])[['stock_code', 'roe']].copy()
    sub = sub[sub['roe'].between(-80, 250)]
    if len(sub) < 300:
        continue
    mu, sig = sub['roe'].mean(), sub['roe'].std(ddof=0)
    if sig < 1e-8:
        continue
    sub['r_roe'] = (sub['roe'] - mu) / sig
    if i >= 4:
        prev_cd = kdates[i - 4]
        prev = (fund[fund['kline_date'] == prev_cd]
                .dropna(subset=['roe'])[['stock_code', 'roe']]
                .drop_duplicates('stock_code', keep='last')
                .rename(columns={'roe': 'rprev'})
                .set_index('stock_code')['rprev'])
        sub['rprev'] = sub['stock_code'].map(prev)
        sub = sub.dropna(subset=['rprev'])
        sub['raw'] = 0.55 * sub['r_roe'] + 0.45 * (sub['r_roe'] - sub['rprev'])
    else:
        sub['raw'] = sub['r_roe']
    for _, r2 in sub.iterrows():
        recs.append((str(cd.date()), str(int(r2['stock_code'])).zfill(6), float(r2['raw'])))

raw = pd.DataFrame(recs, columns=['date', 'stock_code', 'raw'])

out_rows = []
for dt, grp in raw.groupby('date', sort=True):
    amt = amt_day.get(pd.to_datetime(dt))
    if amt is None or np.isnan(amt):
        continue
    if dt not in day_amt_lmap:
        dd = kline[kline['date'] == pd.to_datetime(dt)][['stock_code', 'amount']].dropna()
        day_amt_lmap[dt] = dict(zip(dd['stock_code'], np.log(dd['amount'] + 1)))
    g = grp.copy()
    g['stock_code'] = g['stock_code'].astype(str).str.zfill(6)
    g['la'] = g['stock_code'].map(day_amt_lmap[dt])
    g = g.dropna(subset=['raw', 'la'])
    if len(g) < 100:
        continue
    X = np.column_stack([np.ones(len(g)), g['la'].values])
    y = g['raw'].values
    _, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ b
    med, std = float(np.median(resid)), float(np.std(resid, ddof=0))
    if std < 1e-10:
        continue
    g2 = g[['date', 'stock_code']].copy()
    g2['factor_value'] = (resid - med) / std
    out_rows.append(g2)

out = pd.concat(out_rows, ignore_index=True)

# cross-date two-pass median/MAD center (fixes 2022-10-10 residual)
med_d = out.groupby('date')['factor_value'].median()
mad_d = out.groupby('date')['factor_value'].apply(lambda s: float(np.median(np.abs(s - np.median(s)))))
out['factor_value'] = out.apply(
    lambda r: (r['factor_value'] - med_d[r['date']]) / (mad_d[r['date']] + 1e-10), axis=1)
med2 = out.groupby('date')['factor_value'].transform('median')
mad2 = out.groupby('date')['factor_value'].transform(lambda s: float(np.median(np.abs(s - np.median(s)))))
out['factor_value'] = np.where(mad2 > 1e-10, (out['factor_value'] - med2) / mad2, 0.0)

out[['date', 'stock_code', 'factor_value']].to_csv(
    os.path.join(BASE, 'data', 'factor_accrual_quality_v1.csv'), index=False)
print('written', len(out), 'rows')
print('per-date medians:')
print(out.groupby('date')['factor_value'].median().round(4).to_string())
