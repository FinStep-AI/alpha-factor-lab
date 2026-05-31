"""
Probe MA60 discount factor, then write factor CSV, then call backtest.
Usage:  python3 tmp_probe_ma60_disc.py
"""
import pandas as pd, numpy as np, os, sys, json
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings('ignore')

BASE = '/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab'
os.chdir(BASE)

# ─── 1. load data ───────────────────────────────────────────────────────────
kline = pd.read_csv('data/csi1000_kline_raw.csv', parse_dates=['date'])
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline = (kline.sort_values(['stock_code','date'])
               .drop_duplicates(['date','stock_code']).reset_index(drop=True))

rets = pd.read_csv('data/csi1000_returns.csv', parse_dates=['date'])
rets['stock_code'] = rets['stock_code'].astype(str).str.zfill(6)
rets = (rets.rename(columns={'return':'ret'})
           .drop_duplicates(['date','stock_code']).reset_index(drop=True))

# align dates
common = sorted(set(kline['date']) & set(rets['date']))
kline = kline[kline['date'].isin(common)].drop_duplicates(['date','stock_code']).reset_index(drop=True)
rets  = rets [rets['date'].isin(common)].drop_duplicates(['date','stock_code']).reset_index(drop=True)

print(f'kline {len(kline):,} rows  rets {len(rets):,} rows  dates {kline["date"].nunique()}')

# ─── 2. factor construction ─────────────────────────────────────────────────
for w in [5,20,60]:
    kline[f'ma{w}'] = kline.groupby('stock_code')['close'].transform(
        lambda x: x.rolling(w, min_periods=max(2, w//3)).mean())

den = lambda s: s.abs().clip(lower=1e-8)
kline['dev_ma5']  = (kline['close'] - kline['ma5'])  / den(kline['ma5'])
kline['dev_ma20'] = (kline['close'] - kline['ma20']) / den(kline['ma20'])
kline['dev_ma60'] = (kline['close'] - kline['ma60']) / den(kline['ma60'])

# Core candidate signals  -------------------
# A) neg_dev_ma20_raw – shortest horizon (price under MA20)
# B) neg_dev_ma60_sm5 – 5d smoothed negative MA60 deviation
# C) ma60_disc_rank_sm10 – cross-sectional rank of MA60 discount, smoothed 10d

kline['neg_dev_ma20_raw']   = -kline['dev_ma20']
kline['neg_dev_ma60_sm5']  = kline.groupby('stock_code')['-dev_ma60'.replace('-','')].transform(
    lambda x: 0.0*x)            # placeholder
kline['neg_dev_ma60_d']     = -kline['dev_ma60']
kline['neg_dev_ma60_sm5b'] = kline.groupby('stock_code')['neg_dev_ma60_d']\
    .transform(lambda x: x.rolling(5, min_periods=3).mean())

kline['disc_rank'] = kline.groupby('date')['neg_dev_ma60_d']\
    .rank(method='average', pct=True)
kline['ma60_disc_rank_sm10'] = kline.groupby('stock_code')['disc_rank']\
    .transform(lambda x: x.rolling(10, min_periods=5).mean())

# forward returns by horizon
rets2 = rets.sort_values(['stock_code','date']).reset_index(drop=True)
for n in [5,10,20]:
    rets2[f'ret_f{n}d'] = rets2.groupby('stock_code')['ret'].transform(
        lambda x: (1+x).shift(-1).rolling(n, min_periods=n).apply(np.prod, raw=True) - 1)

r5  = rets2[['date','stock_code','ret_f5d']].dropna(subset=['ret_f5d'])
r10 = rets2[['date','stock_code','ret_f10d']].dropna(subset=['ret_f10d'])
r20 = rets2[['date','stock_code','ret_f20d']].dropna(subset=['ret_f20d'])

scenarios = [
    ('neg_dev_ma20_raw',    r5,  'ret_f5d',  5),
    ('neg_dev_ma60_sm5b',   r5,  'ret_f5d',  5),
    ('neg_dev_ma60_sm5b',   r10, 'ret_f10d', 10),
    ('neg_dev_ma60_sm5b',   r20, 'ret_f20d', 20),
    ('ma60_disc_rank_sm10', r10, 'ret_f10d', 10),
    ('ma60_disc_rank_sm10', r20, 'ret_f20d', 20),
]

print('\n=== cross-sectional IC candidate tests ===')
best = (None, -9.0)
for col_name, ret_df, ret_col, fwd_n in scenarios:
    fac = kline[['date','stock_code', col_name]].drop_duplicates(['date','stock_code']).dropna(subset=[col_name])
    m = fac.merge(ret_df[['date','stock_code',ret_col]], on=['date','stock_code'], how='inner')\
           .drop_duplicates(['date','stock_code'])
    ics = []
    for dt, g in m.groupby('date', sort=False):
        if len(g) < 100: continue
        rv, _ = spearmanr(g[col_name].rank(method='average'), g[ret_col].rank(method='average'))
        if not np.isnan(rv): ics.append(rv)
    if not ics:
        print(f'  {col_name} -> {fwd_n}d : NO DATA'); continue
    a = np.array(ics)
    n, mu, sd = len(a), a.mean(), a.std()
    t = mu/sd*np.sqrt(n) if sd > 0 else np.nan
    pos = float((a>0).mean())
    print(f'  {col_name:<26} -> {fwd_n:2d}d  n={n:4d}  ic={mu:+.4f}  t={t:6.2f}  pos={pos:.3f}')
    if t > best[1]:
        best = ((col_name, ret_col, fwd_n, fac.copy(), t, mu, pos), t)

if best[1] < 2.0:
    print('\nno candidate cleared IC_t>=2.0; exit.')
    sys.exit(0)

col_name, ret_col, fwd_n, fac_best, bt, mu, bpos = best[0]
print(f'\n*** best candidate: {col_name} fwd={fwd_n}d t={bt:.2f} ic={mu:+.4f} pos={bpos:.3f} ***')

# ─── 3. write factor csv (simple z-score per cross-section, market-neutral) ─
fac_out = fac_best.rename(columns={col_name: 'factor_value'}).copy()
# drop duplicates again just in case
fac_out = fac_out.drop_duplicates(['date','stock_code'])
# cross-sectionally demean + z-score
fac_out['factor_value'] = fac_out.groupby('date')['factor_value']\
    .transform(lambda s: ((s - s.mean()) / s.std(ddof=0)) if s.std(ddof=0) > 0 else 0.0)

out_path = f'data/factor_ma60_depth_v1.csv'
fac_out[['date','stock_code','factor_value']].to_csv(out_path, index=False)
print(f'wrote {out_path}  rows={len(fac_out)}')

# ─── 4. quick one-shot backtest shell check ─────────────────────────────────
print(f'\n=== quick IC check on saved file ===')
df = fac_out.copy()
ret_merge = rets2[['date','stock_code','ret']].copy()
ics = []
for dt, g in df.groupby('date', sort=False):
    if len(g) < 100: continue
    m = g.merge(ret_merge[['date','stock_code','ret']], on=['date','stock_code'], how='inner')
    if len(m) < 100: continue
    rv, _ = spearmanr(m['factor_value'].rank(method='average'), m['ret'].rank(method='average'))
    if not np.isnan(rv): ics.append(rv)
a = np.array(ics)
print(f'next-day IC n={len(a)} ic_mean={a.mean():+.4f} t={a.mean()/a.std()*np.sqrt(len(a)):.2f} pos={float((a>0).mean()):.3f}')
