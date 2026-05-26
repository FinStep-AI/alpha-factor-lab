import pandas as pd, numpy as np
BASE='data'
raw  = pd.read_csv(f'{BASE}/csi1000_kline_raw.csv',    parse_dates=['date']).sort_values(['stock_code','date']).reset_index(drop=True)
ret  = raw.pivot_table(index='date', columns='stock_code', values='pct_change').sort_index()/100
turn = raw.pivot_table(index='date', columns='stock_code', values='turnover').sort_index()
ret_series   = raw.pivot_table(index='date', columns='stock_code', values='pct_change').sort_index()/100
fwd_ret = ret_series.fillna(0).rolling(5, min_periods=2).sum().shift(-5)

def corr_vals(fac, fwd, step=5):
    dates = sorted(fac.index.intersection(fwd.index))[::step]
    out = []
    for d in dates:
        f = fac.loc[d].dropna()
        r = fwd.loc[d].dropna()
        s = f.index.intersection(r.index)
        if len(s) < 300:
            continue
        vals = np.corrcoef(f[s].values, r[s].values)
        out.append(float(vals[0, 1]))
    v = np.asarray(out)
    return float(v.mean()), float(v.std()), int(len(v)), float((v > 0).mean())

N = 20
cands = [
    ('neg_frac20', ret_series.clip(upper=0).rolling(N, min_periods=N // 2).count() / ret_series.rolling(N, min_periods=N // 2).count()),
    ('neg_cum20',  ret_series.clip(upper=0).rolling(N, min_periods=N // 2).sum()),
    ('turn20',     turn.fillna(0).rolling(N, min_periods=N // 2).mean()),
]

for name, fac in cands:
    fac = fac.reindex_like(ret_series)
    m, s, n, p = corr_vals(fac, fwd_ret)
    print(f'{name}: ic={m:.4f} std={s:.3f} n={n} pos={p:.2%}')
