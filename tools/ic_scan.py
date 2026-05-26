import pandas as pd, numpy as np

BASE = 'data'
raw  = pd.read_csv(f'{BASE}/csi1000_kline_raw.csv', parse_dates=['date']).sort_values(['stock_code','date']).reset_index(drop=True)
ret  = raw.pivot_table(index='date', columns='stock_code', values='pct_change').sort_index() / 100
turn = raw.pivot_table(index='date', columns='stock_code', values='turnover').sort_index()
amt  = raw.pivot_table(index='date', columns='stock_code', values='amount').sort_index()

fwd_ret_5  = ret.fillna(0).rolling(5,  min_periods=2).sum().shift(-5)
fwd_ret_10 = ret.fillna(0).rolling(10, min_periods=5).sum().shift(-10)

def corr_vals(fac: pd.DataFrame, fwd: pd.DataFrame, step: int = 5):
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
    if len(v) == 0:
        return None, None, 0, None
    return float(v.mean()), float(v.std()), int(len(v)), float((v > 0).mean())

# 1) roll-win negative-return candidates over several horizons x 2 forward horizons
windows = [5, 10, 20, 40, 60]
fwd = [('f5', fwd_ret_5), ('f10', fwd_ret_10)]
for w in windows:
    mp = max(2, w // 2)
    fac_neg_cum_w   = ret.clip(upper=0).rolling(w, min_periods=mp).sum().reindex_like(ret)
    fac_neg_frac_w  = ret.clip(upper=0).rolling(w, min_periods=mp).count() / ret.rolling(w, min_periods=mp).count()
    fac_mean_ret_w  = ret.rolling(w, min_periods=mp).mean()
    fac_skew_w      = ret.rolling(w, min_periods=mp).skew()
    fac_turn_w      = np.log(turn.clip(lower=1e-9).rolling(w, min_periods=mp).mean() + 1)
    fac_amt_w       = np.log(amt.clip(lower=1e-9).rolling(w, min_periods=mp).sum()  + 1)
    for fnm, fr in fwd:
        rows = [
            ('neg_cum',   fac_neg_cum_w),
            ('neg_frac',  fac_neg_frac_w),
            ('mean_ret',  fac_mean_ret_w),
            ('skew',      fac_skew_w),
            ('l_turn',    fac_turn_w),
            ('l_amt20',   fac_amt_w),
        ]
        for nm, fac in rows:
            m, s, n, p = corr_vals(fac, fr)
            if m is None:
                continue
            print(f'{nm}{w}_{fnm}: ic={m:.4f} std={s:.3f} n={n} pos={p:.2%}')

# 2) last-week directional persistence vs mean of prior 4 weeks
prior4 = ret.rolling(20, min_periods=10).mean()  # ~4-week trend
lw = ret.fillna(0).rolling(5, min_periods=3).sum()
for name, fac in [
        ('lw-ret', lw),
        ('lw-vs-prior4', (lw - prior4).clip(-0.5, 0.5)),
        ('lw-dir-cross', (np.sign(lw)*np.sign(prior4)).rolling(1).mean()),
]:
    for fnm, fr in fwd:
        fac = fac.reindex_like(ret)
        m, s, n, p = corr_vals(fac, fr)
        if m is None:
            continue
        print(f'{name}_{fnm}: ic={m:.4f} std={s:.3f} n={n} pos={p:.2%}')
