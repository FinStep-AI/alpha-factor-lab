import numpy as np, pandas as pd, sys, warnings
from scipy import stats
warnings.filterwarnings('ignore')

W = 20          # rolling window
OUT = 'data/gap_contrib_v1.csv'

k = pd.read_csv(
    'data/csi1000_kline_raw.csv',
    parse_dates=['date'],
    dtype={'stock_code': str},
    usecols=['date', 'stock_code', 'open', 'close', 'amount'],
)
k['stock_code'] = k['stock_code'].str.zfill(6)

# Step 1: overnight vs intraday return
k['prev_close'] = k.groupby('stock_code')['close'].shift(1)
k['ov_ret'] = np.where(k['prev_close'].notna(), k['open'] / k['prev_close'] - 1, np.nan)
k['id_ret'] = np.where(k['open'].notna(),    k['close'] / k['open']      - 1, np.nan)

# Step 2: rolling mean of |ov| vs |id| → ratio
kc = (k.sort_values(['stock_code', 'date'])
       .assign(abs_ov=k['ov_ret'].abs(), abs_id=k['id_ret'].abs())
       .set_index('date'))

kv = kc.groupby('stock_code', group_keys=False)
ra  = kv['abs_ov'].transform(lambda s: s.rolling(W, min_periods=12).mean())
rb2 = kv['abs_id'].transform(lambda s: s.rolling(W, min_periods=12).mean())

safe = (ra + rb2) > 1e-7
k['factor_raw'] = np.where(safe & rb2.notna(),
                           np.where(rb2 > 1e-7, ra / rb2, np.nan), np.nan)

# drop first W rows per stock
k = k.groupby('stock_code', group_keys=False, sort=False).apply(
    lambda g: g.iloc[W:].copy()).reset_index(drop=True)
k = k.dropna(subset=['factor_raw'])

# Step 3: log_amount OLS neutralisation (≡ market-cap proxy)
k['log_amount'] = np.log(k['amount'].clip(lower=1))

crsp = k[['date', 'stock_code', 'factor_raw', 'log_amount']].copy()

def neutralize(group):
    y, x = group['factor_raw'].values, group['log_amount'].values
    m = ~(np.isnan(y) | np.isnan(x))
    if m.sum() < 30:
        return pd.Series(np.nan, index=group.index)
    xm, ym = x[m], y[m]
    if xm.std() < 1e-12:          # log_amount 全同值 → 无法回归
        return pd.Series(np.nan, index=group.index)
    b, i, r, p, _ = stats.linregress(xm, ym)
    resid = np.full(len(y), np.nan)
    resid[m] = ym - (b * xm + i)
    return pd.Series(resid, index=group.index)

crsp['factor_value'] = crsp.groupby('date', group_keys=False).apply(neutralize)

# Step 4: MAD winsorise + cross-sectional z-score
def winz(group):
    v = group['factor_value']
    med = v.median()
    mad = (v - med).abs().median() * 1.4826
    lo, hi = med - 5.5 * mad, med + 5.5 * mad
    v = v.clip(lo, hi)
    mu, sd = v.mean(), v.std(ddof=0)
    if sd < 1e-12:
        return pd.Series(np.nan, index=group.index)
    return (v - mu) / sd

crsp['factor_value'] = crsp.groupby('date', group_keys=False).apply(winz)

out = (crsp[['date', 'stock_code', 'factor_value']]
         .dropna(subset=['factor_value'])
         .sort_values(['date', 'stock_code']))
out['date'] = out['date'].dt.strftime('%Y-%m-%d')
out.to_csv(OUT, index=False)

print(f'rows={len(out)}  null={out["factor_value"].isna().sum()}  '
      f'mean={out.factor_value.mean():.4f}  std={out.factor_value.std():.4f}  '
      f'dates={out.date.min()} → {out.date.max()}')
print(f'saved → {OUT}')
