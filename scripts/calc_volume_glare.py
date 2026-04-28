"""
适度冒险因子 v2 — 方正金工研报, 日线近似
"""
import pandas as pd, numpy as np
from pathlib import Path
from numpy.linalg import lstsq

WORKDIR = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab")
DATA   = WORKDIR / "data"
print("=== 适度冒险因子 v2 ===")

kline  = pd.read_csv(DATA / "csi1000_kline_raw.csv", parse_dates=["date"])
print(f"[数据] {kline.shape[0]:,}行  {kline['date'].max().date()}  {kline['stock_code'].nunique()}股")

df = kline[['stock_code','date','close','high','low','volume','amount','pct_change']].copy()
df['ret'] = pd.to_numeric(df['pct_change'], errors='coerce') / 100
df['rng'] = (df['high'] - df['low']) / df['close']

# 截面均值截面的成交截面中性化截面截面成交截面
daily = df.copy()
daily['amt_z'] = daily.groupby('date')['amount'].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-8))

daily['surge']     = (daily['amt_z'] > 0.5).astype(int)
daily['surge_ret'] = daily['ret'].where(daily['surge'].astype(bool))
daily['surge_rng'] = daily['rng'].where(daily['surge'].astype(bool))

def per_stock(g):
    g = g.sort_values('date').copy()
    n = 20; o = pd.DataFrame(index=g.index)
    o['n_surge'] = g['surge'  ].rolling(n, min_periods=10).sum()
    o['m_r']     = g['surge_ret'].rolling(n, min_periods=10).mean()
    o['m_rg']    = g['surge_rng'].rolling(n, min_periods=10).mean()
    o['mc']      = g['close'] * g['volume']    # 市值近似
    return o

print("[因子] 个股截面滚动...", flush=True)
pivot = daily.groupby('stock_code', group_keys=False).apply(per_stock)
daily = pd.concat([daily, pivot], axis=1)

# 截面段: 成交截面截面截面
def neu(yv, xv):
    xv = np.log(xv.fillna(xv.median()))
    m  = yv.notna() & xv.notna() & np.isfinite(xv)
    if m.sum() < 10: return yv.values
    yy = yv[m].values; xx = xv[m].values
    X  = np.column_stack([np.ones(len(xx)), xx])
    b, _, _, _ = lstsq(X, yy, rcond=None)
    res = np.empty(len(yv)); res[:] = np.nan
    res[m.values] = yy - X @ b
    return res

print("[中性化] 截面OLS成交截面成交截面", flush=True)
for col in ['n_surge', 'm_r', 'm_rg']:
    daily[col+'_n'] = daily.groupby('date', group_keys=False).apply(
        lambda x: pd.Series(neu(x[col], x['mc']), index=x.index)
    ).reset_index(level=0, drop=True)

daily['vol_glare'] = (
    daily['n_surge_n'].fillna(0) * 0.4 +
    (-daily['m_r_n'].fillna(0)) * 0.35 +
    (-daily['m_rg_n'].fillna(0)) * 0.25
)

out = daily[['stock_code','date','vol_glare']].dropna().rename(columns={'vol_glare':'volume_glare_factor'})
out.to_csv(DATA / "volume_glare_factor.csv", index=False)
print(f"[完成] {len(out):,}行  {out['date'].min().date()}~{out['date'].max().date()}")
