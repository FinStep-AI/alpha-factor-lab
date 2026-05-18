"""factor_roe_sigtrend_v1
Growth | ROE 显著性趋势 = β × tstat（8Q 滚动）
区别于:
  ROE时序趋势质量 v1         β × R²
  ROE同比增长因子            ΔROE_YoY
  ROE时序自回归持续性 v1    CORR(前4Q,后4Q)
截面中性化: log_amount_20d OLS resid + MAD z-score
"""
import os, warnings, pandas as pd, numpy as np
warnings.filterwarnings('ignore')

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUND  = os.path.join(BASE, 'data', 'csi1000_fundamental_cache.csv')
KLINE = os.path.join(BASE, 'data', 'csi1000_kline_raw.csv')
OUT   = os.path.join(BASE, 'data', 'factor_roe_sigtrend_v1.csv')

STEP, WIN = 63, 8          # 1 quarter, 8 quarters

# ── 1. 财务日频化 ─────────────────────────────────────────────────────────────
fund = (pd.read_csv(FUND, usecols=['stock_code','report_date','roe'])
          .assign(report_date=lambda d: pd.to_datetime(d.report_date))
          .drop_duplicates(['stock_code','report_date'])
          .sort_values(['stock_code','report_date']))
fund['avail'] = fund['report_date'] + pd.Timedelta(days=45)

all_dates = pd.date_range(fund['avail'].min(), fund['avail'].max(), freq='D')
panel = {}
for sc, g in fund.groupby('stock_code', sort=False):
    g = g.set_index('avail').sort_index()
    s = g['roe'].reindex(g.index.union(all_dates)).ffill().reindex(all_dates)
    panel[sc] = s
F = pd.DataFrame(panel, index=all_dates)
F.index.name = 'date'
F = F.reset_index()
F['date'] = F['date'].dt.strftime('%Y-%m-%d')
F = F.set_index('date')

# ── 2. β × tstat ───────────────────────────────────────────────────────────────
v    = F.values.astype(float)
chg  = np.full_like(v, np.nan)
prev = np.abs(v[:-STEP]) + 1e-6
chg[STEP:] = (v[STEP:] - v[:-STEP]) / prev

dates  = F.index.values.astype(str)
stocks = F.columns.astype(int).values
T, N   = v.shape

Xfull = np.column_stack([np.ones(WIN), np.arange(WIN, dtype=float)])
XfXi  = np.linalg.inv(Xfull.T @ Xfull + 1e-12 * np.eye(2))  # (2,2)常数
x2    = float(((np.arange(WIN, dtype=float) - np.arange(WIN, dtype=float).mean()) ** 2).sum())  # Σ(t-μ)²

rows = []
for t in range(WIN * STEP, T):
    start = t - WIN * STEP + 1
    win   = chg[slice(start, t + 1, STEP)]                    # WIN × N
    ok    = (~np.isnan(win)).all(axis=0)
    win   = win[:, ok]
    if win.shape[1] == 0:
        continue
    # z-score 每支股票
    ctr = win.mean(axis=0, keepdims=True)
    sd  = win.std(axis=0, keepdims=True); sd[sd == 0] = np.nan
    yn  = (win - ctr) / sd                                    # WIN × K

    # OLS:  beta = Xi @ X.T @ yn   →  (2, K)
    beta  = XfXi @ (Xfull.T @ yn)                             # (2, K)
    pred  = Xfull @ beta                                      # WIN × K
    resid = yn  - pred                                        # WIN × K
    se2   = (resid ** 2).sum(axis=0) / max(WIN - 2, 1) / max(x2, 1e-12)
    se2[se2 < 0] = 0.0
    ts    = beta[1] / (np.sqrt(se2) + 1e-12)
    score = beta[1] * ts                                      # K

    ok2 = np.isfinite(score)
    if not ok2.any():
        continue
    rows.append(pd.DataFrame({
        'date': dates[t],
        'stock_code': stocks[ok][ok2],
        'factor': score[ok2],
    }))

if not rows:
    raise SystemExit('no rows')
rdf = pd.concat(rows, ignore_index=True)

# ── 3. 截面中性化 ──────────────────────────────────────────────────────────────
kline = pd.read_csv(KLINE, usecols=['date','stock_code','amount'])
kline['date'] = pd.to_datetime(kline['date']).dt.strftime('%Y-%m-%d')
amt = (kline.dropna(subset=['amount'])
         .groupby(['date','stock_code'])['amount'].mean()
         .reset_index())
rdf  = rdf.merge(amt, on=['date','stock_code'], how='inner')
print(f'after merge: {len(rdf)} rows, {rdf.date.nunique()} dates')

O2, N_MIN = np.eye(2) * 1e-8, 30

def _cs(sub):
    s2 = sub.dropna(subset=['amount', 'factor'])
    if len(s2) < N_MIN:
        sub = sub.copy(); sub['fv'] = np.nan; return sub
    X2   = np.column_stack([np.ones(len(s2)), np.log1p(s2['amount'].values)])
    y    = s2['factor'].values.astype(float)
    try:
        Xi2  = np.linalg.inv(X2.T @ X2 + O2)
    except Exception:
        sub = sub.copy(); sub['fv'] = np.nan; return sub
    r    = y - X2 @ (Xi2 @ (X2.T @ y))
    med  = float(np.median(r))
    mad  = float(np.median(np.abs(r - med)))
    z    = ((r - med) / (1.4826 * mad + 1e-9)).clip(-5, 5) if mad > 0 else r * 0
    out  = sub.copy(); out['fv'] = np.nan
    out.loc[s2.index, 'fv'] = z
    return out

out = (rdf.groupby('date', group_keys=False).apply(_cs)
         [['date', 'stock_code', 'fv']]
         .dropna(subset=['fv'])
         .drop_duplicates(['date', 'stock_code'])
         .rename(columns={'fv': 'factor_value'})
         .reset_index(drop=True))
out.to_csv(OUT, index=False)
spd = out.groupby('date')['stock_code'].count()
print(f'wrote {len(out)} rows | dates={out.date.nunique()} | '
      f'avg_stocks/date={spd.mean():.0f} min={spd.min()} max={spd.max()}')
print(out['factor_value'].describe())
