"""
因子：波动率模糊Spread（Vol-of-Vol Modified Spread）
ID: vol_of_vol_cspread_v1

源论文：方正金工 2022-08-04《波动率的波动率与投资者模糊性厌恶——多因子选股系列研究之五》
        （日线约化版，保留核心逻辑）
"""

import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'skills/alpha-factor-lab/scripts')

import numpy as np
import pandas as pd
from factor_calculator import neutralize_cross_section

# ── paths ───────────────────────────────────────────────────────────
BASE    = '.'
KLINE   = f'{BASE}/data/csi1000_kline_raw.csv'
OUT     = f'{BASE}/data/vol_of_vol_cspread_v1.csv'
WIN_VOL = 20    # rolling-std window  (ambiguity proxy)
WIN_LKB = 20    # factor lookback      (≈ 1 month)

# ════════════════════════════════════════════════════════════════════
# 1. load
# ════════════════════════════════════════════════════════════════════
print("Loading kline …")
df = pd.read_csv(KLINE, usecols=['date','stock_code','open','close','high','low','amount'])
df['date']   = pd.to_datetime(df['date'])
for c in ['close','high','low','amount']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=['close','high','low'])
df = df.sort_values(['stock_code','date']).reset_index(drop=True)

# ════════════════════════════════════════════════════════════════════
# 2. close-position proxy  cp_t = (close-low)/(high-low)
# ════════════════════════════════════════════════════════════════════
print("Step 1/5  close-position …")
rng = (df['high'] - df['low']).clip(lower=df['close'] * 0.001)
df['cp'] = ((df['close'] - df['low']) / rng).clip(0, 1)

# ════════════════════════════════════════════════════════════════════
# 3. rolling 20-d std of cp  →  ambiguity proxy
# ════════════════════════════════════════════════════════════════════
print("Step 2/5  cp rolling-std (vol-of-vol) …")
df['cp_std20'] = df.groupby('stock_code')['cp'].transform(
    lambda s: s.rolling(WIN_VOL, min_periods=10).std()
)

# ════════════════════════════════════════════════════════════════════
# 4. daily fuzzy spread + modified version
#    cspread_t   = cp_t − cp_std20_t
#    adj_t       = cspread_t / std10(cspread_t)  when cspread < 0
#                = cspread_t                      otherwise
# ════════════════════════════════════════════════════════════════════
print("Step 3/5  daily fuzzy spread …")
df['cspread'] = df['cp'] - df['cp_std20']

print("Step 4/5  10-d std for noise suppression …")
df['cs_std10'] = df.groupby('stock_code')['cspread'].transform(
    lambda s: s.rolling(10, min_periods=5).std().clip(lower=1e-8)
)
df['adj'] = np.where(df['cspread'] < 0,
                     df['cspread'] / df['cs_std10'],
                     df['cspread'])

# ════════════════════════════════════════════════════════════════════
# 5. cross-sectional magnitude rescale for the negative block
# ════════════════════════════════════════════════════════════════════
print("Step 5/5  cross-sectional rescale + 20d lookback …")
# cross-sectional rescale: keep stock_code + adj_cs, merge back by index
rescale_parts = []
for dt, g in df.groupby('date', sort=True):
    a  = g['adj'].values.copy().astype(float)
    neg_sum = a[a < 0].sum()
    if abs(neg_sum) > 1e-12:
        mask = a < 0
        a[mask] = a[mask] / abs(neg_sum) * abs(neg_sum)
    tmp = pd.DataFrame({'adj_cs': a}, index=g.index)
    rescale_parts.append(tmp)

adj_cs_s = pd.concat(rescale_parts).sort_index()['adj_cs']
df['adj_cs'] = adj_cs_s.values

# 20-d rolling mean → monthly factor value
df['raw_factor'] = df.groupby('stock_code')['adj_cs'].transform(
    lambda s: s.rolling(WIN_LKB, min_periods=15).mean()
)

# ════════════════════════════════════════════════════════════════════
# 6. market-cap neutralize  (log-amount as size proxy)
# ════════════════════════════════════════════════════════════════════
print("Neutralizing by log_amount_20d …")
df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
    lambda s: np.log(s.rolling(20, min_periods=10).mean().clip(lower=1))
)

ready = df[['date','stock_code','raw_factor','log_amount_20d']].dropna().copy()
ready = ready.sort_values(['date','stock_code']).reset_index(drop=True)

ready['factor_value'] = neutralize_cross_section(
    ready, 'raw_factor', neutralize_cols=['log_amount_20d']
)

out = ready[['date','stock_code','factor_value']].dropna()
out['stock_code'] = out['stock_code'].astype(str).str.zfill(6)
out.to_csv(OUT, index=False)
print(f"Done — {len(out)} rows → {OUT}")
print(f"  date range : {out['date'].min()}  ~  {out['date'].max()}")
print(out.tail(5).to_string(index=False))
