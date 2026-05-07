#!/usr/bin/env python3 -u
"""
因子: 波动率效率 v3 (vol_efficiency_v3)
改进: 扩展Parkinson窗口至40d + 取exp(log(Pk40) - log(CC20))而非比值
目标: 冲击IC > 0.015, t > 2, Sharpe > 0.8, mono > 0.8
"""
import numpy as np, pandas as pd, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "csi1000_kline_raw.csv"
RET_PATH = BASE_DIR / "data" / "csi1000_returns.csv"
OUTPUT_DIR = BASE_DIR / "data" / "factor_vol_eff_v3_alt.csv"

print("[1] 加载数据...")
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['stock_code','date']).reset_index(drop=True)

# Build pivots
close_piv = df.pivot_table(index='date', columns='stock_code', values='close')
high_piv  = df.pivot_table(index='date', columns='stock_code', values='high')
low_piv   = df.pivot_table(index='date', columns='stock_code', values='low')
amt_piv   = df.pivot_table(index='date', columns='stock_code', values='amount')

print(f"   Dates: {len(close_piv)}, Stocks: {len(close_piv.columns)}")

# ---- Parkinson estimator (40d) ----
ln_hl = np.log(high_piv / low_piv.clip(lower=1e-8))
pk = ln_hl**2 / (4 * np.log(2))
pk_40d  = pk.rolling(40, min_periods=20).mean() ** 0.5   # Parkinson vol 40d

# ---- Close-close  20d ----
# Forward difference of log close
log_close = np.log(close_piv)
lcc = log_close.diff().rolling(20, min_periods=10).std()  # CC vol 20d

# ---- Risk-free rate proxy (negligible effect) ----
rf = 0.0

# ---- Factor = log( CC20d ) - log(PK40d) ----
# Signal: efficient price discovery (CC returns tracking PK at high ratio)
factor_raw = np.log(lcc.clip(lower=1e-8)) - np.log(pk_40d.clip(lower=1e-8))

print(f"[2] Market-cap proxy (amt / turnover) neutralization...")
# Mktcap proxy
turn_piv = df.pivot_table(index='date', columns='stock_code', values='turnover')
mktcap_piv = amt_piv / turn_piv.replace(0, np.nan)
log_mktcap = np.log(mktcap_piv.replace(0, np.nan))

# 5% winsorize per date
print("[3] Winsorize 5%...")
def winsorize(df_row, pct=0.05):
    lo = df_row.quantile(pct)
    hi = df_row.quantile(1-pct)
    return df_row.clip(lo, hi)

for dt in factor_raw.index:
    factor_raw.loc[dt] = winsorize(factor_raw.loc[dt].dropna())
    log_mktcap.loc[dt] = winsorize(log_mktcap.loc[dt].dropna())

# OLS neutralization (by date)
print("[4] OLS neutralization...")
fa = factor_raw.values.copy()
mk = log_mktcap.values.copy()

for i in range(fa.shape[0]):
    row_mask = np.isfinite(fa[i,:]) & np.isfinite(mk[i,:])
    if row_mask.sum() < 30: continue
    fsub = fa[i,row_mask]
    xsub = mk[i,row_mask]
    X = np.column_stack([np.ones(len(fsub)), xsub])
    try:
        beta = np.linalg.lstsq(X, fsub, rcond=None)[0]
        fa[i,row_mask] = fsub - X @ beta
    except: pass

factor_n = pd.DataFrame(fa, index=close_piv.index, columns=close_piv.columns)

# Cross-sectional standardize per date
print("[5] Cross-sectional standardization...")
means = np.nanmean(fa, axis=1, keepdims=True)
stds  = np.nanstd(fa, axis=1, keepdims=True)
stds[stds < 1e-8] = 1.0
fa_z = (fa - means) / stds

# Clip ±3
fa_z = np.clip(fa_z, -3, 3)

result = pd.DataFrame({
    'date': np.repeat(factor_n.index.values, len(close_piv.columns)),
    'stock_code': np.tile(close_piv.columns.values, len(factor_n.index)),
    'factor_value': fa_z.flatten()
})
result = result.dropna()
result['date'] = result['date'].astype(str).str[:10]
result = result.sort_values(['date','stock_code']).reset_index(drop=True)
result.to_csv(OUTPUT_DIR, index=False)
print(f"[6] Saved {OUTPUT_DIR}: {result.shape}")
print(f"   Date range: {result.date.min()} ~ {result.date.max()}")
print(f"   Factor stats: {result.factor_value.describe()}")
