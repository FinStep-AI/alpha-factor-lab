"""Volume Surge Volatility v1 — fast daily-adapted version of 方正金工「适度冒险因子」
日线适配：用日HL spread近似日内波动，用volume increase近似分钟volume增量
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/csi1000_kline_raw.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)

# 1. Volume increase from prev day (clipped at 0)
df['dvol_inc'] = df.groupby('stock_code')['volume'].diff().clip(lower=0)

# 2. Rolling mean/std of volume increase (20d) — surge threshold per stock
g = df.groupby('stock_code')['dvol_inc']
df['dvol_mean20'] = g.transform(lambda x: x.rolling(20, min_periods=10).mean())
df['dvol_std20'] = g.transform(lambda x: x.rolling(20, min_periods=10).std())

# 3. Surge flag: volume increase > mean + std
df['surge_flag'] = df['dvol_inc'] > (df['dvol_mean20'] + df['dvol_std20'])

# 4. Surge volatility proxy: HL spread / close on surge days
df['hl_spread'] = (df['high'] - df['low']) / df['close'].clip(lower=0.01)
df['surge_vol_raw'] = np.where(df['surge_flag'], df['hl_spread'], 0.0)

# 5. Daily cross-sectional: |surge_vol - daily_cross_mean| zscored within surge stocks only
# Compute daily mean of surge_vol_raw (only for surge_flag=True)
surge_mask = df['surge_flag']
daily_surge_mean = df.loc[surge_mask].groupby('date')['surge_vol_raw'].transform('mean')
# Reindex back
daily_mean_map = df.loc[surge_mask].groupby('date')['surge_vol_raw'].mean()
df['daily_surge_mean'] = df['date'].map(daily_mean_map).fillna(0)
df['surge_dev'] = np.where(
    df['surge_flag'],
    (df['surge_vol_raw'] - df['daily_surge_mean']).abs(),
    0.0
)

# Cross-sectional z-score (only on surge days)
daily_surge_std = df.loc[surge_mask].groupby('date')['surge_dev'].transform('std')
std_map = df.loc[surge_mask].groupby('date')['surge_dev'].std()
df['daily_std'] = df['date'].map(std_map).fillna(0)
df['daily_surge_dev'] = np.where(
    (df['surge_flag']) & (df['daily_std'] > 1e-10),
    df['surge_dev'] / df['daily_std'],
    0.0
)

# 6. Rolling 20-day mean → monthly factor
df['vsv_raw'] = df.groupby('stock_code')['daily_surge_dev'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

# 7. Log amount as size proxy (positive, representative of market cap)
df['log_amount_avg'] = np.log(
    df.groupby('stock_code')['amount']
    .transform(lambda x: x.rolling(20, min_periods=10).mean().clip(lower=1000))
    + 1
)

# 8. Build output: per-date market-cap neutralized values
records = []
for date, sub in df.groupby('date', sort=False):
    mask = sub['vsv_raw'].notna() & np.isfinite(sub['vsv_raw']) & (sub['vsv_raw'] != 0)
    sub_clean = sub[mask].copy()
    if len(sub_clean) < 100:
        continue
    
    y = sub_clean['vsv_raw'].values
    X = sub_clean['log_amount_avg'].values.reshape(-1, 1)
    
    lr = LinearRegression()
    lr.fit(X, y)
    resid = y - lr.predict(X)
    
    mu, std = resid.mean(), resid.std()
    if std > 0:
        resid = (resid - mu) / std
    
    for code, val in zip(sub_clean['stock_code'].values, resid):
        records.append({'date': date, 'stock_code': code, 'vsv_daily_surge_v1': float(val)})

out = pd.DataFrame(records)
pivot_out = out.pivot(index='date', columns='stock_code', values='vsv_daily_surge_v1')
pivot_out.index = pd.to_datetime(pivot_out.index)
pivot_out.to_csv('data/vsv_daily_surge_v1.csv')
n_dates = len(pivot_out)
n_stocks = pivot_out.notna().sum(axis=1).mean()
nnz_pct = (pivot_out.notna().sum().sum() / pivot_out.size) * 100
print(f"Done: {pivot_out.shape} dates×stocks, avg_stocks={n_stocks:.0f}")
print(f"Range: {pivot_out.index.min().date()} ~ {pivot_out.index.max().date()}")
print(f"Non-null: {nnz_pct:.1f}%")
print(f"Sample: {pivot_out.stack().describe().to_string()}")
PYEOF
python3 /tmp/surge_v2.py