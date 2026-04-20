"""Lightweight turnover 3D composite factor computation."""
import pandas as pd
import numpy as np

df = pd.read_csv('data/csi1000_kline_raw.csv', parse_dates=['date'])
df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
df['ret'] = df['pct_change'] / 100.0

# Rolling features using groupby transform
turn = df['turnover']
df['t_ma20'] = turn.groupby(df['stock_code']).transform(lambda x: x.rolling(20, min_periods=10).mean())
df['t_ma5']  = turn.groupby(df['stock_code']).transform(lambda x: x.rolling(5,  min_periods=5).mean())
df['t_std20'] = turn.groupby(df['stock_code']).transform(lambda x: x.rolling(20, min_periods=10).std())
df['t_mean20'] = turn.groupby(df['stock_code']).transform(lambda x: x.rolling(20, min_periods=10).mean())

# Component 1: turnover level (log)
df['f1'] = np.log(df['t_ma20'].clip(lower=0.001) + 1)
# Component 2: turnover deceleration (negative of ratio log)
df['f2'] = -np.log((df['t_ma5'] / (df['t_ma20'] + 1e-8)).clip(lower=1e-8))
# Component 3: turnover CV (negative - low CV is stable/good)
df['f3'] = -(df['t_std20'] / (df['t_mean20'] + 1e-8))

# Log amount for neutralization
df['log_amount_20d'] = np.log(df['amount'].clip(lower=1) + 1)
df['log_amount_20d'] = df['log_amount_20d'].groupby(df['stock_code']).transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

# Cross-sectional z-score each component (per date)
components = ['f1', 'f2', 'f3']
for c in components:
    means = df.groupby('date')[c].transform('mean')
    stds = df.groupby('date')[c].transform('std').clip(lower=1e-8)
    df[f'{c}_z'] = (df[c] - means) / stds

# Clip extreme z-scores before composite
for c in components:
    df[f'{c}_z'] = df[f'{c}_z'].clip(-3, 3)

# Composite = sum of 3 z-scores
df['composite_z'] = df['f1_z'] + df['f2_z'] + df['f3_z']

# Per-date: OLS neutralize by log_amount_20d → MAD winsorize → z-score
out_rows = []
dates = df['date'].unique()

for d in dates:
    m = df['date'] == d
    sub = df[m]
    
    x = sub['composite_z'].values.astype(float)
    y = sub['log_amount_20d'].values.astype(float)
    
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 20:
        continue
    
    xv = x[valid]; yv = y[valid]
    x_c = xv - xv.mean(); y_c = yv - yv.mean()
    denom = np.sum(x_c**2)
    if denom < 1e-10:
        continue
    
    beta = np.sum(x_c * y_c) / denom
    neut = x_c - beta * y_c
    
    md = np.median(neut)
    mad = np.median(np.abs(neut - md))
    if mad < 1e-10:
        continue
    
    neut = np.clip(neut, md - 5.2*mad, md + 5.2*mad)
    
    mu, sigma = neut.mean(), neut.std()
    if sigma < 1e-10:
        continue
    
    z = (neut - mu) / sigma
    idx = sub.index[valid]
    
    for i, row_i in enumerate(idx):
        out_rows.append({
            'date': df.loc[row_i, 'date'],
            'stock_code': df.loc[row_i, 'stock_code'],
            'turnover_3d_final': z[i],
        })

result = pd.DataFrame(out_rows)
if len(result) == 0:
    print("ERROR: No factor values produced!")
    exit(1)

print(f"Rows: {len(result)}, stocks: {result['stock_code'].nunique()}")
print(f"Factor stats:\n{result['turnover_3d_final'].describe()}")
result.to_csv('data/factor_turnover_3d_v1.csv', index=False)
print("Saved to data/factor_turnover_3d_v1.csv")
