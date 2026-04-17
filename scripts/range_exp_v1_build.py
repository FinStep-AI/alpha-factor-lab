"""Intraday Range Expansion v1 — HL spread expansion factor
代理「日内过度交易」: HL spread/close 越高→日内交易越激烈→反应过度→反向
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/csi1000_kline_raw.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)

df['hl_spread'] = (df['high'] - df['low']) / df['close'].clip(lower=0.01)
df['hl_mean20'] = df.groupby('stock_code')['hl_spread'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)
df['hl_ratio'] = df['hl_spread'] / df['hl_mean20'].clip(lower=0.001)

# Market-cap proxy
df['log_amount'] = np.log(
    df.groupby('stock_code')['amount']
    .transform(lambda x: x.rolling(20, min_periods=10).mean().clip(lower=1000)) + 1
)

records = []
for date, sub in df.groupby('date', sort=False):
    mask = sub['hl_ratio'].notna() & np.isfinite(sub['hl_ratio']) & (sub['hl_ratio'] > 0)
    sub_clean = sub[mask]
    if len(sub_clean) < 100:
        continue
    y = (-sub_clean['hl_ratio'].values)  # negative: low expansion = high score
    X = sub_clean['log_amount'].values.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(X, y)
    resid = y - lr.predict(X)
    mu, std = resid.mean(), resid.std()
    if std > 0:
        resid = (resid - mu) / std
    for code, val in zip(sub_clean['stock_code'].values, resid):
        records.append({'date': date.strftime('%Y-%m-%d'), 'stock_code': code, 'range_expansion_v1': val})

out = pd.DataFrame(records)
out.to_csv('data/range_expansion_v1_long.csv', index=False)
print(f'Saved: {out.shape}')
