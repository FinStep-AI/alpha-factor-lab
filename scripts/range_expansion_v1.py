"""
Intraday Range Expansion (RE) — 日内幅度扩张因子
构造思路：HL spread 过扩的股票往往当日过度交易/情绪加剧，次日反转概率高
  step1: hl_spread = (high - low) / close
  step2: hl_expansion = hl_spread / MA20(hl_spread)  ← 当日HL vs 近期平均
  step3: 市值中性化 → 取负值（高扩→低收益，反转逻辑）
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/csi1000_kline_raw.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)

# HL spread
df['hl_spread'] = (df['high'] - df['low']) / df['close'].clip(lower=0.01)

# 20d rolling mean
df['hl_mean20'] = df.groupby('stock_code')['hl_spread'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

# Expansion ratio: >1 = 今日HL比近期均值宽
df['hl_ratio'] = df['hl_spread'] / df['hl_mean20'].clip(lower=0.001)

# Negate: high expansion → low expected return (overreversal logic)
df['re_raw'] = -df['hl_ratio']

# Log amount for neutralization
df['log_amount'] = np.log(
    df.groupby('stock_code')['amount']
    .transform(lambda x: x.rolling(20, min_periods=10).mean().clip(lower=1000))
    + 1
)

# Build per-date, market-cap neutralized
records = []
for date, sub in df.groupby('date', sort=False):
    mask = sub['re_raw'].notna() & np.isfinite(sub['re_raw'])
    sub_clean = sub[mask].copy()
    if len(sub_clean) < 100:
        continue
    
    y = sub_clean['re_raw'].values
    X = sub_clean['log_amount'].values.reshape(-1, 1)
    
    lr = LinearRegression()
    lr.fit(X, y)
    resid = y - lr.predict(X)
    
    mu, std = resid.mean(), resid.std()
    if std > 0:
        resid = (resid - mu) / std
    
    for code, val in zip(sub_clean['stock_code'].values, resid):
        records.append({'date': date, 'stock_code': code, 'range_expansion_v1': float(val)})

out = pd.DataFrame(records)
pivot = out.pivot(index='date', columns='stock_code', values='range_expansion_v1')
pivot.to_csv('data/range_expansion_v1_long.csv')
print(f"Saved: {out.shape}, pivot={pivot.shape}")
print(f"Range: {pivot.index.min()} ~ {pivot.index.max()}")
