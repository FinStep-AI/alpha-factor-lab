"""Build gap efficiency variants and save correctly."""
import pandas as pd, numpy as np

k = pd.read_csv('data/csi1000_kline_raw.csv', parse_dates=['date'])
k = k.sort_values(['stock_code','date']).reset_index(drop=True)
prev_c = k.groupby('stock_code')['close'].shift(1)
rng = (k['high']-k['low']) / (prev_c+1e-8)
gap_r = (k['close']-k['open']).abs() / (prev_c+1e-8)
k['gap_eff'] = (gap_r / (rng+1e-8)).clip(0,1.0)

k['ge_raw'] = k['gap_eff']
k['ge_sma20'] = k.groupby('stock_code')['gap_eff'].transform(
    lambda x: x.rolling(20,min_periods=10).mean())
k['ge_ewm20'] = k.groupby('stock_code')['gap_eff'].transform(
    lambda x: x.ewm(span=20,min_periods=10).mean())
k['ge_sma40'] = k.groupby('stock_code')['gap_eff'].transform(
    lambda x: x.rolling(40,min_periods=20).mean())

log_amt = np.log(k['amount'].clip(lower=1)+1)
k['log_amt_20d'] = log_amt.groupby(k['stock_code']).transform(
    lambda x: x.rolling(20,min_periods=10).mean())

variants = ['ge_raw','ge_sma20','ge_ewm20','ge_sma40']
for v in variants:
    df = k[['date','stock_code',v,'log_amt_20d']].dropna()
    df.to_csv(f'data/factor_{v}_raw.csv', index=False)
    print(f"{v}_raw: {len(df)} rows, mean={df[v].mean():.4f}, std={df[v].std():.4f}")

print("Done.")
