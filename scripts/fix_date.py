import pandas as pd
df = pd.read_csv('data/range_expansion_v1_long.csv')
df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
df.to_csv('data/range_expansion_v1_long_fixed.csv', index=False)
print(f'Fixed: {df.shape}')
print(df.head())
