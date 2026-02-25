#!/usr/bin/env python3
"""从中证1000 K线数据生成收益率文件（供回测引擎使用）"""
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

df = pd.read_csv(os.path.join(DATA_DIR, 'csi1000_kline_raw.csv'))
df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['stock_code', 'date'])

# 日收益率 = pct_change / 100
df['daily_return'] = df['pct_change'] / 100

ret = df[['date', 'stock_code', 'daily_return']].copy()
ret['date'] = ret['date'].dt.strftime('%Y-%m-%d')

out_path = os.path.join(DATA_DIR, 'csi1000_returns.csv')
ret.to_csv(out_path, index=False)
print(f"保存: {out_path}")
print(f"  {ret['stock_code'].nunique()} 只, {ret['date'].nunique()} 日, {len(ret)} 行")
