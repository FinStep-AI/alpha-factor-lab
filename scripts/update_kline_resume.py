#!/usr/bin/env python3
"""断点续传：从增量文件中找到已完成的股票，只跑剩余的"""
import pandas as pd
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_kline import fetch_kline_tencent

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
KLINE_FILE = os.path.join(DATA_DIR, 'csi1000_kline_raw.csv')
TMP_FILE = KLINE_FILE + '.incremental.csv'
CODES_FILE = os.path.join(DATA_DIR, 'csi1000_codes.csv')

# Read existing
existing = pd.read_csv(KLINE_FILE)
existing['stock_code'] = existing['stock_code'].astype(str).str.zfill(6)
old_max_date = existing['date'].max()
print(f"现有数据最新: {old_max_date}")

# Read already fetched incremental
done_codes = set()
if os.path.exists(TMP_FILE):
    inc = pd.read_csv(TMP_FILE)
    inc['stock_code'] = inc['stock_code'].astype(str).str.zfill(6)
    done_codes = set(inc['stock_code'].unique())
    print(f"增量文件已有: {len(done_codes)} 只, {len(inc)} 行")

# Remaining
codes = pd.read_csv(CODES_FILE)['stock_code'].tolist()
codes = [str(c).zfill(6) for c in codes]
remaining = [c for c in codes if c not in done_codes]
print(f"还需拉取: {len(remaining)} 只")

batch = []
success = 0
for i, code in enumerate(remaining):
    try:
        klines = fetch_kline_tencent(code, 30)
        if klines:
            for kl in klines:
                if kl['date'] > old_max_date:
                    batch.append({
                        'date': kl['date'], 'stock_code': code,
                        'open': kl['open'], 'close': kl['close'],
                        'high': kl['high'], 'low': kl['low'],
                        'volume': kl['volume'], 'amount': 0,
                        'amplitude': 0, 'pct_change': 0, 'change': 0, 'turnover': 0,
                    })
            success += 1
    except Exception as e:
        time.sleep(1)
        continue

    # 每50只落盘
    if (i + 1) % 50 == 0 and batch:
        df = pd.DataFrame(batch)
        df.to_csv(TMP_FILE, mode='a', header=not os.path.exists(TMP_FILE), index=False)
        print(f"  [{i+1}/{len(remaining)}] +{len(batch)}条 累计成功{success}")
        batch = []
    
    time.sleep(0.12)

# Final batch
if batch:
    df = pd.DataFrame(batch)
    df.to_csv(TMP_FILE, mode='a', header=not os.path.exists(TMP_FILE), index=False)
    print(f"  [完] +{len(batch)}条 累计成功{success}")

print(f"\n拉取完成，开始合并...")

# Merge
inc_all = pd.read_csv(TMP_FILE)
inc_all['stock_code'] = inc_all['stock_code'].astype(str).str.zfill(6)
combined = pd.concat([existing, inc_all], ignore_index=True)
combined = combined.drop_duplicates(subset=['date', 'stock_code'], keep='last')
combined = combined.sort_values(['stock_code', 'date']).reset_index(drop=True)

# Calc pct_change
print("补算涨跌幅...")
def calc_pct(g):
    g = g.sort_values('date')
    g['pct_change'] = g['close'].pct_change() * 100
    return g
combined = combined.groupby('stock_code', group_keys=False).apply(calc_pct)

combined.to_csv(KLINE_FILE, index=False)
new_dates = sorted(inc_all['date'].unique())
print(f"\n✅ 完成!")
print(f"  {existing['stock_code'].nunique()} → {combined['stock_code'].nunique()} 只")
print(f"  {len(existing)} → {len(combined)} 行")
print(f"  日期: {old_max_date} → {combined['date'].max()}")
print(f"  新增日期: {new_dates}")

# Update returns
ret_file = os.path.join(DATA_DIR, 'csi1000_returns.csv')
print(f"\n更新收益率文件...")
ret = combined[['date','stock_code','close']].sort_values(['stock_code','date'])
ret['return'] = ret.groupby('stock_code')['close'].pct_change()
ret = ret.dropna(subset=['return'])
ret[['date','stock_code','return']].to_csv(ret_file, index=False)
print(f"  收益率: {len(ret)} 行, 最新 {ret['date'].max()}")

# Cleanup
os.remove(TMP_FILE)
print("  已清理临时文件")
