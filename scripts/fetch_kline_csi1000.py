#!/usr/bin/env python3
"""批量获取中证1000日K线（断点续传）"""
import akshare as ak
import pandas as pd
import time, os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

codes = pd.read_csv(os.path.join(DATA_DIR, 'csi1000_codes.csv'))['stock_code'].tolist()
codes = [str(c).zfill(6) for c in codes]

cache_file = os.path.join(DATA_DIR, 'csi1000_kline_raw.csv')
done_codes = set()
all_dfs = []

if os.path.exists(cache_file):
    existing = pd.read_csv(cache_file)
    existing['stock_code'] = existing['stock_code'].astype(str).str.zfill(6)
    done_codes = set(existing['stock_code'].unique())
    all_dfs.append(existing)
    print(f"已有 {len(done_codes)} 只K线数据")

remaining = [c for c in codes if c not in done_codes]
print(f"还需下载 {len(remaining)} 只...")

batch_dfs = []
fail_count = 0
for i, code in enumerate(remaining):
    try:
        df = ak.stock_zh_a_hist(
            symbol=code, period='daily',
            start_date='20221001', end_date='20260222',
            adjust='qfq'
        )
        if df is not None and len(df) > 0:
            df = df.rename(columns={
                '日期': 'date', '股票代码': 'stock_code',
                '开盘': 'open', '收盘': 'close',
                '最高': 'high', '最低': 'low',
                '成交量': 'volume', '成交额': 'amount',
                '振幅': 'amplitude', '涨跌幅': 'pct_change',
                '涨跌额': 'change', '换手率': 'turnover'
            })
            df['stock_code'] = code
            batch_dfs.append(df)
            fail_count = 0

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(remaining)}] {code} done, batch={len(batch_dfs)}")

        time.sleep(0.2)

    except Exception as e:
        print(f"  [!] {code} 失败: {str(e)[:80]}")
        fail_count += 1
        if fail_count >= 5:
            print("连续失败5次，等30秒...")
            time.sleep(30)
            fail_count = 0
        else:
            time.sleep(1)

    # 每100只保存
    if (i + 1) % 100 == 0 and batch_dfs:
        batch_df = pd.concat(batch_dfs, ignore_index=True)
        all_dfs.append(batch_df)
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=['date', 'stock_code'])
        combined.to_csv(cache_file, index=False)
        done_now = combined['stock_code'].nunique()
        print(f"  [保存] 累计 {done_now} 只, {len(combined)} 行")
        batch_dfs = []

if batch_dfs:
    all_dfs.append(pd.concat(batch_dfs, ignore_index=True))

if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['date', 'stock_code'])
    combined.to_csv(cache_file, index=False)
    print(f"\n完成! {combined['stock_code'].nunique()} 只, {len(combined)} 行")
else:
    print("没有获取到任何数据")
