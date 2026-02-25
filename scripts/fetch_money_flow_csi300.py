#!/usr/bin/env python3
"""批量获取沪深300资金流数据（断点续传）"""
import akshare as ak
import pandas as pd
import time, os, sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

codes = pd.read_csv(os.path.join(DATA_DIR, 'csi300_codes.csv'))['stock_code'].tolist()
codes = [str(c).zfill(6) for c in codes]

cache_file = os.path.join(DATA_DIR, 'csi300_money_flow.csv')
done_codes = set()
all_dfs = []

if os.path.exists(cache_file):
    existing = pd.read_csv(cache_file)
    done_codes = set(existing['stock_code'].astype(str).str.zfill(6).unique())
    all_dfs.append(existing)
    print(f"已有 {len(done_codes)} 只资金流数据")

remaining = [c for c in codes if c not in done_codes]
print(f"还需下载 {len(remaining)} 只...")

batch_dfs = []
fail_count = 0
for i, code in enumerate(remaining):
    try:
        market = 'sz' if code.startswith(('0', '3')) else 'sh'
        df = ak.stock_individual_fund_flow(stock=code, market=market)
        if df is not None and len(df) > 0:
            df = df.rename(columns={
                '日期': 'date', '收盘价': 'close', '涨跌幅': 'pct_change',
                '主力净流入-净额': 'main_net_inflow',
                '主力净流入-净占比': 'main_net_inflow_pct',
                '超大单净流入-净额': 'super_large_net',
                '超大单净流入-净占比': 'super_large_pct',
                '大单净流入-净额': 'large_net', '大单净流入-净占比': 'large_pct',
                '中单净流入-净额': 'medium_net', '中单净流入-净占比': 'medium_pct',
                '小单净流入-净额': 'small_net', '小单净流入-净占比': 'small_pct',
            })
            df['stock_code'] = code
            batch_dfs.append(df)
            fail_count = 0
        
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(remaining)}] {code} done")
        time.sleep(0.25)
        
    except Exception as e:
        print(f"  [!] {code} 失败: {str(e)[:80]}")
        fail_count += 1
        if fail_count >= 5:
            print("连续失败5次，等待30秒...")
            time.sleep(30)
            fail_count = 0
        else:
            time.sleep(1)
    
    if (i + 1) % 50 == 0 and batch_dfs:
        batch_df = pd.concat(batch_dfs, ignore_index=True)
        all_dfs.append(batch_df)
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=['date', 'stock_code'])
        combined.to_csv(cache_file, index=False)
        print(f"  [保存] 累计 {combined['stock_code'].nunique()} 只")
        batch_dfs = []

if batch_dfs:
    all_dfs.append(pd.concat(batch_dfs, ignore_index=True))

if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['date', 'stock_code'])
    combined.to_csv(cache_file, index=False)
    print(f"\n完成! {combined['stock_code'].nunique()} 只, {len(combined)} 行")
