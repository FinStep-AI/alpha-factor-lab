#!/usr/bin/env python3 -u
"""构建资金流因子: 主力资金净流入占比 (20日均值, 市值中性化)"""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Step1: 加载K线...", flush=True)
import pandas as pd
import numpy as np
from fintool_client import get_net_flow
from concurrent.futures import ThreadPoolExecutor, as_completed

df_k = pd.read_csv('data/csi1000_kline_raw.csv', usecols=['stock_code','date','close'])
df_k['stock_code'] = df_k['stock_code'].astype(str).str.zfill(6)
df_k['date'] = pd.to_datetime(df_k['date'])
codes = sorted(df_k['stock_code'].unique())
print(f"  {len(codes)}只, {len(df_k)}行", flush=True)

print("Step2: 批量拉取资金流...", flush=True)
all_records = []
success = fail = 0
t0 = time.time()

def fetch(code):
    try:
        rows = get_net_flow(code, '2024-12-01', '2026-03-07')
        return code, rows, None
    except Exception as e:
        return code, None, str(e)[:50]

with ThreadPoolExecutor(max_workers=10) as ex:
    futs = {ex.submit(fetch, c): c for c in codes}
    for i, fut in enumerate(as_completed(futs), 1):
        code, rows, err = fut.result()
        if err or not rows:
            fail += 1
            continue
        success += 1
        for r in rows:
            total_in = (abs(r.get('super_in',0) or 0) + abs(r.get('large_in',0) or 0) +
                       abs(r.get('medium_in',0) or 0) + abs(r.get('little_in',0) or 0))
            major_net = r.get('major_net_flow_in', 0) or 0
            ratio = major_net / total_in if total_in > 0 else 0
            all_records.append({'date': r.get('date',''), 'stock_code': code, 'factor_raw': ratio})
        if i % 100 == 0:
            print(f"  [{i}/{len(codes)}] {time.time()-t0:.0f}s | ok={success} fail={fail}", flush=True)

print(f"  拉取完成: {success}/{len(codes)} ok, {fail} fail, {time.time()-t0:.0f}s", flush=True)
print(f"  记录数: {len(all_records)}", flush=True)

if not all_records:
    print("ERROR: 无数据!", flush=True)
    sys.exit(1)

print("Step3: 20日滚动均值...", flush=True)
df = pd.DataFrame(all_records)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['stock_code','date'])
df['factor_raw'] = df.groupby('stock_code')['factor_raw'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)
df = df.dropna(subset=['factor_raw'])
print(f"  滚动后: {len(df)}行", flush=True)

print("Step4: 市值中性化...", flush=True)
merged = df.merge(df_k[['date','stock_code','close']], on=['date','stock_code'], how='left')
results = []
dates_done = 0
for dt, g in merged.groupby('date'):
    g2 = g.dropna(subset=['factor_raw','close'])
    if len(g2) < 30: continue
    vals = g2['factor_raw'].values.copy()
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    vals = np.clip(vals, lo, hi)
    mcap = np.log(g2['close'].values + 1)
    X = np.column_stack([mcap, np.ones(len(mcap))])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, vals, rcond=None)
        resid = vals - X @ beta
    except:
        resid = vals
    for j, (_, row) in enumerate(g2.iterrows()):
        results.append({'date': dt, 'stock_code': row['stock_code'], 'factor': resid[j]})
    dates_done += 1

out = pd.DataFrame(results)
out.to_csv('data/factor_net_flow_ratio_v1.csv', index=False)
n_d = out['date'].nunique()
n_s = out['stock_code'].nunique()
print(f"Step5: 完成!", flush=True)
print(f"  输出: data/factor_net_flow_ratio_v1.csv", flush=True)
print(f"  {len(out)}行, {n_d}日, {n_s}只", flush=True)
print(f"  因子均值: {out['factor'].mean():.6f}, std: {out['factor'].std():.6f}", flush=True)
