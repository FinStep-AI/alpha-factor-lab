#!/usr/bin/env python3
"""
第二期因子: 估值动量 (PE_TTM 20日对数变化率, 市值中性化)
数据源: fintool get_valuation (日频PE_TTM, 覆盖~800天)
"""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== 估值动量因子构建 ===", flush=True)
print("Step1: 加载成分股...", flush=True)
import pandas as pd
import numpy as np
from fintool_client import get_valuation
from concurrent.futures import ThreadPoolExecutor, as_completed

df_k = pd.read_csv('data/csi1000_kline_raw.csv', usecols=['stock_code','date','close'], dtype={'stock_code':str})
df_k['stock_code'] = df_k['stock_code'].str.zfill(6)
df_k['date'] = pd.to_datetime(df_k['date'])
codes = sorted(df_k['stock_code'].unique())
print(f"  {len(codes)}只", flush=True)

print("Step2: 批量拉取PE估值 (5线程)...", flush=True)
all_records = []
success = fail = 0
t0 = time.time()

def fetch(code):
    try:
        rows = get_valuation(code, begin_date='2024-01-01', end_date='2026-03-07')
        return code, rows, None
    except Exception as e:
        return code, None, str(e)[:50]

with ThreadPoolExecutor(max_workers=5) as ex:
    futs = {ex.submit(fetch, c): c for c in codes}
    for i, fut in enumerate(as_completed(futs), 1):
        code, rows, err = fut.result()
        if err or not rows:
            fail += 1; continue
        success += 1
        for r in rows:
            pe = r.get('pe_ttm')
            if pe and pe > 0:
                all_records.append({
                    'date': r.get('trade_date','')[:10],
                    'stock_code': code,
                    'pe_ttm': pe
                })
        if i % 100 == 0:
            print(f"  [{i}/{len(codes)}] {time.time()-t0:.0f}s | ok={success} fail={fail}", flush=True)

print(f"  拉取完成: {success}/{len(codes)} ok, {fail} fail, {time.time()-t0:.0f}s", flush=True)
print(f"  PE记录: {len(all_records)}", flush=True)

if not all_records:
    print("ERROR: 无数据!"); sys.exit(1)

print("Step3: 计算估值动量 (ln PE 20日差分)...", flush=True)
df = pd.DataFrame(all_records)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['stock_code','date'])
df['ln_pe'] = np.log(df['pe_ttm'])
df['factor_raw'] = df.groupby('stock_code')['ln_pe'].transform(lambda x: x.diff(20))
df = df.dropna(subset=['factor_raw'])
# 去极端值
df = df[df['factor_raw'].abs() < 2.0]
print(f"  有效记录: {len(df)}", flush=True)

print("Step4: 市值中性化...", flush=True)
merged = df.merge(df_k[['date','stock_code','close']], on=['date','stock_code'], how='left')
results = []
for dt, g in merged.groupby('date'):
    g2 = g.dropna(subset=['factor_raw','close'])
    if len(g2) < 50: continue
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

out = pd.DataFrame(results)
out.to_csv('data/factor_valuation_momentum_v1.csv', index=False)
print(f"\nStep5: 完成!", flush=True)
print(f"  输出: data/factor_valuation_momentum_v1.csv", flush=True)
print(f"  {len(out)}行, {out['date'].nunique()}日, {out['stock_code'].nunique()}只", flush=True)
print(f"  均值={out['factor'].mean():.6f}, std={out['factor'].std():.6f}", flush=True)
