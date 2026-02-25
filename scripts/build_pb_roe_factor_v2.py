#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PB-ROE 估值偏离度因子 — 快速构建版（向量化）
输出：data/csi1000_pb_roe_residual.csv
"""
import os, sys, warnings, time
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

DATA_DIR = "data"
KLINE_FILE = os.path.join(DATA_DIR, "csi1000_kline_raw.csv")
CODES_FILE = os.path.join(DATA_DIR, "csi1000_codes.csv")
FUND_CACHE = os.path.join(DATA_DIR, "csi1000_fundamental_cache.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "csi1000_pb_roe_residual.csv")

# 报告期 → 数据可用日期
def avail_date(rd):
    from datetime import datetime
    mm = rd[5:10]
    y = int(rd[:4])
    if mm == "03-31": return f"{y}-05-01"
    elif mm == "06-30": return f"{y}-09-01"
    elif mm == "09-30": return f"{y}-11-01"
    elif mm == "12-31": return f"{y+1}-05-01"
    return rd

def main():
    t0 = time.time()
    print("PB-ROE 估值偏离度因子构建（向量化版）")
    
    # 加载
    codes = set(pd.read_csv(CODES_FILE)['stock_code'].astype(str).str.zfill(6))
    fund = pd.read_csv(FUND_CACHE)
    fund['report_date'] = fund['report_date'].str[:10]
    fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
    
    kline = pd.read_csv(KLINE_FILE)
    kline['date'] = pd.to_datetime(kline['date'])
    kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
    kline = kline[kline['stock_code'].isin(codes)]
    
    print(f"K线: {len(kline)}行, {kline['date'].min().date()} ~ {kline['date'].max().date()}")
    print(f"基本面: {len(fund)}行, {fund['report_date'].nunique()}个报告期")
    
    # 报告期排序
    report_dates = sorted(fund['report_date'].unique())
    ra = [(rd, avail_date(rd)) for rd in report_dates]
    ra.sort(key=lambda x: x[1])
    print(f"报告期映射: {len(ra)}个")
    
    # 为每个交易日分配对应的报告期
    all_dates = sorted(kline['date'].unique())
    date_to_report = {}
    for dt in all_dates:
        dt_str = str(dt.date())
        current = None
        for rd, av in ra:
            if dt_str >= av:
                current = rd
        if current:
            date_to_report[dt] = current
    
    print(f"可匹配交易日: {len(date_to_report)} / {len(all_dates)}")
    
    # 按报告期分组处理
    report_groups = {}
    for dt, rd in date_to_report.items():
        report_groups.setdefault(rd, []).append(dt)
    
    results = []
    for rd, dates_in_group in sorted(report_groups.items()):
        fund_slice = fund[fund['report_date'] == rd].set_index('stock_code')
        fund_slice = fund_slice[fund_slice['bps'] > 0]
        fund_slice = fund_slice[fund_slice['roe'].notna()]
        
        if len(fund_slice) < 30:
            continue
        
        kline_group = kline[kline['date'].isin(dates_in_group)].copy()
        kline_group = kline_group.merge(
            fund_slice[['bps', 'roe']],
            left_on='stock_code', right_index=True, how='inner'
        )
        kline_group = kline_group[kline_group['close'] > 0]
        kline_group['pb'] = kline_group['close'] / kline_group['bps']
        kline_group['ln_pb'] = np.log(kline_group['pb'])
        kline_group['ln_amount'] = np.log(kline_group['amount'].clip(lower=1))
        
        # 按天做截面回归
        for dt, day_df in kline_group.groupby('date'):
            if len(day_df) < 30:
                continue
            
            d = day_df[['stock_code', 'ln_pb', 'roe', 'ln_amount']].dropna().copy()
            if len(d) < 30:
                continue
            
            # Winsorize (MAD)
            for col in ['ln_pb', 'roe']:
                med = d[col].median()
                mad = (d[col] - med).abs().median()
                bound = 5 * 1.4826 * mad
                d[col] = d[col].clip(med - bound, med + bound)
            
            # 截面回归 ln(PB) ~ ROE
            try:
                sl, ic, _, _, _ = sp_stats.linregress(d['roe'].values, d['ln_pb'].values)
                d['residual'] = d['ln_pb'] - (ic + sl * d['roe'])
                
                # 市值中性化
                sl2, ic2, _, _, _ = sp_stats.linregress(d['ln_amount'].values, d['residual'].values)
                d['factor_raw'] = d['residual'] - (ic2 + sl2 * d['ln_amount'])
                
                # Z-score
                m, s = d['factor_raw'].mean(), d['factor_raw'].std()
                if s > 0:
                    d['factor_value'] = (d['factor_raw'] - m) / s
                else:
                    continue
                
                dt_str = dt.strftime('%Y-%m-%d') if hasattr(dt, 'strftime') else str(dt)[:10]
                for _, row in d.iterrows():
                    results.append((dt_str, row['stock_code'], round(row['factor_value'], 6)))
            except:
                continue
        
        print(f"  {rd}: {len(dates_in_group)}天处理完成")
    
    df = pd.DataFrame(results, columns=['date', 'stock_code', 'factor_value'])
    df.to_csv(OUTPUT_FILE, index=False)
    
    elapsed = time.time() - t0
    print(f"\n[OK] {OUTPUT_FILE}: {len(df)}行, {df['date'].nunique()}天, {elapsed:.1f}秒")
    print(f"  平均每天: {len(df)/max(df['date'].nunique(),1):.0f}只")
    print(f"  因子范围: [{df['factor_value'].min():.3f}, {df['factor_value'].max():.3f}]")

if __name__ == "__main__":
    main()
