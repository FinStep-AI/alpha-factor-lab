#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PB-ROE 估值偏离度因子构建脚本

策略：
  1. 从东方财富批量获取每个报告期的 BPS(每股净资产) 和 ROE(加权)
  2. 与K线数据合并，计算 PB = close / BPS
  3. 每个截面日期做回归 ln(PB) ~ ROE，取残差 = 估值偏离度
  4. 残差 > 0 表示估值偏高（贵），残差 < 0 表示估值偏低（便宜）
  5. 做多低残差（便宜），做空高残差（贵）→ factor方向为负（越小越好）
  6. 市值中性化

报告期到实际可用日期的映射（T+1月，保守假设）：
  - Q1 (03-31) → 05-01 可用
  - Q2 (06-30) → 09-01 可用
  - Q3 (09-30) → 11-01 可用
  - Q4 (12-31) → 05-01 可用（年报次年4月底前）

输出：data/csi1000_pb_roe_residual.csv (date, stock_code, factor_value)
"""

import json
import os
import sys
import time
import urllib.request
import warnings
from datetime import datetime, date

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ─── 配置 ───
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
KLINE_FILE = os.path.join(DATA_DIR, "csi1000_kline_raw.csv")
CODES_FILE = os.path.join(DATA_DIR, "csi1000_codes.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "csi1000_pb_roe_residual.csv")
FUNDAMENTAL_CACHE = os.path.join(DATA_DIR, "csi1000_fundamental_cache.csv")

# 需要拉取的报告期
REPORT_DATES = [
    "2022-06-30", "2022-09-30", "2022-12-31",
    "2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31",
    "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
    "2025-03-31", "2025-06-30", "2025-09-30",
]

# 报告期 → 数据可用起始日期（保守假设，考虑财报披露周期）
REPORT_AVAILABLE = {
    "03-31": "+2M",   # Q1季报 → 约5月可用
    "06-30": "+3M",   # 中报 → 约9月可用
    "09-30": "+2M",   # Q3季报 → 约11月可用
    "12-31": "+5M",   # 年报 → 约次年5月可用
}

def get_available_date(report_date_str):
    """根据报告期计算数据可用日期"""
    rd = datetime.strptime(report_date_str, "%Y-%m-%d")
    mmdd = report_date_str[5:]
    
    if mmdd == "03-31":
        return datetime(rd.year, 5, 1)
    elif mmdd == "06-30":
        return datetime(rd.year, 9, 1)
    elif mmdd == "09-30":
        return datetime(rd.year, 11, 1)
    elif mmdd == "12-31":
        return datetime(rd.year + 1, 5, 1)
    else:
        return rd


def fetch_fundamental_batch(report_date: str, page_size: int = 500) -> pd.DataFrame:
    """从东方财富批量获取某个报告期的BPS和ROE"""
    all_rows = []
    page = 1
    
    while True:
        url = (
            "https://datacenter-web.eastmoney.com/api/data/v1/get?"
            "reportName=RPT_LICO_FN_CPD&"
            "columns=SECUCODE,SECURITY_CODE,SECURITY_NAME_ABBR,REPORTDATE,BPS,WEIGHTAVG_ROE,TOTAL_OPERATE_INCOME,PARENT_NETPROFIT&"
            f"filter=(REPORTDATE='{report_date}')&"
            f"pageNumber={page}&pageSize={page_size}&"
            "sortColumns=SECURITY_CODE&sortTypes=1&"
            "source=WEB&client=WEB&_=1"
        )
        
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=20)
            data = json.loads(resp.read())
            
            if not data.get('result') or not data['result'].get('data'):
                break
            
            rows = data['result']['data']
            all_rows.extend(rows)
            
            total_pages = data['result'].get('pages', 1)
            if page >= total_pages:
                break
            
            page += 1
            time.sleep(0.3)  # 控制频率
            
        except Exception as e:
            print(f"  [WARN] 拉取 {report_date} page {page} 失败: {e}")
            break
    
    if not all_rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_rows)
    df = df.rename(columns={
        'SECURITY_CODE': 'stock_code_raw',
        'SECUCODE': 'secucode',
        'SECURITY_NAME_ABBR': 'name',
        'REPORTDATE': 'report_date',
        'BPS': 'bps',
        'WEIGHTAVG_ROE': 'roe',
        'TOTAL_OPERATE_INCOME': 'revenue',
        'PARENT_NETPROFIT': 'net_profit',
    })
    
    # 统一stock_code为6位字符串
    df['stock_code'] = df['stock_code_raw'].astype(str).str.zfill(6)
    df['bps'] = pd.to_numeric(df['bps'], errors='coerce')
    df['roe'] = pd.to_numeric(df['roe'], errors='coerce')
    
    return df[['stock_code', 'report_date', 'bps', 'roe', 'name']].copy()


def fetch_all_fundamentals(codes_set: set) -> pd.DataFrame:
    """获取所有报告期的基本面数据（带缓存）"""
    
    # 检查缓存
    if os.path.exists(FUNDAMENTAL_CACHE):
        print(f"[INFO] 加载缓存: {FUNDAMENTAL_CACHE}")
        cached = pd.read_csv(FUNDAMENTAL_CACHE)
        cached['report_date'] = cached['report_date'].str[:10]
        cached_dates = set(cached['report_date'].unique())
        missing_dates = [d for d in REPORT_DATES if d not in cached_dates]
        
        if not missing_dates:
            print(f"[INFO] 缓存完整，{len(cached)}行，{len(cached_dates)}个报告期")
            return cached
        
        print(f"[INFO] 缓存缺少报告期: {missing_dates}")
    else:
        cached = pd.DataFrame()
        missing_dates = REPORT_DATES
    
    # 拉取缺少的报告期
    new_dfs = []
    for rd in missing_dates:
        print(f"  拉取 {rd} ...", end=" ", flush=True)
        df = fetch_fundamental_batch(rd)
        if len(df) > 0:
            # 过滤中证1000成分股
            df_filtered = df[df['stock_code'].isin(codes_set)]
            print(f"全A {len(df)}行, 中证1000 {len(df_filtered)}行")
            new_dfs.append(df_filtered)
        else:
            print("无数据")
        time.sleep(0.5)
    
    # 合并
    if new_dfs:
        new_data = pd.concat(new_dfs, ignore_index=True)
        if len(cached) > 0:
            all_data = pd.concat([cached, new_data], ignore_index=True)
        else:
            all_data = new_data
    else:
        all_data = cached
    
    # 保存缓存
    if len(all_data) > 0:
        all_data.to_csv(FUNDAMENTAL_CACHE, index=False)
        print(f"[INFO] 缓存已保存: {FUNDAMENTAL_CACHE} ({len(all_data)}行)")
    
    return all_data


def build_factor(kline_df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    构建PB-ROE偏离度因子
    
    对每个交易日：
    1. 找到当前可用的最新报告期
    2. 用该报告期的BPS计算 PB = close / BPS
    3. 截面回归 ln(PB) ~ ROE
    4. 残差 = 估值偏离度
    """
    
    # 准备报告期可用日期映射
    report_avail = []
    for rd in sorted(fund_df['report_date'].unique()):
        rd_clean = rd[:10] if len(rd) > 10 else rd
        avail_date = get_available_date(rd_clean)
        report_avail.append((rd_clean, avail_date))
    
    report_avail.sort(key=lambda x: x[1])
    print(f"\n报告期可用日期映射:")
    for rd, avail in report_avail:
        print(f"  {rd} → {avail.strftime('%Y-%m-%d')}")
    
    # 准备K线数据（每天的close和市值代理）
    kline_df['date'] = pd.to_datetime(kline_df['date'])
    kline_df['stock_code'] = kline_df['stock_code'].astype(str).str.zfill(6)
    
    # 市值代理 = close * volume（用于中性化，不完美但可用）
    # 更好的方案是用 amount，已经是成交金额了
    
    # 准备基本面数据
    fund_df['report_date_clean'] = fund_df['report_date'].str[:10]
    
    # 对每个交易日找到可用的最新报告期
    all_dates = sorted(kline_df['date'].unique())
    print(f"\n交易日范围: {all_dates[0]} ~ {all_dates[-1]}, 共{len(all_dates)}天")
    
    results = []
    prev_report = None
    prev_fund_map = None
    
    for dt in all_dates:
        # 找到当前可用的最新报告期
        current_report = None
        for rd, avail in report_avail:
            if dt >= pd.Timestamp(avail):
                current_report = rd
        
        if current_report is None:
            continue
        
        # 如果报告期没变，复用之前的基本面数据
        if current_report != prev_report:
            fund_slice = fund_df[fund_df['report_date_clean'] == current_report]
            prev_fund_map = fund_slice.set_index('stock_code')[['bps', 'roe']].to_dict('index')
            prev_report = current_report
        
        # 当天的K线
        day_kline = kline_df[kline_df['date'] == dt][['stock_code', 'close', 'amount']].copy()
        
        # 合并基本面
        day_kline['bps'] = day_kline['stock_code'].map(
            lambda x: prev_fund_map.get(x, {}).get('bps', np.nan))
        day_kline['roe'] = day_kline['stock_code'].map(
            lambda x: prev_fund_map.get(x, {}).get('roe', np.nan))
        
        # 过滤有效数据
        valid = day_kline.dropna(subset=['close', 'bps', 'roe'])
        valid = valid[(valid['bps'] > 0) & (valid['close'] > 0)]  # BPS和close必须为正
        valid = valid[valid['roe'].notna()]
        
        if len(valid) < 30:  # 样本太少不做回归
            continue
        
        # 计算 PB 和 ln(PB)
        valid = valid.copy()
        valid['pb'] = valid['close'] / valid['bps']
        valid['ln_pb'] = np.log(valid['pb'])
        
        # Winsorize 极端值（MAD方法）
        for col in ['ln_pb', 'roe']:
            median = valid[col].median()
            mad = (valid[col] - median).abs().median()
            upper = median + 5 * 1.4826 * mad
            lower = median - 5 * 1.4826 * mad
            valid[col] = valid[col].clip(lower, upper)
        
        # 截面回归 ln(PB) ~ ROE
        try:
            slope, intercept, r_value, p_value, std_err = sp_stats.linregress(
                valid['roe'].values, valid['ln_pb'].values)
            
            # 残差 = 实际ln(PB) - 预测ln(PB)
            valid['predicted_ln_pb'] = intercept + slope * valid['roe']
            valid['residual'] = valid['ln_pb'] - valid['predicted_ln_pb']
            
            # 市值中性化（用amount作为市值代理）
            valid['ln_amount'] = np.log(valid['amount'].clip(lower=1))
            mask = valid['ln_amount'].notna() & valid['residual'].notna()
            if mask.sum() > 30:
                slope_mv, intercept_mv, _, _, _ = sp_stats.linregress(
                    valid.loc[mask, 'ln_amount'].values, 
                    valid.loc[mask, 'residual'].values)
                valid.loc[mask, 'factor_value'] = (
                    valid.loc[mask, 'residual'] - 
                    (intercept_mv + slope_mv * valid.loc[mask, 'ln_amount']))
            else:
                valid['factor_value'] = valid['residual']
            
            # Z-score标准化
            fv = valid['factor_value']
            mean_fv = fv.mean()
            std_fv = fv.std()
            if std_fv > 0:
                valid['factor_value'] = (fv - mean_fv) / std_fv
            
            # 收集结果
            for _, row in valid.iterrows():
                if pd.notna(row['factor_value']):
                    results.append({
                        'date': dt.strftime('%Y-%m-%d'),
                        'stock_code': row['stock_code'],
                        'factor_value': round(row['factor_value'], 6)
                    })
            
        except Exception as e:
            continue
    
    result_df = pd.DataFrame(results)
    print(f"\n因子值计算完成: {len(result_df)}行, {result_df['date'].nunique()}个交易日")
    return result_df


def main():
    print("=" * 60)
    print("PB-ROE 估值偏离度因子构建")
    print("=" * 60)
    
    # 加载成分股
    codes_df = pd.read_csv(CODES_FILE)
    codes_set = set(codes_df['stock_code'].astype(str).str.zfill(6))
    print(f"中证1000成分股: {len(codes_set)}只")
    
    # 获取基本面数据
    print("\n[Step 1] 获取基本面数据 (BPS + ROE)...")
    fund_df = fetch_all_fundamentals(codes_set)
    print(f"基本面数据: {len(fund_df)}行, 报告期: {sorted(fund_df['report_date'].str[:10].unique())}")
    
    # 加载K线
    print(f"\n[Step 2] 加载K线数据...")
    kline_df = pd.read_csv(KLINE_FILE)
    kline_df['stock_code'] = kline_df['stock_code'].astype(str).str.zfill(6)
    kline_in_pool = kline_df[kline_df['stock_code'].isin(codes_set)]
    print(f"K线: {len(kline_in_pool)}行, {kline_in_pool['stock_code'].nunique()}只, "
          f"{kline_in_pool['date'].min()} ~ {kline_in_pool['date'].max()}")
    
    # 构建因子
    print(f"\n[Step 3] 构建PB-ROE偏离度因子...")
    factor_df = build_factor(kline_in_pool, fund_df)
    
    if len(factor_df) == 0:
        print("[ERROR] 因子构建失败，无数据")
        sys.exit(1)
    
    # 保存
    factor_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] 因子已保存: {OUTPUT_FILE}")
    print(f"  交易日数: {factor_df['date'].nunique()}")
    print(f"  平均每天股票数: {len(factor_df) / factor_df['date'].nunique():.0f}")
    print(f"  因子值范围: [{factor_df['factor_value'].min():.3f}, {factor_df['factor_value'].max():.3f}]")


if __name__ == "__main__":
    main()
