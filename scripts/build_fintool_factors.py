#!/usr/bin/env python3
"""
第二期改造：基于fintool新数据源构建因子
- net_flow_ratio: 主力资金净流入占比
- valuation_momentum: PE_TTM变化率（估值动量）
- shareholder_concentration: 股东数变化率（负向=集中=利好）

用法:
  python3 scripts/build_fintool_factors.py --factor net_flow_ratio --output data/factor_net_flow_ratio.csv
  python3 scripts/build_fintool_factors.py --factor valuation_momentum --output data/factor_valuation_momentum.csv
  python3 scripts/build_fintool_factors.py --factor shareholder_concentration --output data/factor_shareholder_concentration.csv
  python3 scripts/build_fintool_factors.py --factor all  # 构建全部

数据源: fintool API（聚源）
"""
import sys
import os
import csv
import json
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fintool_client import (get_net_flow, get_valuation, get_shareholder_count,
                             get_kline, get_constituents)

KLINE_FILE = "data/csi1000_kline_raw.csv"


def load_codes():
    """从K线文件加载成分股列表"""
    df = pd.read_csv(KLINE_FILE, usecols=["stock_code"])
    codes = sorted(df["stock_code"].astype(str).str.zfill(6).unique())
    print(f"[INFO] 成分股: {len(codes)} 只")
    return codes


def load_kline():
    """加载K线数据（用于市值中性化）"""
    df = pd.read_csv(KLINE_FILE, dtype={"stock_code": str})
    df["stock_code"] = df["stock_code"].str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    return df


def neutralize_by_market_cap(factor_df, kline_df):
    """市值中性化: OLS回归取残差
    factor_df: columns=[date, stock_code, factor_raw]
    kline_df: 原始K线（含close, volume/amount推算市值代理）
    """
    from scipy.stats import mstats

    # 用 close * volume 作为市值代理（没有真实市值数据时）
    # 更好的方式：用amount（成交额）做代理，或直接用close做排序
    # 这里用 ln(close) 作为市值代理（与市值高相关）
    merged = factor_df.merge(
        kline_df[["date", "stock_code", "close", "amount"]],
        on=["date", "stock_code"], how="left"
    )

    results = []
    for dt, group in merged.groupby("date"):
        g = group.dropna(subset=["factor_raw", "close"])
        if len(g) < 30:
            continue
        
        # 缩尾 5%
        vals = g["factor_raw"].values.copy()
        lo, hi = np.nanpercentile(vals, [2.5, 97.5])
        vals = np.clip(vals, lo, hi)
        
        # 市值代理: ln(close)
        mcap = np.log(g["close"].values + 1)
        
        # OLS: factor ~ mcap
        X = np.column_stack([mcap, np.ones(len(mcap))])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, vals, rcond=None)
            residual = vals - X @ beta
        except:
            residual = vals
        
        for i, (_, row) in enumerate(g.iterrows()):
            results.append({
                "date": dt,
                "stock_code": row["stock_code"],
                "factor": residual[i]
            })
    
    return pd.DataFrame(results)


# ============================================================
# 因子1: 主力资金净流入占比
# ============================================================
def build_net_flow_ratio(codes, start_date, end_date, max_workers=10):
    """
    因子定义: major_net_flow / (super_in + large_in + medium_in + little_in)
    即主力净流入占总流入的比例，20日均值
    """
    print(f"\n[因子] net_flow_ratio: 主力资金净流入占比")
    print(f"  期间: {start_date} ~ {end_date}, {len(codes)}只")
    
    all_records = []
    success = 0
    fail = 0
    t0 = time.time()
    
    def fetch_one(code):
        try:
            rows = get_net_flow(code, start_date, end_date)
            return code, rows, None
        except Exception as e:
            return code, None, str(e)[:60]
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_one, c): c for c in codes}
        for i, fut in enumerate(as_completed(futs), 1):
            code, rows, err = fut.result()
            if err or not rows:
                fail += 1
                continue
            success += 1
            for r in rows:
                total_in = (abs(r.get("super_in", 0) or 0) + 
                           abs(r.get("large_in", 0) or 0) +
                           abs(r.get("medium_in", 0) or 0) + 
                           abs(r.get("little_in", 0) or 0))
                major_net = r.get("major_net_flow_in", 0) or 0
                ratio = major_net / total_in if total_in > 0 else 0
                all_records.append({
                    "date": r.get("date", ""),
                    "stock_code": code,
                    "factor_raw": ratio
                })
            if i % 100 == 0:
                elapsed = time.time() - t0
                print(f"  [{i}/{len(codes)}] {elapsed:.1f}s | 成功={success}")
    
    elapsed = time.time() - t0
    print(f"  完成: {success}/{len(codes)} 成功, {fail}失败, {elapsed:.1f}s")
    
    if not all_records:
        print("  [WARN] 无数据!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"])
    
    # 20日滚动均值
    df = df.sort_values(["stock_code", "date"])
    df["factor_raw"] = df.groupby("stock_code")["factor_raw"].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    df = df.dropna(subset=["factor_raw"])
    
    return df


# ============================================================
# 因子2: 估值动量 (PE_TTM变化率)
# ============================================================
def build_valuation_momentum(codes, start_date, end_date, max_workers=10):
    """
    因子定义: PE_TTM的20日变化率（对数差分）
    PE加速上涨可能意味着情绪推动，反转信号
    """
    print(f"\n[因子] valuation_momentum: PE_TTM变化率")
    print(f"  期间: {start_date} ~ {end_date}, {len(codes)}只")
    
    all_records = []
    success = 0
    fail = 0
    t0 = time.time()
    
    def fetch_one(code):
        try:
            rows = get_valuation(code, begin_date=start_date, end_date=end_date)
            return code, rows, None
        except Exception as e:
            return code, None, str(e)[:60]
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_one, c): c for c in codes}
        for i, fut in enumerate(as_completed(futs), 1):
            code, rows, err = fut.result()
            if err or not rows:
                fail += 1
                continue
            success += 1
            for r in rows:
                pe = r.get("pe_ttm")
                if pe is not None and pe > 0:
                    trade_date = r.get("trade_date", "")[:10]
                    all_records.append({
                        "date": trade_date,
                        "stock_code": code,
                        "pe_ttm": pe
                    })
            if i % 100 == 0:
                elapsed = time.time() - t0
                print(f"  [{i}/{len(codes)}] {elapsed:.1f}s | 成功={success}")
    
    elapsed = time.time() - t0
    print(f"  完成: {success}/{len(codes)} 成功, {fail}失败, {elapsed:.1f}s")
    
    if not all_records:
        print("  [WARN] 无数据!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"])
    
    # 对数PE的20日差分（变化率）
    df["ln_pe"] = np.log(df["pe_ttm"])
    df["factor_raw"] = df.groupby("stock_code")["ln_pe"].transform(
        lambda x: x.diff(20)
    )
    df = df.dropna(subset=["factor_raw"])
    # 去除极端值（PE变化率>200%的剔除）
    df = df[df["factor_raw"].abs() < 2.0]
    
    return df[["date", "stock_code", "factor_raw"]]


# ============================================================
# 因子3: 股东集中度变化
# ============================================================
def build_shareholder_concentration(codes, max_workers=10):
    """
    因子定义: 股东数量环比变化率（负值=股东减少=筹码集中=利好）
    数据频率：季报（低频），适合做长期因子
    """
    print(f"\n[因子] shareholder_concentration: 股东数变化率")
    
    all_records = []
    success = 0
    fail = 0
    t0 = time.time()
    
    def fetch_one(code):
        try:
            rows = get_shareholder_count(code)
            return code, rows, None
        except Exception as e:
            return code, None, str(e)[:60]
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_one, c): c for c in codes}
        for i, fut in enumerate(as_completed(futs), 1):
            code, rows, err = fut.result()
            if err or not rows:
                fail += 1
                continue
            success += 1
            for r in rows:
                sh_count = r.get("total_shareholder_number")
                mom_str = r.get("total_shareholder_number_mom", "")
                end_date = r.get("end_date", "")[:10]
                
                if sh_count and mom_str:
                    # 解析环比变化率
                    try:
                        mom = float(str(mom_str).replace("%", "")) / 100
                    except:
                        mom = None
                    
                    if mom is not None:
                        all_records.append({
                            "date": end_date,
                            "stock_code": code,
                            "factor_raw": mom,
                            "sh_count": sh_count
                        })
            if i % 100 == 0:
                elapsed = time.time() - t0
                print(f"  [{i}/{len(codes)}] {elapsed:.1f}s | 成功={success}")
    
    elapsed = time.time() - t0
    print(f"  完成: {success}/{len(codes)} 成功, {fail}失败, {elapsed:.1f}s")
    
    if not all_records:
        print("  [WARN] 无数据!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"])
    
    # 注意：股东数据是季频，需要forward fill到日频
    # 这里先返回原始季频数据，后续在回测时做填充
    return df[["date", "stock_code", "factor_raw"]]


def main():
    parser = argparse.ArgumentParser(description="fintool新数据因子构建")
    parser.add_argument("--factor", required=True, 
                       choices=["net_flow_ratio", "valuation_momentum", 
                               "shareholder_concentration", "all"],
                       help="要构建的因子")
    parser.add_argument("--output", help="输出CSV路径（默认 data/factor_{name}.csv）")
    parser.add_argument("--start-date", default="2024-01-01", help="开始日期")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"), help="结束日期")
    parser.add_argument("--workers", type=int, default=10, help="并发数")
    parser.add_argument("--neutralize", action="store_true", default=True, help="市值中性化")
    parser.add_argument("--no-neutralize", dest="neutralize", action="store_false")
    args = parser.parse_args()
    
    codes = load_codes()
    kline_df = load_kline() if args.neutralize else None
    
    factors_to_build = [args.factor] if args.factor != "all" else [
        "net_flow_ratio", "valuation_momentum", "shareholder_concentration"
    ]
    
    for factor_name in factors_to_build:
        print(f"\n{'='*60}")
        print(f"构建因子: {factor_name}")
        print(f"{'='*60}")
        
        if factor_name == "net_flow_ratio":
            raw_df = build_net_flow_ratio(codes, args.start_date, args.end_date, args.workers)
        elif factor_name == "valuation_momentum":
            raw_df = build_valuation_momentum(codes, args.start_date, args.end_date, args.workers)
        elif factor_name == "shareholder_concentration":
            raw_df = build_shareholder_concentration(codes, args.workers)
        else:
            continue
        
        if raw_df.empty:
            print(f"[SKIP] {factor_name} 无数据")
            continue
        
        # 市值中性化
        if args.neutralize and kline_df is not None and factor_name != "shareholder_concentration":
            print(f"  市值中性化...")
            factor_df = neutralize_by_market_cap(raw_df, kline_df)
        else:
            factor_df = raw_df.rename(columns={"factor_raw": "factor"})
        
        # 输出
        output = args.output or f"data/factor_{factor_name}.csv"
        factor_df.to_csv(output, index=False)
        
        # 统计
        n_dates = factor_df["date"].nunique()
        n_stocks = factor_df["stock_code"].nunique()
        print(f"\n  输出: {output}")
        print(f"  覆盖: {n_dates}个交易日, {n_stocks}只股票, {len(factor_df)}条记录")
        print(f"  因子均值: {factor_df['factor'].mean():.6f}")
        print(f"  因子标准差: {factor_df['factor'].std():.6f}")


if __name__ == "__main__":
    main()
