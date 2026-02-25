#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三因子构建与回测脚本
============================
因子1: 分析师预期修正加速度 (analyst_revision_accel)
因子2: 情绪价量复合因子 (sentiment_pv_composite)  
因子3: 资金行为因子 (money_flow_momentum)

股票池: 沪深300
回测期: 2022-10 ~ 2026-02 (~3.3年)
数据源: 东方财富(akshare) + 既有回测框架

Usage:
  python3 scripts/build_three_factors.py --step fetch_kline
  python3 scripts/build_three_factors.py --step fetch_money_flow
  python3 scripts/build_three_factors.py --step build_factors
  python3 scripts/build_three_factors.py --step backtest
  python3 scripts/build_three_factors.py --step all
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
SCRIPTS_DIR = BASE_DIR / "scripts"

DATA_DIR.mkdir(exist_ok=True)


# ============================================================
# Step 1: 获取沪深300成分股 + 日K线数据
# ============================================================

def fetch_csi300_constituents():
    """获取沪深300当前成分股列表"""
    import akshare as ak
    df = ak.index_stock_cons_csindex(symbol='000300')
    codes = df['成分券代码'].tolist()
    print(f"[信息] 获取沪深300成分股 {len(codes)} 只")
    return codes


def fetch_kline_data(codes, start_date='20221001', end_date='20260222'):
    """
    批量获取日K线数据（含换手率、振幅等）
    断点续传：已下载的跳过
    """
    import akshare as ak
    
    cache_file = DATA_DIR / "csi300_kline_raw.parquet"
    progress_file = DATA_DIR / "kline_progress.json"
    
    # 加载已有进度
    done_codes = set()
    all_dfs = []
    if cache_file.exists():
        existing = pd.read_parquet(cache_file)
        done_codes = set(existing['stock_code'].unique())
        all_dfs.append(existing)
        print(f"[信息] 已有 {len(done_codes)} 只股票K线，断点续传...")
    
    remaining = [c for c in codes if c not in done_codes]
    print(f"[信息] 还需下载 {len(remaining)} 只...")
    
    batch_dfs = []
    for i, code in enumerate(remaining):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period='daily',
                start_date=start_date, end_date=end_date,
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
            
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(remaining)}] {code} done")
            
            # 限速：东方财富API限制
            time.sleep(0.3)
            
        except Exception as e:
            print(f"  [警告] {code} 失败: {e}")
            time.sleep(1)
        
        # 每50只保存一次
        if (i + 1) % 50 == 0 and batch_dfs:
            batch_df = pd.concat(batch_dfs, ignore_index=True)
            all_dfs.append(batch_df)
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_parquet(cache_file, index=False)
            print(f"  [保存] 累计 {combined['stock_code'].nunique()} 只")
            batch_dfs = []
    
    # 最终保存
    if batch_dfs:
        all_dfs.append(pd.concat(batch_dfs, ignore_index=True))
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        # 去重
        combined = combined.drop_duplicates(subset=['date', 'stock_code'])
        combined.to_parquet(cache_file, index=False)
        print(f"[完成] K线数据: {combined['stock_code'].nunique()} 只, {len(combined)} 行")
        return combined
    else:
        print("[错误] 没有获取到任何K线数据")
        return None


def fetch_money_flow_data(codes):
    """
    批量获取资金流数据
    东方财富每只股票返回最近120个交易日的资金流
    """
    import akshare as ak
    
    cache_file = DATA_DIR / "csi300_money_flow.parquet"
    progress_file = DATA_DIR / "money_flow_progress.json"
    
    done_codes = set()
    all_dfs = []
    if cache_file.exists():
        existing = pd.read_parquet(cache_file)
        done_codes = set(existing['stock_code'].unique())
        all_dfs.append(existing)
        print(f"[信息] 已有 {len(done_codes)} 只资金流数据，断点续传...")
    
    remaining = [c for c in codes if c not in done_codes]
    print(f"[信息] 还需下载 {len(remaining)} 只资金流...")
    
    batch_dfs = []
    for i, code in enumerate(remaining):
        try:
            # 判断市场
            market = 'sz' if code.startswith(('0', '3')) else 'sh'
            df = ak.stock_individual_fund_flow(stock=code, market=market)
            if df is not None and len(df) > 0:
                df = df.rename(columns={
                    '日期': 'date',
                    '收盘价': 'close',
                    '涨跌幅': 'pct_change',
                    '主力净流入-净额': 'main_net_inflow',
                    '主力净流入-净占比': 'main_net_inflow_pct',
                    '超大单净流入-净额': 'super_large_net',
                    '超大单净流入-净占比': 'super_large_pct',
                    '大单净流入-净额': 'large_net',
                    '大单净流入-净占比': 'large_pct',
                    '中单净流入-净额': 'medium_net',
                    '中单净流入-净占比': 'medium_pct',
                    '小单净流入-净额': 'small_net',
                    '小单净流入-净占比': 'small_pct',
                })
                df['stock_code'] = code
                batch_dfs.append(df)
            
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(remaining)}] {code} done")
            
            time.sleep(0.3)
            
        except Exception as e:
            print(f"  [警告] {code} 资金流失败: {e}")
            time.sleep(1)
        
        if (i + 1) % 50 == 0 and batch_dfs:
            batch_df = pd.concat(batch_dfs, ignore_index=True)
            all_dfs.append(batch_df)
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_parquet(cache_file, index=False)
            print(f"  [保存] 累计 {combined['stock_code'].nunique()} 只")
            batch_dfs = []
    
    if batch_dfs:
        all_dfs.append(pd.concat(batch_dfs, ignore_index=True))
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=['date', 'stock_code'])
        combined.to_parquet(cache_file, index=False)
        print(f"[完成] 资金流数据: {combined['stock_code'].nunique()} 只, {len(combined)} 行")
        return combined
    return None


# ============================================================
# Step 2: 因子构建
# ============================================================

def build_sentiment_pv_factor(kline_df):
    """
    因子2: 情绪价量复合因子
    
    公式: turnover_zscore × pv_corr_5d × amplitude_10d
    
    - turnover_zscore: 5日换手率相对20日均值的z-score
    - pv_corr_5d: 近5日量价相关性 (volume与close的滚动相关)
    - amplitude_10d: 近10日平均振幅
    
    预期方向: 反向（高情绪→过热→后续收益差）
    """
    print("[因子2] 构建情绪价量复合因子...")
    
    df = kline_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date'])
    
    results = []
    for code, group in df.groupby('stock_code'):
        g = group.set_index('date').sort_index()
        if len(g) < 30:
            continue
        
        # 换手率z-score: (5日均换手 - 20日均换手) / 20日换手标准差
        turnover_ma5 = g['turnover'].rolling(5, min_periods=3).mean()
        turnover_ma20 = g['turnover'].rolling(20, min_periods=10).mean()
        turnover_std20 = g['turnover'].rolling(20, min_periods=10).std()
        turnover_zscore = (turnover_ma5 - turnover_ma20) / turnover_std20.replace(0, np.nan)
        
        # 量价相关性: 5日rolling corr(volume, close)
        pv_corr = g['volume'].rolling(5, min_periods=3).corr(g['close'])
        
        # 10日平均振幅 (已经是百分比)
        amp_10d = g['amplitude'].rolling(10, min_periods=5).mean()
        
        # 复合因子 = turnover_zscore × |pv_corr| × amp_10d
        # 用|pv_corr|因为量价正相关和负相关都代表强联动/极端情绪
        factor = turnover_zscore * pv_corr.abs() * amp_10d
        
        temp = pd.DataFrame({
            'date': g.index,
            'stock_code': code,
            'factor': factor.values
        })
        results.append(temp)
    
    result_df = pd.concat(results, ignore_index=True)
    result_df = result_df.dropna(subset=['factor'])
    
    # 截面winsorize + zscore
    def cross_section_process(group):
        vals = group['factor']
        # MAD winsorize
        med = vals.median()
        mad = (vals - med).abs().median()
        if mad > 0:
            scaled_mad = 1.4826 * mad
            vals = vals.clip(med - 3 * scaled_mad, med + 3 * scaled_mad)
        # z-score
        std = vals.std()
        if std > 0:
            vals = (vals - vals.mean()) / std
        group['factor'] = vals
        return group
    
    result_df = result_df.groupby('date', group_keys=False).apply(cross_section_process)
    
    print(f"  有效值: {result_df['factor'].notna().sum()}")
    print(f"  日期范围: {result_df['date'].min()} ~ {result_df['date'].max()}")
    print(f"  股票数: {result_df['stock_code'].nunique()}")
    
    return result_df


def build_money_flow_factor(kline_df, money_flow_df):
    """
    因子3: 资金行为因子
    
    公式: main_net_inflow_pct 的 20日动量 (近5日均值 vs 前20日均值)
    
    由于资金流数据只有最近120天，这个因子回测期较短
    用K线数据构建"大单代理因子"来延长回测期:
      大单代理 = 成交额中由大额交易驱动的比例
      proxy = amount / volume * (high - low) / close
      → 高价差且高均价意味着大单推动
    
    预期方向: 正向（主力持续流入→后续收益好）
    """
    print("[因子3] 构建资金行为因子...")
    
    # 方案A: 用实际资金流数据（近120天）
    if money_flow_df is not None and len(money_flow_df) > 0:
        mf = money_flow_df.copy()
        mf['date'] = pd.to_datetime(mf['date'])
        mf = mf.sort_values(['stock_code', 'date'])
        
        results_mf = []
        for code, group in mf.groupby('stock_code'):
            g = group.set_index('date').sort_index()
            if len(g) < 25:
                continue
            
            # 主力净占比的5日均值
            main_pct_ma5 = g['main_net_inflow_pct'].rolling(5, min_periods=3).mean()
            # 主力净占比的20日均值
            main_pct_ma20 = g['main_net_inflow_pct'].rolling(20, min_periods=10).mean()
            # 动量 = 近期 - 长期
            factor = main_pct_ma5 - main_pct_ma20
            
            temp = pd.DataFrame({
                'date': g.index,
                'stock_code': code,
                'factor': factor.values
            })
            results_mf.append(temp)
        
        mf_result = pd.concat(results_mf, ignore_index=True) if results_mf else pd.DataFrame()
    else:
        mf_result = pd.DataFrame()
    
    # 方案B: 用K线构建大单代理因子（覆盖全回测期）
    df = kline_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date'])
    
    results_proxy = []
    for code, group in df.groupby('stock_code'):
        g = group.set_index('date').sort_index()
        if len(g) < 30:
            continue
        
        # 大单代理 = 均价偏离 × 换手率异常
        # avg_price = amount / volume → 越高说明大单推高均价
        # 但直接用金额/量无意义（股价不同），改用标准化版本:
        # proxy = (成交额 / 成交量 - 收盘价) / 收盘价 → 大单溢价
        avg_price = g['amount'] / g['volume'].replace(0, np.nan)
        price_premium = (avg_price - g['close']) / g['close']
        
        # 5日均值 vs 20日均值
        premium_ma5 = price_premium.rolling(5, min_periods=3).mean()
        premium_ma20 = price_premium.rolling(20, min_periods=10).mean()
        factor = premium_ma5 - premium_ma20
        
        temp = pd.DataFrame({
            'date': g.index,
            'stock_code': code,
            'factor': factor.values
        })
        results_proxy.append(temp)
    
    proxy_result = pd.concat(results_proxy, ignore_index=True)
    
    # 合并：优先用真实资金流，没有的用代理
    if len(mf_result) > 0:
        # 给真实数据和代理数据分别标准化，然后拼接
        print(f"  真实资金流数据: {mf_result['stock_code'].nunique()} 只, {mf_result['date'].nunique()} 天")
        # 近120天用真实数据，之前用代理
        latest_mf_date = mf_result['date'].min()
        proxy_early = proxy_result[proxy_result['date'] < latest_mf_date]
        result_df = pd.concat([proxy_early, mf_result], ignore_index=True)
    else:
        result_df = proxy_result
    
    result_df = result_df.dropna(subset=['factor'])
    
    # 截面处理
    def cross_section_process(group):
        vals = group['factor']
        med = vals.median()
        mad = (vals - med).abs().median()
        if mad > 0:
            scaled_mad = 1.4826 * mad
            vals = vals.clip(med - 3 * scaled_mad, med + 3 * scaled_mad)
        std = vals.std()
        if std > 0:
            vals = (vals - vals.mean()) / std
        group['factor'] = vals
        return group
    
    result_df = result_df.groupby('date', group_keys=False).apply(cross_section_process)
    
    print(f"  有效值: {result_df['factor'].notna().sum()}")
    print(f"  日期范围: {result_df['date'].min()} ~ {result_df['date'].max()}")
    print(f"  股票数: {result_df['stock_code'].nunique()}")
    
    return result_df


def build_analyst_revision_factor(kline_df):
    """
    因子1: 分析师预期修正加速度因子
    
    由于免费API无法获取300只股票的历史一致预期时序数据，
    改用"量价信息中的预期修正代理"：
    
    公式: 盈利公告后超额收益的动量变化（Earnings Surprise Proxy）
    = 20日超额收益加速度 = (近20日超额收益) - (前20日超额收益)
    其中超额收益 = 个股收益 - 等权市场收益
    
    经济逻辑: 如果市场参与者持续上修预期，个股会持续跑赢大盘，
    且超额收益加速说明预期修正在加速。
    
    更精确的版本: 用SUE(Standardized Unexpected Earnings)的代理
    = (近期累计超额 / 超额波动) 的变化率
    
    预期方向: 正向（预期加速上修→后续收益好）
    """
    print("[因子1] 构建分析师预期修正加速度因子...")
    
    df = kline_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date'])
    
    # 先算日收益率
    df['returns'] = df.groupby('stock_code')['close'].pct_change()
    
    # 算等权市场日收益（每日所有股票平均）
    market_ret = df.groupby('date')['returns'].mean().rename('market_return')
    df = df.merge(market_ret, on='date', how='left')
    
    # 超额收益
    df['excess_return'] = df['returns'] - df['market_return']
    
    results = []
    for code, group in df.groupby('stock_code'):
        g = group.set_index('date').sort_index()
        if len(g) < 50:
            continue
        
        # 20日累计超额收益
        cum_excess_20d = g['excess_return'].rolling(20, min_periods=10).sum()
        # 前20日累计超额收益 (lag 20天)
        cum_excess_prev20d = cum_excess_20d.shift(20)
        
        # 超额收益加速度 = 近期 - 前期
        excess_accel = cum_excess_20d - cum_excess_prev20d
        
        # 标准化：除以超额收益波动（过去40天）
        excess_vol = g['excess_return'].rolling(40, min_periods=20).std()
        factor = excess_accel / (excess_vol * np.sqrt(20)).replace(0, np.nan)
        
        temp = pd.DataFrame({
            'date': g.index,
            'stock_code': code,
            'factor': factor.values
        })
        results.append(temp)
    
    result_df = pd.concat(results, ignore_index=True)
    result_df = result_df.dropna(subset=['factor'])
    
    # 截面处理
    def cross_section_process(group):
        vals = group['factor']
        med = vals.median()
        mad = (vals - med).abs().median()
        if mad > 0:
            scaled_mad = 1.4826 * mad
            vals = vals.clip(med - 3 * scaled_mad, med + 3 * scaled_mad)
        std = vals.std()
        if std > 0:
            vals = (vals - vals.mean()) / std
        group['factor'] = vals
        return group
    
    result_df = result_df.groupby('date', group_keys=False).apply(cross_section_process)
    
    print(f"  有效值: {result_df['factor'].notna().sum()}")
    print(f"  日期范围: {result_df['date'].min()} ~ {result_df['date'].max()}")
    print(f"  股票数: {result_df['stock_code'].nunique()}")
    
    return result_df


# ============================================================
# Step 3: 准备回测输入
# ============================================================

def prepare_backtest_inputs(factor_df, kline_df, factor_id, factor_name):
    """
    将因子数据和收益率数据转成回测引擎需要的格式
    """
    print(f"[准备] {factor_name} 回测输入...")
    
    # 因子值 CSV
    factor_path = DATA_DIR / f"{factor_id}_factor.csv"
    factor_out = factor_df[['date', 'stock_code', 'factor']].copy()
    factor_out['date'] = pd.to_datetime(factor_out['date']).dt.strftime('%Y-%m-%d')
    factor_out.to_csv(factor_path, index=False)
    
    # 收益率 CSV
    returns_path = DATA_DIR / f"{factor_id}_returns.csv"
    kline = kline_df.copy()
    kline['date'] = pd.to_datetime(kline['date'])
    kline['returns'] = kline.groupby('stock_code')['close'].pct_change()
    returns_out = kline[['date', 'stock_code', 'returns']].dropna()
    returns_out['date'] = returns_out['date'].dt.strftime('%Y-%m-%d')
    returns_out.to_csv(returns_path, index=False)
    
    print(f"  因子值: {factor_path} ({len(factor_out)} 行)")
    print(f"  收益率: {returns_path} ({len(returns_out)} 行)")
    
    return factor_path, returns_path


# ============================================================
# Step 4: 运行回测
# ============================================================

def run_backtest(factor_id, factor_path, returns_path, 
                 rebalance_freq=5, forward_days=5, cost=0.002):
    """调用回测引擎"""
    import subprocess
    
    output_dir = OUTPUT_DIR / factor_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "backtest_report.json"
    
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "factor_backtest.py"),
        "--factor", str(factor_path),
        "--returns", str(returns_path),
        "--n-groups", "5",
        "--rebalance-freq", str(rebalance_freq),
        "--forward-days", str(forward_days),
        "--cost", str(cost),
        "--output-report", str(report_path),
        "--output-dir", str(output_dir),
        "--factor-name", factor_id,
    ]
    
    print(f"\n{'='*60}")
    print(f"  运行回测: {factor_id}")
    print(f"  调仓: {rebalance_freq}d, 前瞻: {forward_days}d, 成本: {cost}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if report_path.exists():
        with open(report_path, 'r') as f:
            report = json.load(f)
        return report
    return None


# ============================================================
# Step 5: 可视化
# ============================================================

def run_visualizer(factor_id, factor_path):
    """调用可视化模块"""
    import subprocess
    
    output_dir = OUTPUT_DIR / factor_id
    report_path = output_dir / "backtest_report.json"
    
    if not report_path.exists():
        print(f"[跳过] {factor_id} 无回测报告")
        return
    
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "visualizer.py"),
        "--backtest-report", str(report_path),
        "--factor", str(factor_path),
        "--output-dir", str(output_dir),
        "--factor-name", factor_id,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[完成] {factor_id} 图表已生成")
    else:
        print(f"[警告] {factor_id} 图表生成失败: {result.stderr[:200]}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', required=True,
                        choices=['fetch_kline', 'fetch_money_flow', 
                                'build_factors', 'backtest', 'visualize', 'all'])
    args = parser.parse_args()
    
    if args.step in ('fetch_kline', 'all'):
        codes = fetch_csi300_constituents()
        kline_df = fetch_kline_data(codes)
    
    if args.step in ('fetch_money_flow', 'all'):
        if args.step != 'all':
            codes = fetch_csi300_constituents()
        money_flow_df = fetch_money_flow_data(codes