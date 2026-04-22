#!/usr/bin/env python3
"""
成交量-收益率极端日差异因子 (Volume-Cluster Return Contrast, VCRC)
===============================================================================

核心思路:
  成交量分布的"浓淡"差异体现在极端交易日的收益率分布上。
  - 高成交量日(80分位+)往往对应信息事件(公告、事件驱动、知情交易)
  - 低成交量日(20分位-)对应噪音/流日
  
  如果: 高量日收益 > 低量日收益 → 信息驱动市场 (好消息→买入继续)
  如果: 高量日收益 < 低量日收益 → 噪音主导/散户博弈

公式:
  VCRC_raw = mean(ret_{t+1:t+N} | volume_{t-20:t} ≥ 70%分位) 
             - mean(ret_{t+1:t+N} | volume_{t-20:t} ≤ 30%分位)
  
  窗口: 20日滚动 (检测近期20日内的极端成交量)
  前瞻: 5日 (t+1 到 t+5)
  中性化: 成交额OLS中性化 + MAD winsorize + z-score

预期方向:
  正向 → 高量日后续收益更高 = 知情交易持续 → 动量延续
  
Barra风格: MICRO (微观结构/信息效率)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    kline_file = data_dir / "csi1000_kline_raw.csv"
    returns_file = data_dir / "csi1000_returns.csv"
    output_file = data_dir / "factor_vcrc_v1.csv"
    
    print("加载数据...")
    df = pd.read_csv(kline_file, dtype={"stock_code": str}, parse_dates=["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    # Load returns matrix (forward returns)
    ret_df = pd.read_csv(returns_file, parse_dates=["date"])
    ret_matrix = ret_df.pivot_table(index="date", columns="stock_code", values="return")
    
    print(f"数据范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"股票数: {df['stock_code'].nunique()}")
    
    # ========================================
    # Step 1: 计算20日滚动成交量分位阈值
    # ========================================
    WINDOW = 20
    HIGH_Q = 0.70  # 高量日 ≥ 70%分位
    LOW_Q = 0.30   # 低量日 ≤ 30%分位
    FORWARD_DAYS = 5
    
    print(f"\n计算 {WINDOW}日滚动成交量分位...")
    
    # 对每只股票，计算20日滚动窗口的成交量分位数
    def calc_vol_percentiles(group):
        group = group.sort_values("date").copy()
        vol = group["volume"].values
        high_thresh = np.full(len(vol), np.nan)
        low_thresh = np.full(len(vol), np.nan)
        
        for i in range(WINDOW - 1, len(vol)):
            window_vol = vol[i-WINDOW+1:i+1]
            high_thresh[i] = np.percentile(window_vol, HIGH_Q * 100)
            low_thresh[i] = np.percentile(window_vol, LOW_Q * 100)
        
        group["high_vol_thresh"] = high_thresh
        group["low_vol_thresh"] = low_thresh
        return group
    
    df = df.groupby("stock_code", group_keys=False).apply(calc_vol_percentiles)
    
    # 标记高量日/低量日/正常日
    df["is_high_vol"] = (df["volume"] >= df["high_vol_thresh"]).astype(int)
    df["is_low_vol"] = (df["volume"] <= df["low_vol_thresh"]).astype(int)
    
    # 统计
    n_high = df["is_high_vol"].sum()
    n_low = df["is_low_vol"].sum()
    n_total = len(df)
    print(f"  高量日(>=70%分位): {n_high}/{n_total} ({n_high/n_total*100:.1f}%)")
    print(f"  低量日(<=30%分位): {n_low}/{n_total} ({n_low/n_total*100:.1f}%)")
    
    # ========================================
    # Step 2: 计算前瞻N日收益 (使用returns矩阵)
    # ========================================
    print(f"\n计算 {FORWARD_DAYS}日前瞻收益...")
    
    # Convert df to matrix for forward returns
    value_col = "is_high_vol"  # dummy
    high_vol_matrix = df.pivot_table(index="date", columns="stock_code", values="is_high_vol", fill_value=0)
    low_vol_matrix = df.pivot_table(index="date", columns="stock_code", values="is_low_vol", fill_value=0)
    
    # Align with returns matrix
    common_dates = high_vol_matrix.index.intersection(ret_matrix.index)
    high_vol_matrix = high_vol_matrix.loc[common_dates]
    low_vol_matrix = low_vol_matrix.loc[common_dates]
    ret_matrix = ret_matrix.loc[common_dates]
    
    # Compute forward returns from the returns matrix
    log_ret = np.log1p(ret_matrix.clip(lower=-0.999))
    cum_log = log_ret.cumsum()
    forward_cum = cum_log.shift(-FORWARD_DAYS) - cum_log
    forward_ret = np.expm1(forward_cum)
    
    # ========================================
    # Step 3: 计算VCRC因子
    # ========================================
    print("计算VCRC因子...")
    
    factor_records = []
    
    for i in range(WINDOW - 1, len(high_vol_matrix)):
        date = high_vol_matrix.index[i]
        
        # For each stock at this date
        for stock in high_vol_matrix.columns:
            # Check if stock has data
            hv = high_vol_matrix.iloc[i][stock] if stock in high_vol_matrix.columns else 0
            lv = low_vol_matrix.iloc[i][stock] if stock in low_vol_matrix.columns else 0
            fr = forward_ret.iloc[i][stock] if stock in forward_ret.columns else np.nan
            
            if np.isnan(fr) or (hv == 0 and lv == 0):
                continue
            
            factor_records.append({
                "date": date,
                "stock_code": stock,
                "high_vol_flag": hv,
                "low_vol_flag": lv, 
                "fw_ret": fr
            })
    
    factor_df = pd.DataFrame(factor_records)
    
    if len(factor_df) == 0:
        print("ERROR: No factor records generated!")
        return
    
    print(f"  原始记录: {len(factor_df)}")
      # ========================================
    # For each date, compute per-stock factor:
    # contrast = fw_ret when is_high_vol - fw_ret when is_low_vol
    # (This is the "volume-cluster return contrast")
    # ========================================
    
    # Actually, let me rethink. The factor should be per-stock per-date.
    # For a given stock at date t:
    # - Look at its volume distribution over past 20 days
    # - Is 80% of its volume in up days or down days?
    # NO - that's different from volume_cluster
    
    # Let me implement the original VCRC properly:
    # For EACH stock, each date:
    # 1. Compute its own 20-day rolling volume percentile threshold
    # 2. Classify which days were high-volume days (>=70% fractile)
    # 3. Compute forward returns FOR those high-volume days
    # 4. Same for low-volume days
    # 5. Factor = mean(fwd_ret | high_vol_day) - mean(fwd_ret | low_vol_day)
    pass
