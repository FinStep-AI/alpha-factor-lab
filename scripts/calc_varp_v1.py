#!/usr/bin/env python3
"""
Volume-Acceleration vs Price-Acceleration Factor (VARP)
==============================================================================

核心定义:
  VARP = - log( MA5(volume) / MA20(volume) ) - ( ret_5d / |ret_20d| )
  
  第一部分: -log(MA5/MA20(volume)) = 成交量加速度(正值=加速放量)
  第二部分: ret_5d / |ret_20d| = 短期动量/中期波动率 = 价格加速度(动量强度)
  
  差值 = 成交量加速 - 价格加速
  高正值 = 成交量放大量价不同步 → 信息积累阶段 → 后续正alpha
  低负值 = 价格先行成交量跟涨/缩量上涨 → 动能不足 → 后续负alpha

理论基础:
  在A股中证1000中:
  - 成交量放大通常先于价格 → 未来正alpha
  - 但单纯放量(vol_mom)不work (IC=-0.003)
  - "量价背离"方向更关键: 量起而价未动→信息积累期
  - 减去价格加速度后(标准化): 纯量-价背离信号

对比已有因子:
  - turnover_decel_v1: -log(MA5/MA20(turnover)) — 成交量动量反转
  - 本因子: 在此基础上减去价格动量强度 → 量价协调性

Barra风格: MICRO (微观结构/动量)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    kline_file = data_dir / "csi1000_kline_raw.csv"
    returns_file = data_dir / "csi1000_returns.csv"
    output_file = data_dir / "factor_varp_v1.csv"
    
    print("加载数据...")
    df = pd.read_csv(kline_file, dtype={"stock_code": str}, parse_dates=["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    df["pct_ret"] = df["pct_change"] / 100.0  # pct_change is in %, convert to decimal
    
    n_stocks = df["stock_code"].nunique()
    print(f"数据: {df['date'].min().date()} ~ {df['date'].max().date()}, {n_stocks}只股票")
    
    WINDOW = 20
    SHORT_WINDOW = 5
    MIN_PERIODS = 10
    
    # ========================================
    # Step 1: 成交量加速度 = -log(MA5/MA20(volume))
    # ========================================
    print("计算成交量加速度...")
    g = df.groupby("stock_code")
    
    ma5_vol = g["volume"].transform(lambda x: x.rolling(SHORT_WINDOW, min_periods=5).mean())
    ma20_vol = g["volume"].transform(lambda x: x.rolling(WINDOW, min_periods=MIN_PERIODS).mean())
    df["vol_accel"] = -np.log((ma5_vol / (ma20_vol + 1e-10)).clip(lower=1e-10))
    
    # ========================================
    # Step 2: 价格加速度 = ret_5d / |ret_20d|
    # ========================================
    print("计算价格加速度...")
    ret5 = g["pct_ret"].transform(lambda x: x.rolling(SHORT_WINDOW).sum())
    ret20 = g["pct_ret"].transform(lambda x: x.rolling(WINDOW).sum())
    abs_ret20 = ret20.abs().clip(lower=1e-6)
    df["ret_accel"] = ret5 / abs_ret20
    
    # ========================================
    # Step 3: VARP = vol_accel - ret_accel
    # ========================================
    print("计算VARP因子...")
    df["varp_raw"] = df["vol_accel"] - df["ret_accel"]
    
    # ========================================
    # Neutralization: log(20d avg amount) OLS
    # ========================================
    print("中性化: log_amount_20d OLS + MAD + z-score...")
    
    df["log_amount_20d"] = g["amount"].transform(
        lambda x: np.log(x.rolling(WINDOW, min_periods=MIN_PERIODS).mean() + 1)
    )
    
    results = []
    for date, group in df.groupby("date"):
        sub = group[["stock_code", "varp_raw", "log_amount_20d"]].dropna()
        if len(sub) < 50:
            continue
        
        y = sub["varp_raw"].values.astype(float)
        x = sub["log_amount_20d"].values.astype(float)
        
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            continue
        
        X = np.column_stack([np.ones(mask.sum()), x[mask]])
        try:
            beta = np.linalg.solve(X.T @ X + 1e-10 * np.eye(2), X.T @ y[mask])
            residual = y[mask] - X @ beta
        except:
            continue
        
        # MAD winsorize
        med = np.nanmedian(residual)
        mad = np.nanmedian(np.abs(residual - med))
        if mad > 1e-10:
            scaled = 1.4826 * mad
            residual = np.clip(residual, med - 3*scaled, med + 3*scaled)
        
        # z-score
        std = np.nanstd(residual)
        if std > 1e-4:
            residual = (residual - np.nanmean(residual)) / std
        
        for i, (idx, row) in enumerate(sub.iloc[np.where(mask)[0]].iterrows()):
            results.append({
                "date": date,
                "stock_code": row["stock_code"],
                "factor_value": float(residual[i])
            })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.dropna(subset=["factor_value"])
    result_df["date"] = pd.to_datetime(result_df["date"])
    
    print(f"\n原始因子记录: {len(result_df)}")
    print(f"日期: {result_df['date'].min().date()} ~ {result_df['date'].max().date()}")
    print(f"第1天截面: mean={result_df[result_df['date']==result_df['date'].min()]['factor_value'].mean():.4f}, std={result_df[result_df['date']==result_df['date'].min()]['factor_value'].std():.4f}")
    
    result_df["date_str"] = result_df["date"].dt.strftime("%Y-%m-%d")
    result_df.to_csv(output_file, index=False)
    
    daily_n = result_df.groupby("date")["stock_code"].count()
    print(f"\n✓ 保存: {output_file}")
    print(f"  日均覆盖: {daily_n.mean():.0f}只")
    
    # Quick stats
    print(f"\n因子初步统计:")
    print(f"  全样本: mean={result_df['factor_value'].mean():.4f}, std={result_df['factor_value'].std():.4f}")
    
    # Single-day stats
    day1 = result_df[result_df["date"] == result_df["date"].min()]
    if len(day1) > 0:
        print(f"  首日截面: mean={day1['factor_value'].mean():.4f}, std={day1['factor_value'].std():.4f}")

if __name__ == "__main__":
    main()
