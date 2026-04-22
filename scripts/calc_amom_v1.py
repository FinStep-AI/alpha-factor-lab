#!/usr/bin/env python3
"""
Amplitude-Adjusted Momentum v1 (AMOM v1)
===============================================================================

核心思想:
  纯收益率动量(Momentum)在中证1000上效果一般
  --- 除非考虑"收益率是如何实现的" ---
  
  高振幅 + 高收益 = 动量强但波动大 = 高风险溢价
  低振幅 + 高收益 = 稳定上涨 = Quality momentum
  
因子定义:
  AMOM = rank( return_60d ) - rank( amplitude_60d )
  
  高AMOM = 高收益 + 低振幅 = 稳定上涨 = 最强alpha
  低AMOM = 低收益 + 高振幅 = 混乱震荡 = 最差alpha
  
本质: 收益率与波动率的"净动量"——消除了波动率混淆的收益动量。
    与ret_ir_v1(mean/std)不同,这里使用绝对amplitude而非波动率标准差,
    对极端事件更鲁棒,且amplitude在A股涨跌停制度下比std更有区分度。

Note: 本质上类似"调整波动率后的动量",但用日内价格范围代替已实现波动率
(position range = 最接近realized volatility的量价代理)

Barra风格: 在Momentum和Quality之间(偏Momentum)
"""

import sys, time
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    kline_file = data_dir / "csi1000_kline_raw.csv"
    output_file = data_dir / "factor_amom_v1.csv"
    
    print("加载数据...")
    df = pd.read_csv(kline_file, dtype={"stock_code": str}, parse_dates=["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    df["ret"] = df["pct_change"] / 100.0
    
    n_stocks = df["stock_code"].nunique()
    print(f"数据: {df['date'].min().date()} ~ {df['date'].max().date()}, {n_stocks}只股票")
    
    G = df.groupby("stock_code")
    W = 60  # 60日窗口
    MIN_PERIODS = 30
    
    # MA60(return) — 60日均收益
    print("计算MA60(return)...")
    df["ma60_ret"] = G["ret"].transform(lambda x: x.rolling(W, min_periods=MIN_PERIODS).mean())
    
    # MA60(amplitude) — 60日均振幅 (%
    print("计算MA60(amplitude)...")
    df["ma60_amp"] = G["amplitude"].transform(
        lambda x: x.rolling(W, min_periods=MIN_PERIODS).mean()
    )
    
    # 有效率数据
    valid_mask = np.isfinite(df["ma60_ret"]) & np.isfinite(df["ma60_amp"]) & (df["ma60_amp"] > 0)
    df = df[valid_mask].copy()
    
    if len(df) == 0:
        print("ERROR: 有效数据为0")
        return
    
    # ========================================
    # AMOM = rank(ma60_ret) - rank(ma60_amp)
    #      高值 = 高收益 + 低振幅 = 质量动量
    # ========================================
    print("计算AMOM = rank(ret_60d) - rank(amp_60d)...")
    df["rank_ret_60d"] = df.groupby("date")["ma60_ret"].rank(pct=True)
    df["rank_amp_60d"] = df.groupby("date")["ma60_amp"].rank(pct=True)
    df["amom_raw"] = df["rank_ret_60d"] - df["rank_amp_60d"]
    
    # 20日成交额中性化用
    AMT_W = 20
    df["log_amount_20d"] = G["amount"].transform(
        lambda x: np.log(x.rolling(AMT_W, min_periods=10).mean() + 1)
    )
    
    # ========================================
    # 中性化: OLS + MAD winsorize + z-score
    # ========================================
    print("中性化处理...")
    results = []
    
    sub_cols = ["stock_code", "amom_raw", "log_amount_20d"]
    
    t0 = time.time()
    date_count = 0
    for date, group in df.groupby("date"):
        sub = group[sub_cols].dropna()
        if len(sub) < 50:
            continue
        date_count += 1
        
        y = sub["amom_raw"].values.astype(float)
        x = sub["log_amount_20d"].values.astype(float)
        
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            continue
        
        X = np.column_stack([np.ones(mask.sum()), x[mask]])
        try:
            beta = np.linalg.solve(X.T @ X + 1e-10*np.eye(2), X.T @ y[mask])
            residual = y[mask] - X @ beta
        except Exception as e:
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
        
        for i in range(len(residual)):
            row_idx = sub.iloc[np.where(mask)[0]].index[i]
            results.append({
                "date": date,
                "stock_code": df.loc[row_idx, "stock_code"],
                "factor_value": float(residual[i])
            })
    
    print(f"  {date_count} dates processed in {time.time()-t0:.1f}s")
    
    if not results:
        print("ERROR: No results!")
        return
    
    out_df = pd.DataFrame(results)
    out_df = out_df.dropna(subset=["factor_value"])
    out_df["date"] = pd.to_datetime(out_df["date"])
    
    print(f"\n✓ AMOM v1: {len(out_df)} rows")
    print(f"  日期: {out_df['date'].min().date()} ~ {out_df['date'].max().date()}")
    print(f"  截面数: {out_df['date'].nunique()}")
    print(f"  日均覆盖: {out_df.groupby('date')['stock_code'].count().mean():.0f}只")
    print(f"  值范围: [{out_df['factor_value'].min():.4f}, {out_df['factor_value'].max():.4f}]")
    
    out_df["date_str"] = out_df["date"].dt.strftime("%Y-%m-%d")
    out_df.to_csv(output_file, index=False)
    print(f"\n✓ 保存: {output_file}")

if __name__ == "__main__":
    main()
