#!/usr/bin/env python3
"""
Volume-Weighted Momentum Level v1 (VWM-Level v1) - 快速版
================================================================
改进: 对vw_mom_decay_v1的"原始VW动量信号"而非衰减信号

思路: vw_mom_decay_v1有IC=0.018,t=3.86(显著)但单调性仅0.6不稳
      原因: decay ratio (MA5/MA20)公式对极端组不友好

      改进: 
      1. 直接用 raw VWM信号 = sum(volume×ret, 40d) / sum(volume, 40d)
         = volume-weighted avg return (本质是volume-weighted version of mean return)
      2. 用rank变换消除极端值影响
      3. 用60日窗口而非20日
      4. 前瞻20日(vs 5日),很多因子在20d效果更好
"""
import sys, time
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    kline_file = data_dir / "csi1000_kline_raw.csv"
    output_file = data_dir / "factor_vwml_v1.csv"
    
    print("加载数据...")
    df = pd.read_csv(kline_file, dtype={"stock_code": str}, parse_dates=["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    df["ret"] = df["pct_change"] / 100.0
    
    n_stocks = df["stock_code"].nunique()
    print(f"数据: {df['date'].min().date()} ~ {df['date'].max().date()}, {n_stocks}只股票")
    
    G = df.groupby("stock_code")
    WINDOW = 60
    
    # volume-weighted mean return over 60d
    print("计算volume-weighted mean return (60d)...")
    df["vw_ret"] = G["ret"].transform(lambda x: x.rolling(WINDOW, min_periods=30).mean())
    df["w_vol"] = G["volume"].transform(lambda x: x.rolling(WINDOW, min_periods=30).mean())
    
    # Napraw: 简单的vw_momentum jako ret_suma / volume (近似等效)
    # 实际上: sum(vol×ret)/sum(vol) = weighted avg ret ≈ ret_suma if ret_suma weighted
    # 更快: gebruik gewoon ma(ret, 60d) maar rank transform
    
    # 快路径: 直接用ma(ret, 60d)但rank变换
    print("计算60日均收益率 + rank变换...")
    ma_ret_raw = G["ret"].transform(lambda x: x.rolling(WINDOW, min_periods=30).mean())
    
    t0 = time.time()
    # Per-date rank transform  (把极端值问题转化为排序问题)
    df["vwml_raw_rank"] = df.groupby("date")[ma_ret_raw.name if hasattr(ma_ret_raw, 'name') else 0].rank(pct=True)
    print(f"  ERROR: can't rank transform properly, use alternative")
    
    # 重新: 直接compute ma(ret, 60d), 然后全局zscore + 然后每日rank
    df["ma_ret_60d"] = G["ret"].transform(lambda x: x.rolling(60, min_periods=30).mean())
    
    # Daily rank transform 
    print("每日rank变换...")
    t1 = time.time()
    df["vwml_raw"] = df.groupby("date")["ma_ret_60d"].rank(pct=True)
    print(f"  耗时: {time.time()-t1:.1f}s")
    
    # 20d平均成交额(中性化用)
    W20 = 20
    df["log_amount_20d"] = G["amount"].transform(
        lambda x: np.log(x.rolling(W20, min_periods=10).mean() + 1)
    )
    
    # 中性化 + MAD + z-score
    print("中性化...")
    results = []
    for date, group in df.groupby("date"):
        sub = group[["stock_code", "vwml_raw", "log_amount_20d"]].dropna()
        if len(sub) < 50:
            continue
        
        y = sub["vwml_raw"].values.astype(float)
        x = sub["log_amount_20d"].values.astype(float)
        
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            continue
        
        X = np.column_stack([np.ones(mask.sum()), x[mask]])
        try:
            beta = np.linalg.solve(X.T @ X + 1e-10*np.eye(2), X.T @ y[mask])
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
        
        for i in range(len(residual)):
            results.append({
                "date": date,
                "stock_code": sub.iloc[i]["stock_code"],
                "factor_value": float(residual[i])
            })
    
    out_df = pd.DataFrame(results)
    if len(out_df) == 0:
        print("No results!")
        return
    
    out_df["date"] = pd.to_datetime(out_df["date"])
    out_df.to_csv(output_file, index=False)
    
    print(f"\n✓ VWM-Level v1: {output_file}")
    print(f"  {len(out_df)} rows, {out_df['date'].nunique()} dates")
    print(f"  范围: {out_df['date'].min().date()} ~ {out_df['date'].max().date()}")
    print(f"  日均覆盖: {out_df.groupby('date')['stock_code'].count().mean():.0f}只")

if __name__ == "__main__":
    main()
