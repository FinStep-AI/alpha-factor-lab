#!/usr/bin/env python3
"""
换手率-振幅效率因子 (Turnover Amplitude Efficiency, TAE)
===========================================================

公式: neutralize(log(MA20(turnover) / MA20(amplitude + epsilon)), log_amount_20d)

逻辑:
  - 换手率高但振幅低 = 分歧小但参与度高 = 信息传播效率高 = Quality proxy
  - 换手率低但振幅大 = 少量资金推动大幅波动 = 不健康的价格发现
  - 类似Barra Quality概念：盈利稳定性的量价代理
  
预期方向: 正向（高TAE = 高预期收益）

Barra风格: Quality（量价代理）
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    kline_file = data_dir / "csi1000_kline_raw.csv"
    output_file = data_dir / "factor_tae_v1.csv"
    
    print("读取K线数据...")
    df = pd.read_csv(kline_file, dtype={"stock_code": str})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    print(f"数据: {df['date'].min().date()} ~ {df['date'].max().date()}, {df['stock_code'].nunique()}只股票")
    
    # 计算因子
    g = df.groupby("stock_code")
    
    # MA20(turnover)
    df["ma20_turnover"] = g["turnover"].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    
    # MA20(amplitude), amplitude可能是百分比需要加epsilon防止0
    epsilon = 0.01
    df["ma20_amplitude"] = g["amplitude"].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    
    # TAE = log(MA20_turnover / (MA20_amplitude + epsilon))
    df["tae_raw"] = np.log(df["ma20_turnover"] / (df["ma20_amplitude"] + epsilon))
    
    # 中性化变量：log(20日均成交额)
    df["log_amount_20d"] = g["amount"].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean())
    )
    
    # 横截面中性化 + MAD winsorize + z-score
    results = []
    for date, group in df.groupby("date"):
        sub = group[["stock_code", "tae_raw", "log_amount_20d"]].dropna()
        if len(sub) < 50:
            continue
        
        y = sub["tae_raw"].values.astype(float)
        x = sub["log_amount_20d"].values.astype(float)
        
        # OLS中性化
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            continue
        
        X = np.column_stack([np.ones(mask.sum()), x[mask]])
        y_clean = y[mask]
        try:
            beta = np.linalg.solve(X.T @ X + 1e-10 * np.eye(2), X.T @ y_clean)
            residual = np.full_like(y, np.nan)
            residual[mask] = y_clean - X @ beta
        except:
            continue
        
        # MAD winsorize
        med = np.nanmedian(residual)
        mad = np.nanmedian(np.abs(residual - med))
        if mad > 0:
            scaled = 1.4826 * mad
            residual = np.clip(residual, med - 3 * scaled, med + 3 * scaled)
        
        # z-score
        std = np.nanstd(residual)
        if std > 0:
            residual = (residual - np.nanmean(residual)) / std
        
        for i, (_, row) in enumerate(sub.iterrows()):
            results.append({
                "date": date,
                "stock_code": row["stock_code"],
                "factor_value": residual[i] if i < len(residual) else np.nan
            })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.dropna(subset=["factor_value"])
    
    print(f"\n因子统计:")
    print(f"  日期范围: {result_df['date'].min()} ~ {result_df['date'].max()}")
    print(f"  总行数: {len(result_df)}")
    print(f"  截面数: {result_df['date'].nunique()}")
    print(f"  因子值分布: mean={result_df['factor_value'].mean():.4f}, std={result_df['factor_value'].std():.4f}")
    
    # 保存
    result_df["date"] = result_df["date"].dt.strftime("%Y-%m-%d")
    result_df.to_csv(output_file, index=False)
    print(f"\n已保存到: {output_file}")

if __name__ == "__main__":
    main()
