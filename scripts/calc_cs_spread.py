#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corwin-Schultz (2012) High-Low Spread Estimator
论文: "A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices"
      Journal of Finance, Vol 67, pp 719-760

公式:
  对每个2日窗口 (t, t+1):
    β = ln(H_t/L_t)^2 + ln(H_{t+1}/L_{t+1})^2
    γ = ln(max(H_t,H_{t+1}) / min(L_t,L_{t+1}))^2
    α = (sqrt(2β) - sqrt(β)) / (3 - 2*sqrt(2)) - sqrt(γ/(3-2*sqrt(2)))
    S = 2*(exp(α) - 1) / (1 + exp(α))
    S = max(S, 0)  # 截断负值

然后取20日滚动均值，做成交额OLS中性化 + MAD winsorize + z-score。

反向使用：高spread = 高信息不对称 = 高风险溢价 → 做多（正向因子）
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

def corwin_schultz_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """计算单只股票的CS spread序列"""
    ln_hl = np.log(high / low)
    ln_hl_sq = ln_hl ** 2
    
    # β: 连续2日的ln(H/L)^2之和
    beta = ln_hl_sq + ln_hl_sq.shift(-1)
    
    # γ: 2日合并窗口的ln(H_max/L_min)^2
    h_2d = pd.concat([high, high.shift(-1)], axis=1).max(axis=1)
    l_2d = pd.concat([low, low.shift(-1)], axis=1).min(axis=1)
    gamma = np.log(h_2d / l_2d) ** 2
    
    # α
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    
    # spread
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    spread = spread.clip(lower=0)  # 截断负值
    
    return spread

def main():
    data_dir = Path(__file__).parent.parent / "data"
    kline_path = data_dir / "csi1000_kline_raw.csv"
    output_path = data_dir / "factor_cs_spread_v1.csv"
    
    window = 20  # 滚动窗口
    
    print("读取K线数据...")
    df = pd.read_csv(kline_path, dtype={"stock_code": str})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    # 过滤异常数据
    df = df[(df["high"] > 0) & (df["low"] > 0) & (df["high"] >= df["low"])].copy()
    
    print(f"数据范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"股票数: {df['stock_code'].nunique()}")
    
    # 计算每只股票的CS spread
    print("计算Corwin-Schultz spread...")
    results = []
    for code, grp in df.groupby("stock_code"):
        grp = grp.sort_values("date").copy()
        spread = corwin_schultz_spread(grp["high"], grp["low"])
        
        # 20日滚动均值
        spread_ma = spread.rolling(window, min_periods=int(window * 0.7)).mean()
        
        tmp = grp[["date", "stock_code"]].copy()
        tmp["cs_spread_raw"] = spread_ma.values
        results.append(tmp)
    
    factor_df = pd.concat(results, ignore_index=True)
    factor_df = factor_df.dropna(subset=["cs_spread_raw"])
    
    print(f"原始因子: {factor_df.shape[0]} 行")
    print(f"CS spread 均值: {factor_df['cs_spread_raw'].mean():.6f}, 中位数: {factor_df['cs_spread_raw'].median():.6f}")
    
    # 对数变换（spread > 0 且分布右偏）
    factor_df["cs_spread_log"] = np.log1p(factor_df["cs_spread_raw"])
    
    # 成交额中性化
    print("成交额中性化...")
    # 合并成交额
    amt_df = df.groupby(["date", "stock_code"])["amount"].first().reset_index()
    # 计算20日平均成交额
    amt_pivot = amt_df.pivot(index="date", columns="stock_code", values="amount")
    amt_ma20 = amt_pivot.rolling(window, min_periods=int(window * 0.7)).mean()
    amt_long = amt_ma20.stack().reset_index()
    amt_long.columns = ["date", "stock_code", "amount_ma20"]
    
    factor_df = factor_df.merge(amt_long, on=["date", "stock_code"], how="left")
    factor_df["log_amount"] = np.log(factor_df["amount_ma20"].clip(lower=1))
    factor_df = factor_df.dropna(subset=["log_amount"])
    
    # 截面OLS中性化
    from sklearn.linear_model import LinearRegression
    
    neutralized = []
    for dt, grp in factor_df.groupby("date"):
        if len(grp) < 50:
            continue
        x = grp["log_amount"].values.reshape(-1, 1)
        y = grp["cs_spread_log"].values
        
        # MAD winsorize
        med = np.median(y)
        mad = np.median(np.abs(y - med))
        if mad > 0:
            y_clip = np.clip(y, med - 5 * 1.4826 * mad, med + 5 * 1.4826 * mad)
        else:
            y_clip = y
        
        lr = LinearRegression()
        lr.fit(x, y_clip)
        resid = y_clip - lr.predict(x)
        
        # z-score
        std = resid.std()
        if std > 0:
            resid = (resid - resid.mean()) / std
        
        tmp = grp[["date", "stock_code"]].copy()
        tmp["factor_value"] = resid
        neutralized.append(tmp)
    
    result = pd.concat(neutralized, ignore_index=True)
    result = result.sort_values(["date", "stock_code"]).reset_index(drop=True)
    
    print(f"最终因子: {result.shape[0]} 行, {result['stock_code'].nunique()} 只股票")
    print(f"日期范围: {result['date'].min().date()} ~ {result['date'].max().date()}")
    print(f"因子均值: {result['factor_value'].mean():.4f}, std: {result['factor_value'].std():.4f}")
    
    result.to_csv(output_path, index=False)
    print(f"保存到: {output_path}")

if __name__ == "__main__":
    main()
