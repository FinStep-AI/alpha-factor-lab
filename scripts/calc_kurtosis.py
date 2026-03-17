#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Return Kurtosis Factor (收益率峰度因子)

学术背景:
  - Dittmar (2002) "Nonlinear Pricing Kernels, Kurtosis Preference, and Evidence 
    from the Cross Section of Equity Returns", JF
  - Conrad, Dittmar & Ghysels (2013) "Ex Ante Skewness and Expected Stock Returns", JF
  
核心思路:
  高峰度(leptokurtic) = 收益分布尾部厚 = "黑天鹅"风险大
  投资者厌恶高峰度 → 高峰度股票需要更高风险补偿 → 正向因子
  
  但在A股散户驱动的小盘(中证1000)中，可能相反：
  高峰度 = 极端收益出现频繁 → 投机吸引 → 高估 → 反向因子

计算: 
  20日滚动收益率的excess kurtosis (Fisher, 减3)
  做成交额OLS中性化 + MAD winsorize + z-score
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kurtosis as sp_kurtosis
from sklearn.linear_model import LinearRegression

def main():
    data_dir = Path(__file__).parent.parent / "data"
    kline_path = data_dir / "csi1000_kline_raw.csv"
    
    window = 20
    
    print("读取数据...")
    df = pd.read_csv(kline_path, dtype={"stock_code": str})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    # 日收益率
    df["ret"] = df.groupby("stock_code")["close"].pct_change()
    
    print("计算峰度因子...")
    results = []
    for code, grp in df.groupby("stock_code"):
        grp = grp.sort_values("date").copy()
        
        # 20日滚动excess kurtosis
        kurt = grp["ret"].rolling(window, min_periods=int(window * 0.7)).apply(
            lambda x: sp_kurtosis(x, fisher=True, bias=False) if len(x) >= 10 else np.nan,
            raw=True
        )
        
        tmp = grp[["date", "stock_code"]].copy()
        tmp["kurt_raw"] = kurt.values
        results.append(tmp)
    
    factor_df = pd.concat(results, ignore_index=True)
    factor_df = factor_df.dropna(subset=["kurt_raw"])
    
    print(f"原始因子: {factor_df.shape[0]} 行")
    print(f"峰度均值: {factor_df['kurt_raw'].mean():.4f}, 中位数: {factor_df['kurt_raw'].median():.4f}")
    
    # 成交额中性化
    print("成交额中性化...")
    amt_df = df.groupby(["date", "stock_code"])["amount"].first().reset_index()
    amt_pivot = amt_df.pivot(index="date", columns="stock_code", values="amount")
    amt_ma20 = amt_pivot.rolling(window, min_periods=int(window * 0.7)).mean()
    amt_long = amt_ma20.stack().reset_index()
    amt_long.columns = ["date", "stock_code", "amount_ma20"]
    
    factor_df = factor_df.merge(amt_long, on=["date", "stock_code"], how="left")
    factor_df["log_amount"] = np.log(factor_df["amount_ma20"].clip(lower=1))
    factor_df = factor_df.dropna(subset=["log_amount"])
    
    neutralized = []
    for dt, grp in factor_df.groupby("date"):
        if len(grp) < 50:
            continue
        x = grp["log_amount"].values.reshape(-1, 1)
        y = grp["kurt_raw"].values
        
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
        
        std = resid.std()
        if std > 0:
            resid = (resid - resid.mean()) / std
        
        tmp = grp[["date", "stock_code"]].copy()
        tmp["factor_value"] = resid
        neutralized.append(tmp)
    
    result = pd.concat(neutralized, ignore_index=True)
    result = result.sort_values(["date", "stock_code"]).reset_index(drop=True)
    
    # 正向(高峰度做多)
    output_pos = data_dir / "factor_kurtosis_v1.csv"
    result.to_csv(output_pos, index=False)
    print(f"正向保存: {output_pos} ({result.shape[0]} 行)")
    
    # 反向(低峰度做多)
    result_neg = result.copy()
    result_neg["factor_value"] = -result_neg["factor_value"]
    output_neg = data_dir / "factor_kurtosis_neg_v1.csv"
    result_neg.to_csv(output_neg, index=False)
    print(f"反向保存: {output_neg}")

if __name__ == "__main__":
    main()
