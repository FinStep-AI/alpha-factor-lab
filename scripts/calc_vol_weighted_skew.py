#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成交量加权收益偏度因子 (Volume-Weighted Return Skewness) - 优化版

来源：Boyer, Mitton & Vorkink (2010, RFS) "Expected Idiosyncratic Skewness"
      
构造：20日滚动窗口中，成交量加权收益率的偏度
      使用向量化滚动矩计算（不用apply）
"""

import sys
import numpy as np
import pandas as pd

def rolling_skewness_vectorized(series, window=20, min_periods=15):
    """用滚动中心矩公式计算偏度，避免逐行apply"""
    # 先做rolling统计量
    r_mean = series.rolling(window, min_periods=min_periods).mean()
    r_std = series.rolling(window, min_periods=min_periods).std()
    r_count = series.rolling(window, min_periods=min_periods).count()
    
    # 三阶中心矩: E[(x-mu)^3]
    dev = series - r_mean
    m3 = (dev**3).rolling(window, min_periods=min_periods).mean()
    
    # 偏度 = m3 / std^3  (adjusted)
    # 使用 pandas 调整公式: n/((n-1)(n-2)) * sum((x-mu)/s)^3
    n = r_count
    skew = (n * (n - 1)).pow(0.5) / (n - 2) * m3 / (r_std ** 3)
    
    # 样本量不足时设为NaN
    skew[r_count < min_periods] = np.nan
    skew[r_std < 1e-10] = np.nan
    
    return skew

def main():
    print("Reading data...")
    kline = pd.read_csv("data/csi1000_kline_raw.csv", parse_dates=["date"])
    kline.sort_values(["stock_code", "date"], inplace=True)
    
    # 收益率
    kline["ret"] = kline["pct_change"] / 100.0
    kline["ret"] = kline["ret"].fillna(0)
    kline["volume"] = kline["volume"].fillna(0)
    
    print(f"Total rows: {len(kline)}, stocks: {kline['stock_code'].nunique()}")
    
    results = []
    stocks = kline["stock_code"].unique()
    
    for i, code in enumerate(stocks):
        if (i+1) % 200 == 0:
            print(f"  Processing {i+1}/{len(stocks)}...")
        
        grp = kline[kline["stock_code"] == code].sort_values("date").copy()
        
        if len(grp) < 20:
            continue
        
        # MA20(volume)
        vol_ma20 = grp["volume"].rolling(20, min_periods=15).mean()
        
        # 相对成交量权重
        vol_weight = grp["volume"] / vol_ma20.replace(0, np.nan)
        
        # 成交量加权收益
        ret_vw = grp["ret"] * vol_weight
        
        # 20日滚动偏度（向量化）
        vw_skew = rolling_skewness_vectorized(ret_vw, window=20, min_periods=15)
        
        mask = vw_skew.notna()
        if mask.sum() == 0:
            continue
        
        tmp = pd.DataFrame({
            "date": grp.loc[mask.index[mask], "date"].values,
            "stock_code": code,
            "factor_value": vw_skew[mask].values
        })
        results.append(tmp)
    
    df = pd.concat(results, ignore_index=True)
    
    print(f"\nRaw factor: {len(df)} observations, {df['stock_code'].nunique()} stocks")
    print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")
    print(f"Factor stats: mean={df['factor_value'].mean():.4f}, std={df['factor_value'].std():.4f}")
    
    # ---- 成交额中性化 ----
    amt = kline[["date", "stock_code", "amount"]].drop_duplicates()
    amt["log_amount"] = np.log(amt["amount"].replace(0, np.nan))
    df = df.merge(amt[["date", "stock_code", "log_amount"]], on=["date", "stock_code"], how="left")
    
    def neutralize(group):
        y = group["factor_value"].values
        x = group["log_amount"].values
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            group["factor_value"] = np.nan
            return group
        
        y_m, x_m = y[mask], x[mask]
        x_mat = np.column_stack([np.ones(len(x_m)), x_m])
        try:
            beta = np.linalg.lstsq(x_mat, y_m, rcond=None)[0]
            residuals = y_m - x_mat @ beta
            result = np.full(len(y), np.nan)
            result[mask] = residuals
            group["factor_value"] = result
        except:
            group["factor_value"] = np.nan
        return group
    
    print("Neutralizing by log_amount...")
    df = df.groupby("date", group_keys=False).apply(neutralize)
    
    # ---- MAD Winsorize ----
    def mad_winsorize(group, n_mad=5):
        v = group["factor_value"].values
        mask = np.isfinite(v)
        if mask.sum() < 10:
            return group
        v_valid = v[mask]
        med = np.median(v_valid)
        mad = np.median(np.abs(v_valid - med)) * 1.4826
        if mad < 1e-10:
            return group
        lower = med - n_mad * mad
        upper = med + n_mad * mad
        v_clipped = np.clip(v, lower, upper)
        group["factor_value"] = v_clipped
        return group
    
    df = df.groupby("date", group_keys=False).apply(mad_winsorize)
    
    # ---- Z-score ----
    def zscore(group):
        v = group["factor_value"].values
        mask = np.isfinite(v)
        if mask.sum() < 10:
            group["factor_value"] = 0
            return group
        mu = np.nanmean(v)
        sigma = np.nanstd(v)
        if sigma < 1e-10:
            group["factor_value"] = 0
        else:
            group["factor_value"] = (v - mu) / sigma
        return group
    
    df = df.groupby("date", group_keys=False).apply(zscore)
    
    # 输出
    out = df[["date", "stock_code", "factor_value"]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv("data/factor_vol_weighted_skew_v1.csv", index=False)
    
    print(f"\nFinal: {out['factor_value'].notna().sum()} valid values")
    print(f"Saved to data/factor_vol_weighted_skew_v1.csv")

if __name__ == "__main__":
    main()
