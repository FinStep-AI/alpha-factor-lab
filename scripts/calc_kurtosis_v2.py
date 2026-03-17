#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Return Kurtosis Factor (收益率峰度因子) - 向量化版本

学术背景:
  Dittmar (2002) "Nonlinear Pricing Kernels, Kurtosis Preference", JF
  Conrad, Dittmar & Ghysels (2013) "Ex Ante Skewness and Expected Stock Returns", JF

高峰度 = 收益分布厚尾 → 需要风险补偿 (正向)
或在A股: 高峰度 = 投机吸引 → 高估 (反向)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression

def rolling_kurtosis_vectorized(returns_pivot, window=20, min_periods=14):
    """
    向量化滚动峰度计算
    returns_pivot: index=date, columns=stock_code
    """
    arr = returns_pivot.values.astype(np.float64)
    n_dates, n_stocks = arr.shape
    result = np.full_like(arr, np.nan)
    
    for i in range(min_periods - 1, n_dates):
        start = max(0, i - window + 1)
        window_data = arr[start:i+1]  # (w, n_stocks)
        
        n = (~np.isnan(window_data)).sum(axis=0)  # count per stock
        valid = n >= min_periods
        
        # nanmean, nanstd
        mu = np.nanmean(window_data, axis=0)
        diff = window_data - mu[np.newaxis, :]
        m2 = np.nansum(diff**2, axis=0) / np.maximum(n - 1, 1)
        m4 = np.nansum(diff**4, axis=0) / np.maximum(n - 1, 1)
        
        # excess kurtosis (unbiased)
        std4 = m2 ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            # raw kurtosis = m4/std^4 - 3
            kurt = np.where(
                (std4 > 0) & valid,
                (n * (n + 1) * m4 / ((n - 1) * std4) - 3 * (n - 1)) * (n - 1) / np.maximum((n - 2) * (n - 3), 1),
                np.nan
            )
            # 简化版excess kurtosis: m4/m2^2 - 3
            kurt = np.where(
                (std4 > 0) & valid,
                m4 / std4 - 3,
                np.nan
            )
        
        result[i] = kurt
    
    return pd.DataFrame(result, index=returns_pivot.index, columns=returns_pivot.columns)

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
    
    # pivot
    ret_pivot = df.pivot(index="date", columns="stock_code", values="ret")
    
    print(f"收益率矩阵: {ret_pivot.shape}")
    
    # 向量化滚动峰度
    print("计算20日滚动峰度...")
    kurt_pivot = rolling_kurtosis_vectorized(ret_pivot, window=window, min_periods=14)
    
    print(f"峰度均值: {np.nanmean(kurt_pivot.values):.4f}, 中位数: {np.nanmedian(kurt_pivot.values):.4f}")
    
    # 成交额中性化
    print("成交额中性化...")
    amt_pivot = df.pivot(index="date", columns="stock_code", values="amount")
    amt_ma20 = amt_pivot.rolling(window, min_periods=14).mean()
    log_amt = np.log(amt_ma20.clip(lower=1))
    
    # 截面OLS中性化 + MAD + z-score
    factor_records = []
    for dt in kurt_pivot.index:
        kurt_row = kurt_pivot.loc[dt]
        if dt not in log_amt.index:
            continue
        amt_row = log_amt.loc[dt]
        
        valid = kurt_row.notna() & amt_row.notna()
        if valid.sum() < 50:
            continue
        
        x = amt_row[valid].values.reshape(-1, 1)
        y = kurt_row[valid].values
        codes = kurt_row[valid].index.tolist()
        
        # MAD winsorize
        med = np.median(y)
        mad = np.median(np.abs(y - med))
        if mad > 0:
            y = np.clip(y, med - 5 * 1.4826 * mad, med + 5 * 1.4826 * mad)
        
        lr = LinearRegression()
        lr.fit(x, y)
        resid = y - lr.predict(x)
        
        std = resid.std()
        if std > 0:
            resid = (resid - resid.mean()) / std
        
        for code, val in zip(codes, resid):
            factor_records.append({"date": dt, "stock_code": code, "factor_value": val})
    
    result = pd.DataFrame(factor_records)
    result = result.sort_values(["date", "stock_code"]).reset_index(drop=True)
    
    print(f"最终因子: {result.shape[0]} 行, {result['stock_code'].nunique()} 只股票")
    
    output_path = data_dir / "factor_kurtosis_v1.csv"
    result.to_csv(output_path, index=False)
    print(f"保存: {output_path}")

if __name__ == "__main__":
    main()
