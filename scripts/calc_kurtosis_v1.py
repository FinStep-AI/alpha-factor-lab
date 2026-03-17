#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
截面峰度因子 v1 (Realized Kurtosis)
======================================
基于 Amaya, Christoffersen, Jacobs & Vasquez (2015) JFE 的思路。
原文研究了 realized skewness（我们已测试失败），这里测试 realized kurtosis。

构造：
  - 计算每只股票过去10日收益率的峰度 (excess kurtosis)
  - 正向使用：高峰度 = 更多极端收益 = 可能的风险溢价/注意力效应
  - 也测试反向：低峰度 = 收益分布更正常 = 更稳定
  - 成交额中性化 + MAD winsorize + z-score

数据：data/csi1000_kline_raw.csv (OHLCV)
输出：data/factor_kurtosis_v1.csv
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ============================================================
# 参数
# ============================================================
WINDOW = 10       # 滚动窗口(短期，类似neg_day_freq)
MIN_PERIODS = 8   # 最少有效天数

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
KLINE_FILE = DATA_DIR / "csi1000_kline_raw.csv"
OUTPUT_FILE = DATA_DIR / "factor_kurtosis_v1.csv"


def mad_winsorize(s, n_mad=5):
    median = s.median()
    mad = (s - median).abs().median() * 1.4826
    lower = median - n_mad * mad
    upper = median + n_mad * mad
    return s.clip(lower, upper)


def neutralize_ols(df, factor_col, neutral_col):
    from numpy.linalg import lstsq
    result = df[[factor_col]].copy()
    result["residual"] = np.nan
    for date, group in df.groupby("date"):
        y = group[factor_col].values
        x = group[neutral_col].values
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            continue
        X = np.column_stack([np.ones(mask.sum()), x[mask]])
        beta, _, _, _ = lstsq(X, y[mask], rcond=None)
        resid = y.copy()
        resid[mask] = y[mask] - X @ beta
        resid[~mask] = np.nan
        result.loc[group.index, "residual"] = resid
    return result["residual"]


def fast_rolling_kurtosis(arr, window, min_periods):
    """使用向量化的滚动峰度计算"""
    n = len(arr)
    result = np.full(n, np.nan)
    
    for i in range(min_periods - 1, n):
        start = max(0, i - window + 1)
        x = arr[start:i+1]
        valid = x[np.isfinite(x)]
        if len(valid) < min_periods:
            continue
        m = valid.mean()
        d = valid - m
        n_v = len(valid)
        if n_v < 4:
            continue
        s2 = (d**2).sum() / (n_v - 1)
        if s2 < 1e-20:
            result[i] = 0.0
            continue
        s4 = (d**4).sum()
        # Excess kurtosis (Fisher, unbiased)
        kurt = ((n_v * (n_v + 1)) / ((n_v - 1) * (n_v - 2) * (n_v - 3))) * (s4 / (s2**2)) - \
               (3 * (n_v - 1)**2) / ((n_v - 2) * (n_v - 3))
        result[i] = kurt
    
    return result


def main():
    print("=" * 60)
    print("Realized Kurtosis Factor v1")
    print("=" * 60)

    # 1. 读取数据
    print("\n[1] 读取K线数据...")
    df = pd.read_csv(KLINE_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    print(f"  数据量: {len(df):,} 行, {df['stock_code'].nunique()} 只股票")

    # 2. 计算日收益率
    print("\n[2] 计算日收益率...")
    df["ret"] = df.groupby("stock_code")["close"].pct_change()

    # 3. 滚动峰度
    print(f"\n[3] 计算滚动峰度 (窗口={WINDOW})...")
    df["kurt_raw"] = df.groupby("stock_code")["ret"].transform(
        lambda x: pd.Series(fast_rolling_kurtosis(x.values, WINDOW, MIN_PERIODS), index=x.index)
    )

    # 4. 中性化
    print("\n[4] 中性化处理...")
    df["log_amount_20d"] = np.log(
        df.groupby("stock_code")["amount"].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        ).clip(lower=1)
    )

    # MAD winsorize
    df["kurt_win"] = df.groupby("date")["kurt_raw"].transform(mad_winsorize)

    # OLS 中性化
    df["factor_neutral"] = neutralize_ols(df, "kurt_win", "log_amount_20d")

    # z-score
    df["factor_value"] = df.groupby("date")["factor_neutral"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

    # 5. 统计
    valid = df.dropna(subset=["factor_value"])
    print(f"\n[5] 统计:")
    print(f"  有效样本: {len(valid):,} 行")
    print(f"  日均覆盖: {valid.groupby('date')['factor_value'].count().mean():.0f} 只")
    print(f"  峰度原始均值: {df['kurt_raw'].mean():.4f}")
    print(f"  峰度原始中位数: {df['kurt_raw'].median():.4f}")

    # 6. 输出
    print(f"\n[6] 输出 → {OUTPUT_FILE}")
    out = valid[["stock_code", "date", "factor_value"]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"  写入 {len(out):,} 行")
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()
