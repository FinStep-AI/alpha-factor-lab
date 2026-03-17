#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阳线频率因子 v1 (Bullish Day Frequency)
=========================================
10日内阳线（close > open）天数的频率。

构造：
  - bull_freq = count(close > open, 10d) / 10
  - 正向：高阳线频率 → 短期动量延续
  - 反向：低阳线频率 → 反转做多（类似neg_day_freq逻辑）
  - 成交额中性化 + MAD winsorize + z-score

灵感：
  - neg_day_freq_v1（极端负收益日频率）效果好(Sharpe=1.93)
  - 这里不看幅度极端性，而看方向频率
  - 更平滑但可能捕捉不同信息
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

WINDOW = 10
MIN_PERIODS = 8

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
KLINE_FILE = DATA_DIR / "csi1000_kline_raw.csv"
OUTPUT_FILE = DATA_DIR / "factor_bull_freq_v1.csv"


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


def main():
    print("=" * 60)
    print("Bullish Day Frequency Factor v1")
    print("=" * 60)

    # 1. 读取数据
    print("\n[1] 读取K线数据...")
    df = pd.read_csv(KLINE_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    print(f"  数据量: {len(df):,} 行, {df['stock_code'].nunique()} 只股票")

    # 2. 计算阳线标记
    print("\n[2] 计算阳线频率...")
    df["is_bull"] = (df["close"] > df["open"]).astype(float)
    
    # 10日滚动阳线频率
    df["bull_freq"] = df.groupby("stock_code")["is_bull"].transform(
        lambda x: x.rolling(WINDOW, min_periods=MIN_PERIODS).mean()
    )

    # 3. 中性化
    print("\n[3] 中性化处理...")
    df["log_amount_20d"] = np.log(
        df.groupby("stock_code")["amount"].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        ).clip(lower=1)
    )

    # MAD winsorize
    df["bull_win"] = df.groupby("date")["bull_freq"].transform(mad_winsorize)

    # OLS 中性化
    df["factor_neutral"] = neutralize_ols(df, "bull_win", "log_amount_20d")

    # z-score
    df["factor_value"] = df.groupby("date")["factor_neutral"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

    # 4. 统计
    valid = df.dropna(subset=["factor_value"])
    print(f"\n[4] 统计:")
    print(f"  有效样本: {len(valid):,} 行")
    print(f"  阳线频率均值: {df['bull_freq'].mean():.4f}")
    print(f"  阳线频率std: {df['bull_freq'].std():.4f}")

    # 5. 输出
    print(f"\n[5] 输出 → {OUTPUT_FILE}")
    out = valid[["stock_code", "date", "factor_value"]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"  写入 {len(out):,} 行")
    
    # 同时输出反向
    out_neg = out.copy()
    out_neg["factor_value"] = -out_neg["factor_value"]
    neg_file = DATA_DIR / "factor_bull_freq_neg_v1.csv"
    out_neg.to_csv(neg_file, index=False)
    print(f"  反向写入 → {neg_file}")

    print("\n✅ 完成！")


if __name__ == "__main__":
    main()
