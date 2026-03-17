#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缩量反弹频率因子 v1 (Low-Volume Rally Frequency)
====================================================
灵感：Llorente, Michaely, Saar & Wang (2002) JF
"Dynamic Volume-Return Relation of Individual Stocks"

核心逻辑：
  - 识别"缩量反弹日"：ret > 0 且 volume < MA20(volume)
  - 缩量反弹 = 卖压枯竭信号（价格已转正但成交量低=没有新增抛售）
  - 近10日缩量反弹日频率越高 → 卖压枯竭越充分 → 后续上涨概率高

与已有因子的区别：
  - pv_corr_v1: 连续变量的相关性，本因子是离散事件频率
  - neg_day_freq: 看下跌日频率(反转)，本因子看缩量+上涨(卖压枯竭)
  - turnover_decay: 纯换手率衰减，本因子结合了方向信息

构造：
  - low_vol_rally = (ret > 0) & (volume < MA20_volume)
  - factor = rolling_mean(low_vol_rally, 10d)
  - 成交额中性化 + MAD winsorize + z-score
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

WINDOW = 10
VOL_MA = 20
MIN_PERIODS = 8

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
KLINE_FILE = DATA_DIR / "csi1000_kline_raw.csv"
OUTPUT_FILE = DATA_DIR / "factor_low_vol_rally_v1.csv"


def mad_winsorize(s, n_mad=5):
    median = s.median()
    mad = (s - median).abs().median() * 1.4826
    return s.clip(median - n_mad * mad, median + n_mad * mad)


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
    print("Low-Volume Rally Frequency Factor v1")
    print("=" * 60)

    # 1. 读取数据
    print("\n[1] 读取K线数据...")
    df = pd.read_csv(KLINE_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    print(f"  数据量: {len(df):,} 行, {df['stock_code'].nunique()} 只股票")

    # 2. 计算日收益率和MA20成交量
    print("\n[2] 计算收益率和成交量基准...")
    df["ret"] = df.groupby("stock_code")["close"].pct_change()
    df["vol_ma20"] = df.groupby("stock_code")["volume"].transform(
        lambda x: x.rolling(VOL_MA, min_periods=10).mean()
    )

    # 3. 识别缩量反弹日
    print("\n[3] 识别缩量反弹日...")
    df["is_low_vol_rally"] = (
        (df["ret"] > 0) & 
        (df["volume"] < df["vol_ma20"])
    ).astype(float)
    
    # NaN处理：如果ret或vol_ma20是NaN，标记为NaN而非0
    df.loc[df["ret"].isna() | df["vol_ma20"].isna(), "is_low_vol_rally"] = np.nan
    
    print(f"  缩量反弹日占比: {df['is_low_vol_rally'].mean():.1%}")

    # 4. 10日滚动频率
    print(f"\n[4] 计算 {WINDOW}日 缩量反弹频率...")
    df["lvr_freq"] = df.groupby("stock_code")["is_low_vol_rally"].transform(
        lambda x: x.rolling(WINDOW, min_periods=MIN_PERIODS).mean()
    )

    # 5. 中性化
    print("\n[5] 中性化处理...")
    df["log_amount_20d"] = np.log(
        df.groupby("stock_code")["amount"].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        ).clip(lower=1)
    )
    
    # MAD winsorize
    df["lvr_win"] = df.groupby("date")["lvr_freq"].transform(mad_winsorize)
    
    # OLS 中性化
    df["factor_neutral"] = neutralize_ols(df, "lvr_win", "log_amount_20d")
    
    # z-score
    df["factor_value"] = df.groupby("date")["factor_neutral"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

    # 6. 统计
    valid = df.dropna(subset=["factor_value"])
    print(f"\n[6] 统计:")
    print(f"  有效样本: {len(valid):,} 行")
    print(f"  日均覆盖: {valid.groupby('date')['factor_value'].count().mean():.0f} 只")
    print(f"  缩量反弹频率均值: {df['lvr_freq'].mean():.4f}")
    print(f"  缩量反弹频率中位数: {df['lvr_freq'].median():.4f}")
    print(f"  缩量反弹频率std: {df['lvr_freq'].std():.4f}")

    # 7. 输出
    print(f"\n[7] 输出 → {OUTPUT_FILE}")
    out = valid[["stock_code", "date", "factor_value"]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"  写入 {len(out):,} 行")

    # 反向也输出
    out_neg = out.copy()
    out_neg["factor_value"] = -out_neg["factor_value"]
    neg_file = DATA_DIR / "factor_low_vol_rally_neg_v1.csv"
    out_neg.to_csv(neg_file, index=False)
    print(f"  反向写入 → {neg_file}")

    print("\n✅ 完成！")


if __name__ == "__main__":
    main()
