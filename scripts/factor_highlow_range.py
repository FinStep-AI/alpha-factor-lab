"""
高低位波动率占比因子（国盛证券「量价淘金」系列第四 / 日频版）
===========================================================
核心逻辑（国盛证券 2023-12-19）：
  原始：高/低位"放量" = price_percentile × volatility_share
  我们的日频可执行版本：price_percentile × range_upper_share
    - price_percentile := 当日close在 20日滚动高低 区间内的位置 (0~1)，高值=近期高位
    - range_upper_share := 当日振幅 (high-low) 中位于 close 以上的那部分占总
                          振幅的比例；即 (high - close) / (high - low)
                          高值=当日波动主要来自上方压力（高位放量感的波动形态）

方向：正向（高 composite 因子 → 预期未来正收益）
  国盛原报告用的是波动率版本 IC=-0.066（高值→低收益，做反向）；
  这里我们取另一侧——"高位波动主要来自上方"这一侧的信号做正向。

输出：data/factor_highlow_range.csv
  列：date, stock_code, price_pct, range_up_share, factor
"""

import numpy as np
import pandas as pd
from pathlib import Path


def calc_highlow_factor(kline_path: str,
                        lookback: int = 20,
                        out_csv: str = "data/factor_highlow_range.csv") -> str:
    df = pd.read_csv(kline_path, parse_dates=["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    # --- rolling high/low 20日 ---
    roll = df.groupby("stock_code")["high"].transform(
        lambda s: s.rolling(lookback, min_periods=lookback).max())
    roll_low = df.groupby("stock_code")["low"].transform(
        lambda s: s.rolling(lookback, min_periods=lookback).min())

    df["roll_high"] = roll
    df["roll_low"] = roll_low

    # --- 价格位置百分位（0~1）---
    rng = df["roll_high"] - df["roll_low"]
    df["price_pct"] = np.where(
        rng > 0,
        (df["close"] - df["roll_low"]) / rng,
        np.nan
    )

    # --- 上半振幅占比 ---
    df["rng"] = df["high"] - df["low"]
    # (high - close) / (high - low): 波动中「向上」的空间占多少
    df["range_up_share"] = np.where(
        df["rng"] > 0,
        (df["high"] - df["close"]) / df["rng"],
        np.nan
    )

    # --- 日频合成因子（两个分量 z-score 后合成，不对，此处用直接乘积保留截面可比性）---
    #   直接乘积后截面 rank，再做 min-max 归一
    df["raw"] = df["price_pct"] * df["range_up_share"]

    def cross_rank(g):
        r = g["raw"].rank(pct=True)
        return r

    df["factor"] = df.groupby("date")["raw"].transform(
        lambda s: s.rank(pct=True, method="average")
        if s.notna().sum() > 10 else np.nan
    )

    # 只保留有效截面
    out = df[["date", "stock_code", "factor"]].dropna(subset=["factor"]).copy()

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    n_dates = out["date"].nunique()
    n_stocks_per = out.groupby("date")["stock_code"].count().median()
    print(f"✅ factor_highlow_range saved → {out_csv}")
    print(f"   截面数: {n_dates}  |  每截面中位股票数: {n_stocks_per:.0f}")
    print(f"   factor 均值={out['factor'].mean():.4f}  标准差={out['factor'].std():.4f}")
    return out_csv


if __name__ == "__main__":
    import sys
    kline = sys.argv[1] if len(sys.argv) > 1 else "data/csi1000_kline_raw.csv"
    calc_highlow_factor(kline)
