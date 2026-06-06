#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
turnover_recency_weight_v1 — 近因换手集中度因子
================================================
构造：
  以 20 日为滚动窗口，将窗口内交易日分为 5 个子段（每段 4 个交易日，从旧到新）：
  sub_i = sum(turnover) over days [(i-1)*4+1 .. i*4]，i=1..5
  total = sum(turnover, 20d)
  因子原始值 = sub_5 / total   ← 最近 4 天换手占总换手的比例

解释：
  高 recency_weight → 最近 4 天换手明显集中 → 近期有异乎寻常的交易活动（可能是
  知情交易者入场/散户注意力集中/信息冲击后博弈剧烈），经 OLS 成交额中性化后，
  反映的是换手时序分布的非基本面 alpha。
  低 recency_weight → 换手均匀分布在旧段 → 近期平淡 / 已经冷下来。

中性化：对 log(amount_20d) 做 OLS 回归取残差，再做 MAD 缩尾 + z-score。
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── 路径 ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
KLINE  = BASE / "data" / "csi1000_kline_raw.csv"
OUT    = BASE / "data" / "factor_turnover_recency_weight_v1.csv"


def mad_winsorize(s: pd.Series, n: float = 5.2) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if mad < 1e-12:
        return s * 0
    lo, hi = med - n * mad, med + n * mad
    return s.clip(lo, hi)


def compute_factor():
    print("[1/3] loading kline …")
    df = pd.read_csv(KLINE, parse_dates=["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    # 只保留有换手率的行
    df = df.dropna(subset=["turnover"]).copy()
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)

    print(f"      rows={len(df):,}  stocks={df['stock_code'].nunique()}  "
          f"dates={df['date'].min().date()} ~ {df['date'].max().date()}")

    # ── 20 日成交额均值（用于中性化） ────────────────────────────────────
    print("[2/3] computing turnover recency weight …")
    df["amount_20d"] = (
        df.groupby("stock_code")["amount"]
        .transform(lambda x: x.rolling(20, min_periods=15).mean())
    )
    df["log_amount_20d"] = np.log(df["amount_20d"].clip(lower=1))

    # 最近 4 天换手之和 / 20 天总换手
    # 用 rolling(20).sum() 当 total
    df["turn20_sum"] = (
        df.groupby("stock_code")["turnover"]
        .transform(lambda x: x.rolling(20, min_periods=15).sum())
    )
    # 最近 4 天换手之和
    df["turn4_sum"] = (
        df.groupby("stock_code")["turnover"]
        .transform(lambda x: x.rolling(4, min_periods=3).sum())
    )

    df["factor_raw"] = df["turn4_sum"] / df["turn20_sum"]

    # 去掉 inf / NaN
    df["factor_raw"] = df["factor_raw"].replace([np.inf, -np.inf], np.nan)

    # ── 截面 OLS 中性化 ──────────────────────────────────────────────────
    def neutralize_cs(group: pd.DataFrame) -> pd.Series:
        y = group["factor_raw"]
        x = group["log_amount_20d"]
        mask = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            return pd.Series(np.nan, index=group.index)
        yy, xx = y[mask], x[mask]
        x_dm = xx - xx.mean()
        beta = (x_dm * yy).sum() / (x_dm ** 2).sum() if (x_dm ** 2).sum() > 0 else 0
        alpha = yy.mean() - beta * xx.mean()
        resid = yy - (alpha + beta * xx)
        # MAD winsorize + z-score
        r = mad_winsorize(resid, n=5.2)
        std = r.std()
        if std < 1e-12:
            return pd.Series(0.0, index=group.index)
        z = (r - r.mean()) / std
        out = pd.Series(np.nan, index=group.index)
        out[mask.index[mask.values]] = z
        return out

    print("      cross-sectional neutralization …")
    df["factor"] = df.groupby("date", group_keys=False).apply(neutralize_cs)

    # ── 输出 ─────────────────────────────────────────────────────────────
    print("[3/3] saving …")
    out = df[["date", "stock_code", "factor"]].dropna(subset=["factor"])
    out = out.rename(columns={"factor": "factor_value"})
    out.to_csv(OUT, index=False)
    print(f"      saved {len(out):,} rows → {OUT}")

    # 基本统计
    print("\n=== factor stats ===")
    print(out["factor_value"].describe())
    print(f"\ndates  : {out['date'].min().date()} ~ {out['date'].max().date()}")
    print(f"stocks : {out['stock_code'].nunique()}")
    print(f"cross-sections: {out['date'].nunique()}")


if __name__ == "__main__":
    compute_factor()
