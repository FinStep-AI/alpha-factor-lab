#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
换手率加速度离散度因子 v1 — turnover_accel_disp_v1
计算每只股票 换手率的变异系数(CV = std/mean, 窗口20d)，成交额OLS中性化后输出。

逻辑：换手率的横截面分布随时间的波动（加速度离散度）→ 换手率变动最剧烈的股票
      往往是消息驱动 / 知情交易推动，后续5日持有收益更稳健。
      形式上是换手率的二阶动量信号（turnover"加速度"的截面波动）的中性化代理。
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

# ── 函数 ────────────────────────────────────────────────────────────────────

def mad_winsorize(s: pd.Series, k: float = 5.2) -> pd.Series:
    med = s.median()
    dev = (s - med).abs().median() * 1.4826 + 1e-12
    lo, hi = med - k * dev, med + k * dev
    return s.clip(lo, hi)

# ── 主流程 ──────────────────────────────────────────────────────────────────

REPO   = Path(__file__).resolve().parents[1]
KLINE  = REPO / "data" / "csi1000_kline_raw.csv"
OUT    = REPO / "data" / "factor_turnover_accel_disp_v1.csv"

WINDOW = 20          # 换手率滚动均值+标准差窗口
NEUT_WIN = 20        # 中性化用 20 日成交额均值

print(f"[1/4] 读取 {KLINE.name} …")
df = pd.read_csv(KLINE, parse_dates=["date"])

# 保留有换手率 和 成交额的数据
df = df.dropna(subset=["turnover", "amount"])
df = df[df["turnover"] > 0].copy()
df = df[df["amount"] > 0].copy()
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

print(f"[2/4] 计算 {WINDOW}日换手率变异系数 (std/mean) …")

def turnover_cv(g: pd.DataFrame) -> pd.Series:
    roll = g["turnover"].rolling(WINDOW, min_periods=WINDOW)
    mu   = roll.mean()
    sig  = roll.std(ddof=1)
    return sig / (mu + 1e-12)           # CV

df["raw_factor"] = df.groupby("stock_code", group_keys=False).apply(
    turnover_cv
)

print("[3/4] 成交额OLS中性化 + Winsorize + z-score …")

# 20日成交额均值作为中性化变量
df["log_amount_neut"] = np.log(
    df.groupby("stock_code")["amount"]
      .transform(lambda x: x.rolling(NEUT_WIN, min_periods=NEUT_WIN).mean())
      + 1.0
)

def neutralize_cross(g: pd.DataFrame) -> pd.Series:
    """每截面OLS残差"""
    y = g["raw_factor"].values
    x = g["log_amount_neut"].values
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 30:
        return pd.Series(np.nan, index=g.index)
    xm, ym = x[mask], y[mask]
    xm = (xm - xm.mean()) / (xm.std() + 1e-12)
    ym = (ym - ym.mean()) / (ym.std() + 1e-12)
    b  = np.cov(xm, ym)[0, 1] / (np.var(xm) + 1e-12)
    resid = ym - b * xm
    out = np.full(len(g), np.nan)
    out[mask] = resid
    return pd.Series(out, index=g.index)

df["neu_factor"] = df.groupby("date", group_keys=False).apply(
    neutralize_cross
)

# z-score
def zscore_cross(g: pd.DataFrame) -> pd.Series:
    v = g["neu_factor"]
    m, s = v.mean(), v.std()
    return (v - m) / (s + 1e-12) if s > 1e-12 else v * 0

df["factor_value"] = df.groupby("date", group_keys=False).apply(
    lambda g: mad_winsorize(zscore_cross(g))
).reset_index(level=0, drop=True)

print("[4/4] 写出因子 CSV …")
out = df[["date", "stock_code", "factor_value"]].dropna(subset=["factor_value"])
out.columns = ["date", "stock_code", "factor_value"]
out.to_csv(OUT, index=False)

n_days = out["date"].nunique()
n_stocks = out["stock_code"].nunique()
print(f"✅  写入 {OUT.name}: {len(out)} 行, {n_days} 个截面, {n_stocks} 只票")
print(f"   date 范围: {out['date'].min()} ~ {out['date'].max()}")
