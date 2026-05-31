#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
roe_trend_dev_v1 — ROE 长短期趋势背离因子
论文来源：北大经济学院 (2025) "A股市场公司特征动态变化的定价效应——基于长短期差异的实证研究"
        杨琳、戈舒怡、李少然 — https://econ.pku.edu.cn/kxyj/gzlw/520dbb93fdb04cb8aceeeebf25dde723.htm

构造逻辑
---------
对每只股票在每期报告日，用过去 N 个季度的 ROE 做最小二乘回归（按报告日期排序），
取回归斜率 β 作为"ROE 趋势强度"：
  β ≈ d(ROE) / d(quarter)
同时用拟合优度 R² 作为趋势可信度权重。

核心因子 = β_short × R²_short − β_long × R²_long，方向：高因子值 → 高预期收益。
解释：短期趋势改善（大于长期趋势的股票）

中性化：对数成交额 OLS 残差，MAD 缩尾，z-score。

输出：data/factor_roe_trend_dev_v1.csv  (date, stock_code, factor)
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# ── path ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
FUND_CSV   = BASE / "data" / "csi1000_fundamental_cache.csv"
KLINE_CSV  = BASE / "data" / "csi1000_kline_raw.csv"
OUT_CSV    = BASE / "data" / "factor_roe_trend_dev_v1.csv"
LAG_DAYS   = 60  # 报告日信息滞后，避免前视偏差


# ── helpers ────────────────────────────────────────────────────────────────

def _ols_slope_r2(y: np.ndarray) -> tuple[float, float]:
    """OLS(y ~ trend) 对等距索引 0,1,2,... 做回归，返回 (slope, r2)。"""
    n = len(y)
    if n < 3:
        return np.nan, np.nan
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    sx  = x - xm
    sy  = y - ym
    ss_xx = (sx ** 2).sum()
    ss_yy = (sy ** 2).sum()
    ss_xy = (sx * sy).sum()
    if ss_xx == 0 or ss_yy == 0:
        return np.nan, np.nan
    beta  = ss_xy / ss_xx
    ss_res = ss_yy - beta ** 2 * ss_xx
    r2    = 1.0 - ss_res / ss_yy if ss_yy else np.nan
    return beta, r2


def _mad_zscore(s: pd.Series, k: float = 5.5) -> pd.Series:
    """MAD 缩尾 + z-score。"""
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if mad == 0 or np.isnan(mad):
        return pd.Series(0.0, index=s.index)
    upper = med + k * mad
    lower = med - k * mad
    s2 = s.clip(lower, upper)
    return (s2 - s2.mean()) / s2.std(ddof=0)


# ── step 1: 把季报 ROE 滚到每个截面日期 ──────────────────────────────────

print("[1/4] 读取基本面数据 …")

fund = pd.read_csv(FUND_CSV, dtype={"stock_code": str})
fund["stock_code"] = fund["stock_code"].str.zfill(6)
fund["report_date"] = pd.to_datetime(fund["report_date"])

# 可用报告期
fund = fund.sort_values(["stock_code", "report_date"])

# kline 交易日
kline = pd.read_csv(KLINE_CSV, dtype={"stock_code": str})
kline["stock_code"] = kline["stock_code"].str.zfill(6)
kline["date"] = pd.to_datetime(kline["date"])
trade_dates = sorted(kline["date"].unique())

# 量额聚合
amt = (kline.groupby(["date", "stock_code"])["amount"]
       .sum().reset_index())
amt["log_amount_20d"] = (
    amt.groupby("stock_code")["amount"]
    .transform(lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1))
)

print(f"  fund: {fund['stock_code'].nunique()} stocks, "
      f"{fund['report_date'].nunique()} report dates "
      f"[{fund['report_date'].min().date()} ~ {fund['report_date'].max().date()}]")
print(f"  trade dates: {len(trade_dates)}, "
      f"[{trade_dates[0].date()} ~ {trade_dates[-1].date()}]")

# ── step 2: 每只股票 / 每个报告期计算 β × R² ───────────────────────────

print("[2/4] 计算每报告期 ROE 趋势 β×R² …")

SHORT_Q = 3   # 短期：最近 3 个季度
LONG_Q  = 8   # 长期：过去 8 个季度（短窗口的"更前"部分用 8−3+1 … 8）

records = []
for code, grp in fund.groupby("stock_code", sort=False):
    grp = grp.sort_values("report_date").reset_index(drop=True)
    roe_arr = grp["roe"].values
    rd_arr  = grp["report_date"].values
    n = len(grp)

    # 对于每一行 i（即每个报告期），以 i-7 .. i-4 为"长期段"，i-2..i 为"短期段"
    for i in range(LONG_Q - 1, n):
        # — 长期段：i-7 .. i-4（4 个季度，即 1 年）
        lo_long = i - 7
        hi_long = i - 3   # 不包含 i-3，即 i-7,i-6,i-5,i-4
        # — 短期段：i-2 .. i（3 个季度，含最新）
        lo_short = i - 2
        hi_short = i + 1

        yl = roe_arr[lo_long:hi_long]
        ys = roe_arr[lo_short:hi_short]

        # 这里维护数据质量：两个窗口都要 >= 3 && <= 8 个点；
        # 跳过缺失
        if np.isnan(yl).any() or np.isnan(ys).any():
            continue
        # 早期部分缺失跳过
        if len(yl) < 3 or len(ys) < 3:
            continue

        b_long, r2_long = _ols_slope_r2(yl)
        b_short, r2_short = _ols_slope_r2(ys)
        if np.isnan(b_long) or np.isnan(b_short):
            continue

        signal = b_short * r2_short - b_long * r2_long
        # 对数 1+abs 再带符号，弱极端放大
        if signal >= 0:
            val = np.log1p(signal)
        else:
            val = -np.log1p(-signal)

        records.append({
            "stock_code":  code,
            "report_date": rd_arr[i],
            "factor_raw":  float(val),
        })

trend_raw = pd.DataFrame(records)
print(f"  短期×截面 初步 rows: {len(trend_raw)}")

# ── step 3: 报告日 + 60 天滞后 → 日频扩展，成交额中性化 ─────────────────

print("[3/4] 展平到日频并做截面中性化 …")

trend_raw["report_date"] = pd.to_datetime(trend_raw["report_date"])
trend_raw["avail_date"] = trend_raw["report_date"] + pd.Timedelta(days=LAG_DAYS)

# 每只股票最多保留最新一条
trend_raw = (trend_raw.sort_values("avail_date")
             .groupby(["stock_code", "avail_date"], as_index=False)
             .last())

# 与量额 merge
merged = amt[["date", "stock_code", "log_amount_20d"]].merge(
    trend_raw[["stock_code", "avail_date", "factor_raw"]],
    left_on=["stock_code", "date"],
    right_on=["stock_code", "avail_date"],
    how="inner",
)
merged = merged.dropna(subset=["factor_raw", "log_amount_20d"]).copy()

# 逐日截面：OLS 中性化 → MAD z-score
rows = []
for dt, day in merged.groupby("date", sort=False):
    y = day["factor_raw"].values
    x = day["log_amount_20d"].values
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 30:
        continue
    ym, xm = y[mask], x[mask]
    x_dm = xm - xm.mean()
    beta = (x_dm * ym).sum() / (x_dm ** 2).sum() if (x_dm ** 2).sum() != 0 else 0
    alpha = ym.mean() - beta * xm.mean()
    resid = ym - (alpha + beta * xm)

    s = pd.Series(resid, index=day.index[mask])
    z = _mad_zscore(s, k=5.5)
    for idx, val in zip(day.index[mask], z):
        rows.append({"date": dt, "stock_code": day.loc[idx, "stock_code"],
                     "factor": float(val)})

factor = pd.DataFrame(rows).sort_values(["date", "stock_code"]).reset_index(drop=True)
print(f"  日频截面 rows: {len(factor)}, dates: {factor['date'].nunique()}")
print(f"  factor stats: mean={factor['factor'].mean():.4f}  "
      f"std={factor['factor'].std():.4f}  "
      f"[{factor['date'].min().date()} ~ {factor['date'].max().date()}]")

factor.to_csv(OUT_CSV, index=False)
print(f"[4/4] 写 → {OUT_CSV}  ✓")
