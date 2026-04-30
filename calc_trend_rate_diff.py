#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trend_rate_diff_v1 — 价格趋势速率差异因子 (固定版)
概念：短期(5d)均线OLS斜率 - 中期(60d)均线OLS斜率，
     除以60日均线做截面标准化。
经济逻辑：短期加速上行股票通常有更强的价格持续性。

计时路径：
  1. load kline → calc MA5, MA20, MA60
  2. OLS slope on MA5 (5d window) and MA60 (60d window)
  3. neutralize(turnover) → z-score per cross-section
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR  = Path("data")

# ── fast OLS slope (no overhead) ─────────────────────────────────────────
def _fast_slope(y: np.ndarray) -> float:
    """OLS slope for equally-spaced x=0..n-1. Returns NaN if n < 2."""
    n = len(y)
    if n < 2:
        return float("nan")
    # remove NaN
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return float("nan")
    x  = np.arange(n, dtype=float)[valid]
    xm = x.mean()
    yi = y[valid]
    ym = yi.mean()
    ss_xx = ((x - xm) ** 2).sum()
    if ss_xx <= 0:
        return float("nan")
    return float(((x - xm) * (yi - ym)) / ss_xx)


# ── rolling OLS via expanding window patch (vectorized) ───────────────────
def roll_slope(series_arr: np.ndarray, window: int) -> np.ndarray:
    """Compute OLS slope for each rolling window of length 'window'.
    Returns array of same length; NaN before first full window."""
    n = len(series_arr)
    out   = np.full(n, np.nan)
    x_raw = np.arange(window, dtype=float)
    x_bar = x_raw.mean()
    ss_xx = ((x_raw - x_bar) ** 2).sum()

    # pre-compute x_centered once
    x_c  = x_raw - x_bar
    buf  = np.empty(window)

    for i in range(window - 1, n):
        buf[:] = series_arr[i - window + 1: i + 1]
        if not np.all(np.isfinite(buf)):
            continue
        y_bar = buf.mean()
        out[i] = float((x_c * (buf - y_bar)).sum() / ss_xx)
    return out


# ── neutralize: remove cross-sectional linear dependence on 'neutralizer' ──
def neutralize_cs(factor: np.ndarray, neutralizer: np.ndarray, min_obs: int = 10) -> np.ndarray:
    """Demean + regress out neutralizer; return residuals."""
    mask = np.isfinite(factor) & np.isfinite(neutralizer)
    out  = np.full_like(factor, np.nan)
    if mask.sum() < min_obs:
        return out
    f_m  = factor[mask]
    n_m  = neutralizer[mask]
    A    = np.column_stack([np.ones(len(n_m)), n_m])
    try:
        coef, _, _, _ = np.linalg.lstsq(A, f_m, rcond=None)
        out[mask] = f_m - A @ coef
    except Exception:
        pass
    return out


# ── main ──────────────────────────────────────────────────────────────────
def main():
    print("Loading kline data …")
    df = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv",
                     parse_dates=["date"],
                     dtype={"stock_code": str,  # keep as str for safety
                            "close": float,
                            "turnover": float})
    df["stock_code"] = df["stock_code"].str.zfill(6)  # "000012" style

    # sort and group
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    codes       = df["stock_code"].unique()
    date_list   = sorted(df["date"].unique().tolist())
    n_stocks    = len(codes)
    n_days      = len(date_list)
    print(f"  Rows: {len(df):,} | Stocks: {n_stocks} | Days: {n_days}")

    # build 3-d arrays
    code_idx = {c: i for i, c in enumerate(codes)}
    day_idx  = {d: j for j, d in enumerate(date_list)}

    close_arr  = np.full((n_stocks, n_days), np.nan)
    to_arr     = np.full((n_stocks, n_days), np.nan)

    for row in df.itertuples():
        ci = code_idx[row.stock_code]
        di = day_idx[row.date]
        close_arr[ci, di]  = float(row.close)
        to_arr[ci, di]     = float(row.turnover)
    print(f"  close_arr shape: {close_arr.shape},  coverage: {np.isfinite(close_arr).mean():.1%}")

    # ── compute MA5, MA20, MA60 ─────────────────────────────────────────────
    print("Computing rolling MAs …")
    def roll_mean(arr: np.ndarray, w: int) -> np.ndarray:
        """Row-wise rolling mean (symmetric), padding with NaN."""
        n_s, n_d = arr.shape
        out = np.full_like(arr, np.nan)
        for i in range(n_s):
            v = arr[i]
            # pad NaN edges
            padded = np.concatenate([np.full(w - 1, np.nan), v])
            for j in range(n_d):
                seg = padded[j: j + w]
                if np.isfinite(seg).sum() == w:
                    out[i, j] = np.nanmean(seg)
        return out

    ma5_arr  = roll_mean(close_arr, 5)
    ma20_arr = roll_mean(close_arr, 20)
    ma60_arr = roll_mean(close_arr, 60)

    # ── OLS slope ───────────────────────────────────────────────────────────
    print("Computing OLS slope_MA5 (window=5) …")
    slope_ma5  = np.full((n_stocks, n_days), np.nan)
    for i in range(n_stocks):
        slope_ma5[i] = roll_slope(ma5_arr[i], 5)

    print("Computing OLS slope_MA60 (window=60) …")
    slope_ma60 = np.full((n_stocks, n_days), np.nan)
    for i in range(n_stocks):
        slope_ma60[i] = roll_slope(ma60_arr[i], 60)

    # ── raw factor ──────────────────────────────────────────────────────────
    denom = ma60_arr.copy()
    denom[denom == 0] = np.nan
    factor_raw = (slope_ma5 - slope_ma60) / denom

    # ── cross-section neutralize(turnover) per day ──────────────────────────
    print("Neutralizing by turnover per day …")
    factor_neutral = np.full((n_stocks, n_days), np.nan)
    for j in range(n_days):
        f = factor_raw[:, j]
        t = to_arr[:, j]
        factor_neutral[:, j] = neutralize_cs(f, t)

    # ── per-day z-score ─────────────────────────────────────────────────────
    print("Z-scoring per day …")
    factor_final = np.full((n_stocks, n_days), np.nan)
    for j in range(n_days):
        v = factor_neutral[:, j]
        mu = np.nanmean(v)
        sd = np.nanstd(v)
        if sd and sd > 0:
            factor_final[:, j] = (v - mu) / sd

    # ── build DataFrame ─────────────────────────────────────────────────────
    print("Building output …")
    rows = []
    for j, day in enumerate(date_list):
        col = factor_final[:, j]
        valid = np.nonzero(np.isfinite(col))[0]
        for ci in valid:
            rows.append({
                "date":     day.strftime("%Y-%m-%d"),
                "stock_code": codes[ci],
                "factor":   float(col[ci]),
            })

    out_df = pd.DataFrame(rows)
    out_path = OUT_DIR / "factor_trend_rate_diff_v1.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n✅  Saved → {out_path}  ({len(out_df):,} rows)")

    # quick stats
    s = out_df["factor"]
    print(f"\n── Factor Stats ──")
    print(f"  Mean={s.mean():.4f}  Std={s.std():.4f}  "
          f"Min={s.min():.4f}  Max={s.max():.4f}")


if __name__ == "__main__":
    main()
