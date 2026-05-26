#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROE Acceleration Factor v2
==========================
用最近 9 个季度 ROE 构造两个「同比改善幅度」，
加速度 = yoy_0（最新一期同比）- yoy_1（上一期同比），正向=加速改善。

输出：data/factor_roe_accel_v2.csv  (date, stock_code, factor)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

BASE  = Path(__file__).resolve().parent.parent
DATA  = BASE / "data"
OUT   = DATA / "factor_roe_accel_v2.csv"

# ── load ────────────────────────────────────────────────────────────────────
print("[1] load …")
kl   = pd.read_csv(DATA / "csi1000_kline_raw.csv",   parse_dates=["date"])
fund = pd.read_csv(DATA / "csi1000_fundamental_cache.csv", parse_dates=["report_date"])

trade_dates = sorted(kl["date"].unique())
stocks      = sorted(kl["stock_code"].unique())

# 20-day rolling amount mean (log), for neutralizer
amt_raw = kl.pivot_table(index="date", columns="stock_code", values="amount").sort_index()
amt_20d = np.log(amt_raw.rolling(20, min_periods=10).mean() + 1.0)   # log
# → long
amt_long = amt_20d.stack().rename("log_amt").reset_index()
amt_long.columns = ["date", "stock_code", "log_amt"]

# ── ROE acceleration per report ────────────────────────────────────────────
print("[2] ROE acceleration per report …")
fund = fund.sort_values(["stock_code", "report_date"]).reset_index(drop=True)

results = []
for sc, g in fund.groupby("stock_code"):
    g = g.sort_values("report_date").reset_index(drop=True)
    roes  = g["roe"].values
    rdates = g["report_date"].values
    afrom = (g["report_date"] + pd.Timedelta(days=45)).values   # 披露后45天生效

    # need ≥ 9 non-nan consecutive for clean 2-yoy computation
    # use 8-quarter block: first4 vs last4 → yoy_0; shifted block → yoy_1
    # make the two yoy windows overlap in time as much as possible:
    #    [0..3] [4..7]  are two disjoint 4-quarter blocks → yoy_0 = mean[4:8]-mean[0:4]
    #    [1..4] [5..8]  need 9 quarters; yoy_1 = mean[5:9]-mean[1:5]
    # pick the anchor at position i = min index where we have i+1 >= 9
    valid_idx = np.where(~np.isnan(roes))[0]
    if len(valid_idx) < 9:
        continue

    # use contiguous best 9 most-recent valid quarters
    # greedily expand from the last valid index backwards 8
    last = valid_idx[-1]
    start = max(0, last - 8)
    window = roes[start : last + 1]
    if len(window) < 9:
        # try fill from beginning
        window = roes[valid_idx[:9]]
    if len(window) < 9:
        continue

    yoy_prev = np.nanmean(window[4:9]) - np.nanmean(window[0:5])  # mean[5:9]-mean[1:5]
    yoy_cur  = np.nanmean(window[4:8])  - np.nanmean(window[0:4])  # mean[4:8]-mean[0:4]
    accel = yoy_cur - yoy_prev
    if np.isnan(accel):
        continue

    # assign factor valid from the date when the 9th-quarter-in-window report is known
    avail = pd.Timestamp(rdates[last]) + pd.Timedelta(days=45)

    results.append(dict(stock_code=int(sc), avail_date=avail, factor_raw=accel))

raw = pd.DataFrame(results)
print(f"  report-level rows: {len(raw)}")

# ── expand to daily panel ──────────────────────────────────────────────────
print("[3] expand to daily panel …")
raw = raw.sort_values(["stock_code", "avail_date"])

rows_out = []
for sc, g in raw.groupby("stock_code"):
    g = g.sort_values("avail_date")
    avail_dates = g["avail_date"].values
    vals        = g["factor_raw"].values

    stock_ts = []
    val_idx  = 0
    for dt in trade_dates:
        while val_idx + 1 < len(avail_dates) and avail_dates[val_idx + 1] <= dt:
            val_idx += 1
        if avail_dates[val_idx] <= dt:
            stock_ts.append((dt, int(sc), float(vals[val_idx])))
    rows_out.extend(stock_ts)

panel = pd.DataFrame(rows_out, columns=["date", "stock_code", "factor_raw"])
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.sort_values(["date", "stock_code"]).reset_index(drop=True)
print(f"  raw panel rows: {len(panel)}")

# ── neutralise + winsorise + z-score per cross-section ─────────────────────
print("[4] neutralise …")
amt_long["date"] = pd.to_datetime(amt_long["date"])
panel = panel.merge(amt_long, on=["date", "stock_code"], how="left")

WINSOR_MAD = 5.2
out = []
for dt, g in panel.groupby("date"):
    g = g.dropna(subset=["factor_raw", "log_amt"]).copy()
    if len(g) < 30:
        continue
    x = g["log_amt"].values.reshape(-1, 1)
    y = g["factor_raw"].values
    xm = x - x.mean()
    denom = (xm ** 2).sum()
    beta  = float((xm.ravel() * y).sum() / denom) if denom > 1e-12 else 0.0
    alpha = float(y.mean() - beta * x.mean())
    resid = y - (alpha + beta * x.ravel())

    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med))) * 1.4826
    if mad > 1e-9:
        clipped = np.clip(resid, med - WINSOR_MAD * mad, med + WINSOR_MAD * mad)
        z = (clipped - clipped.mean()) / (clipped.std() + 1e-9)
    else:
        z = np.zeros_like(resid)

    gg = g[["date", "stock_code"]].copy()
    gg["factor"] = z
    out.append(gg)

panel_final = pd.concat(out, ignore_index=True)
print(f"  final rows: {len(panel_final)}")

# ── save ────────────────────────────────────────────────────────────────────
print("[5] save …")
panel_final.to_csv(OUT, index=False)
print(f"  → {OUT}")
print(f"  date range: {panel_final['date'].min().date()} ~ {panel_final['date'].max().date()}")
print(f"  stocks     : {panel_final['stock_code'].nunique()}")
print()
print(panel_final.groupby("date")["factor"].describe().tail(5).to_string())
