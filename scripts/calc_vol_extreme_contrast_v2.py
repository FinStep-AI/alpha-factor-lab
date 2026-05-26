#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vol_extreme_contrast_v2 — 成交量极值回报差因子
==============================================
v1 研究记录: IC=0.0595, t=4.46, LS Sharpe=1.29, G5 ann=21.65%, mono=0.5
v2 改进: 同公式, 加 rank + OLS 成交額中性化 + MAD winsorize + z-score

因子公式:
  raw_i(t) = E[fwd5_ret | vol ∈ Q70~100% in rolling 20d stock-window]
           − E[fwd5_ret | vol ∈ Q0~30%   in rolling 20d stock-window]

parameter:  WIN=20  HIGH_Q=0.70  LOW_Q=0.30  FWD=5d
"""
import os, json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

KLINE   = "data/csi1000_kline_raw.csv"
RET     = "data/csi1000_returns.csv"
OUT_CSV = "data/factor_vol_extreme_contrast_v2.csv"
OUT_DIR = "output/vol_extreme_contrast_v2"

WIN      = 20
HIGH_Q   = 0.70
LOW_Q    = 0.30
FWD      = 5
MAD_K    = 3.0
MIN_HI   = 3    # min high-vol days in window to compute mean
MIN_LO   = 3

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load + fwd5 merge ─────────────────────────────────────────────────────
print("[1/4] load + compute fwd5 …", flush=True)
df = pd.read_csv(KLINE, parse_dates=["date"]).sort_values(
    ["stock_code", "date"]
).reset_index(drop=True)

ret_df  = pd.read_csv(RET, parse_dates=["date"])
ridx    = ret_df.set_index(["date", "stock_code"])["return"].unstack("stock_code")
# 20d rolling sum *centered on that date* then shift -5 rows
fwd5piv = ridx.rolling(FWD, min_periods=FWD).sum().shift(-FWD)
multi   = pd.MultiIndex.from_product(
    [ridx.index, ridx.columns], names=["date", "stock_code"]
)
fwd5_s  = pd.Series(fwd5piv.values.ravel(), index=multi, name="fwd5")
df      = df.merge(fwd5_s, on=["date", "stock_code"], how="left")
print(f"  fwd5 non-null: {df.fwd5.notna().sum():,}", flush=True)

# ── 2. Per-stock rolling vol-quantile flags ───────────────────────────────────
print("[2/4] rolling vol-quantile flags …", flush=True)

def band_flags(vol_arr, n, win, hq, lq):
    lo = np.zeros(n, dtype=bool)
    hi = np.zeros(n, dtype=bool)
    for i in range(win - 1, n):
        w = vol_arr[i - win + 1 : i + 1]
        qlo, qhi = np.nanquantile(w, [lq, hq])
        v = vol_arr[i]
        if np.isfinite(v):
            lo[i] = v <= qlo
            hi[i] = v >= qhi
    return lo, hi

lo_all = np.zeros(len(df), dtype=bool)
hi_all = np.zeros(len(df), dtype=bool)
win    = WIN
half   = max(win // 2, 1)
for sc, gidx in df.groupby("stock_code", sort=False).groups.items():
    gidx = np.sort(gidx.to_numpy())
    vol  = df["volume"].values[gidx]
    lo, hi = band_flags(vol, len(vol), win, HIGH_Q, LOW_Q)
    lo_all[gidx] = lo
    hi_all[gidx] = hi
df["is_lo"] = lo_all
df["is_hi"] = hi_all
print(f"  hi-days {hi_all.sum():,}  lo-days {lo_all.sum():,}", flush=True)

# ── 3. Per-stock rolling 20d contrast ────────────────────────────────────────
print("[3/4] rolling 20d high-vol mean − low-vol mean …", flush=True)

def rolling_contrast(g):
    dates = g["date"].values
    f5    = g["fwd5"].values
    ih    = g["is_hi"].values
    il    = g["is_lo"].values
    n     = len(g)
    raw   = np.full(n, np.nan)
    lo    = 0
    for hi in range(n):
        while hi > lo and hi - lo >= win and (
            pd.Timestamp(dates[hi]) - pd.Timestamp(dates[lo])
        ).days > 60:
            lo += 1
        if hi - lo + 1 < half:
            continue
        wi = ih[lo : hi + 1]
        wl = il[lo : hi + 1]
        nh = wi.sum()
        nl = wl.sum()
        if nh >= MIN_HI and nl >= MIN_LO:
            raw[hi] = np.nanmean(f5[lo : hi + 1][wi]) - np.nanmean(
                f5[lo : hi + 1][wl]
            )
    return pd.Series(raw, index=g.index, name="factor_raw")

raw_s = df.groupby("stock_code", sort=False, group_keys=False).apply(
    rolling_contrast
)
df["factor_raw"] = raw_s
df["factor_raw"] = df.groupby("stock_code")["factor_raw"].ffill(limit=3)
print(f"  raw non-null: {df.factor_raw.notna().sum():,}", flush=True)

# ── 4. Cross-section rank→OLS neutralize→MAD→zscore ──────────────────────────
print("[4/4] cross-section transform …", flush=True)

frames = []
for dt, sub in df.groupby("date", sort=True):
    sub2 = sub[["stock_code", "factor_raw", "amount"]].dropna(subset=["factor_raw"])
    if len(sub2) < 50:
        continue
    x  = sub2["factor_raw"].values.astype(float)
    w  = np.log(sub2["amount"].values.astype(float) + 1)
    m  = np.isfinite(x) & np.isfinite(w)
    if m.sum() < 50:
        continue
    x0, w0, sc0 = x[m], w[m], sub2["stock_code"].values[m]

    # rank  → [0, 1]
    xr = rankdata(x0, method="average") / (len(x0) + 1)

    # OLS neutralize by log_amount  →  residuals
    X = np.column_stack([np.ones(len(xr)), w0])
    beta, _, _, _ = np.linalg.lstsq(X, xr, rcond=None)
    resid = xr - X @ beta

    # MAD winsorize
    med = np.nanmedian(resid)
    mad = np.nanmedian(np.abs(resid - med)) * 1.4826 + 1e-12
    z   = (resid - med) / mad
    z   = np.clip(z, -MAD_K, MAD_K)

    # z-score
    mu, sd = np.nanmean(z), np.nanstd(z) + 1e-12
    z = (z - mu) / sd
    frames.append(pd.DataFrame({"date": dt, "stock_code": sc0, "factor_vol_extreme_contrast_v2": z}))

result = pd.concat(frames, ignore_index=True).sort_values(
    ["date", "stock_code"]
).reset_index(drop=True)
result["date"] = result["date"].dt.strftime("%Y-%m-%d")
result.to_csv(OUT_CSV, index=False)
print(f"\n✅  {len(result):,} rows → {OUT_CSV}", flush=True)
print(result["factor_vol_extreme_contrast_v2"].describe(), flush=True)

meta = dict(
    factor_id              = "vol_extreme_contrast_v2",
    version                = "v2",
    source_factor_id       = "vol_extreme_contrast_v1",
    formula                = (
        "mean(fwd5_ret | vol ∈ Q70-100%, 20d) "
        "− mean(fwd5_ret | vol ∈ Q0-30%,   20d)  per stock; "
        "cross-section: rank(x) + OLS neutralize(log_amount) + MAD + zscore"
    ),
    win=WIN, high_q=HIGH_Q, low_q=LOW_Q, fwd_days=FWD,
    n_dates=int(result["date"].nunique()),
    n_stocks=int(result["stock_code"].nunique()),
    n_rows=len(result),
)
with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
