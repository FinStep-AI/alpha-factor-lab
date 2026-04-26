#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Turnover CV Negative v1 — 换手率变异系数(负)"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
KLINE = DATA_DIR / "csi1000_kline_raw.csv"
OUT   = DATA_DIR / "factor_turnover_cv_neg_v1.csv"
WINDOW = 20

print("=== TurnOver CV Negative v1 ===")
df = pd.read_csv(KLINE, parse_dates=["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

g = df.groupby("stock_code")

feat = []
for code, grp in g:
    grp = grp.sort_values("date").copy()
    grp["to_cv_raw"] = -(grp["turnover"].rolling(WINDOW, min_periods=15).std() /
                          grp["turnover"].rolling(WINDOW, min_periods=15).mean().clip(lower=1e-6))
    grp["log_amount"] = np.log1p(grp["amount"].clip(lower=1))
    feat.append(grp[["date","stock_code","to_cv_raw","log_amount"]])

allf = pd.concat(feat).dropna(subset=["to_cv_raw","log_amount"])

# 截面OLS中和
def _ols_neutral(to, amt):
    X = np.column_stack([np.ones(len(amt)), amt])
    try:
        b = np.linalg.lstsq(X, to, rcond=None)[0]
        return to - X @ b
    except:
        return to * np.nan

res = []
for dt, sub in allf.groupby("date"):
    if len(sub) < 30:
        continue
    neu = _ols_neutral(sub["to_cv_raw"].values, sub["log_amount"].values)
    tmp = sub.copy()
    tmp["tocv_neu"] = neu
    res.append(tmp[["date","stock_code","tocv_neu"]])

out = pd.concat(res).dropna()
# MAD + z-score
def mad_z(s):
    med = s.median(); mad = (s-med).abs().median()*1.4826
    if mad < 1e-8: return s*0.0
    return ((s-med)/mad).clip(-5,5)

out["factor_value"] = out.groupby("date")["tocv_neu"].transform(mad_z)
out.to_csv(OUT, index=False)
print(f"写入: {OUT} ({out['date'].nunique()}天, 均{int(out.groupby('date')['stock_code'].count().mean())}股)")
