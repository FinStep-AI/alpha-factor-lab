#!/usr/bin/env python3


"""
ret_asym_j2020 - Patil/Jiang tail-asymmetry factor
version: j2020_v1

Patil et al.2012 + Jiang/Wu/Zhou 2020(JFQA) + Chen/Wu/Zhu 2022(PBFJ)
"""


import warnings, argparse, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
HOME = Path(__file__).resolve().parent.parent


def _ie(rv):
    n = len(rv)
    if n < 2: return 0.0
    sg = rv.std(ddof=0)
    if sg < 1e-14: return 1e4 if abs(rv.mean()) > 0 else 0.0
    return abs(rv.mean()) / sg


def _m_ret(win, k):
    r = np.asarray(win, dtype="f8")
    med = np.median(r)
    M = np.median(np.abs(r))
    if M < 1e-14: M = 1e-14
    t = k * M
    up = r[r > med + t] - med
    dn = r[r < med - t] - med
    return _ie(up) - _ie(dn)


def _neutralize(fraw, sz, w=3.0):
    f = fraw.copy(); s = sz.copy()
    ok = f.notna() & s.notna()
    if ok.sum() < 50: return pd.Series(np.nan, index=f.index)
    x, y = s[ok].values, f[ok].values
    X = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ b
    f2 = pd.Series(np.nan, index=f.index); f2[ok] = res
    med = f2.median(); mad = (f2 - med).abs().median() * 1.4826
    if mad < 1e-14: mad = 1e-14
    lo, hi = med - w*mad, med + w*mad
    f3 = f2.clip(lo, hi)
    m2, s2 = f3.mean(), f3.std(ddof=0)
    if s2 < 1e-14: return pd.Series(0.0, index=f.index)
    return (f3 - m2) / s2


def run(version="j2020_v1", ret_lookback=5, k_sigma=0.30, roll_window=20, winsor=3.0):
    ret = pd.read_csv(HOME / "data" / "csi1000_returns.csv", parse_dates=["date"])
    kln = pd.read_csv(HOME / "data" / "csi1000_kline_raw.csv", parse_dates=["date"])
    kln = kln.sort_values(["date","stock_code"]).reset_index(drop=True)
    amt20 = kln.groupby("stock_code")["amount"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    kln["amt20"] = amt20
    amt_map = kln[["date","stock_code","amt20"]].drop_duplicates(["date","stock_code"])
    ret  = ret.sort_values(["date","stock_code"]).reset_index(drop=True)
    df   = ret.merge(amt_map, on=["date","stock_code"], how="left")
    df   = df.sort_values(["date","stock_code"]).reset_index(drop=True)
    recs = []
    for code, g in df.groupby("stock_code", sort=False):
        g = g.sort_values("date")
        r = g["return"].values.astype("f8"); n = len(r)
        mrets = np.full(n, np.nan)
        for i in range(ret_lookback - 1, n):
            w = r[max(0, i - roll_window + 1): i + 1]
            mrets[i] = _m_ret(w, k_sigma)
        tmp = g[["date","stock_code"]].copy()
        tmp["factor_raw"] = mrets
        tmp["amt20"]    = g["amt20"].values
        recs.append(tmp)
    raw = pd.concat(recs, ignore_index=True)
    raw["log_amt"] = np.log(raw["amt20"].clip(lower=1.0))
    raw["factor_z"] = raw.groupby("date", group_keys=False).apply(lambda df: _neutralize(df["factor_raw"], df["log_amt"], winsor))
    out = HOME / "data" / f"factor_{version}.csv"
    raw[["date","stock_code","factor_raw","factor_z"]].dropna(subset=["factor_z"]).to_csv(out, index=False)
    print(f"{out}  n={raw.factor_z.notna().sum()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="j2020_v1")
    ap.add_argument("--ret_lookback", type=int, default=5)
    ap.add_argument("--k_sigma", type=float, default=0.30)
    ap.add_argument("--roll_window", type=int, default=20)
    ap.add_argument("--winsor", type=float, default=3.0)
    A = ap.parse_args(); run(A.version, A.ret_lookback, A.k_sigma, A.roll_window, A.winsor)

