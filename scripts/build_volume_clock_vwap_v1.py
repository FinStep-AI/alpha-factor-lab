#!/usr/bin/env python3
"""
因子: 成交量时钟 VWAP 偏离 (Volume-Clock VWAP Deviation) v1
factor_id: volume_clock_vwap_v1

与 vwap_dev_v1 的区别
---------------------
vwap_dev_v1 = MA20((close - VWAP) / ATR)，
看的是收盘价相对传统 VWAP = Σ(price×vol)/Σ(vol) 的位置（尾盘端）。

volume_clock_vwap_v1 用典型价格 TP=(2·close − high − low)/2 作成交价代理，
占终日成交额 k = vwap(tp_approx)，
换一个角度去看（较高价格 ⇒ 重心偏高 ⇒ 高成交价 看不是纯在你方向不 Kurz Laurence）。

构造
  1. TP(t) = (2·close − high − low) / 2
  2. VWAP_approx_20d = Σ(TP·amount) / Σ(amount)，20日滚动
  3. vol_bias = (close − VWAP_approx) / ATR(20)
  4. factor_raw = MA20(vol_bias)
  5. 成交额 OLS 中性化 + 3σ MAD 缩尾 + z-score

假设
  高值 = 收盘高于成交量加权全天重心，A 股小盘股多在信息流确认型动量逻辑
"""
import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

WINDOW = 20
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KLINE = os.path.join(BASE, "data", "csi1000_kline_raw.csv")
OUT   = os.path.join(BASE, "data", "factor_volume_clock_vwap_v1.csv")


def neutralize_cs(group):
    y = group["factor_raw"].values.astype(float).copy()
    x = group["log_amount_20d"].values.astype(float).copy()
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 40:
        return pd.Series(np.nan, index=group.index)
    yv, xv = y[ok].copy(), x[ok].copy()
    med = np.median(yv)
    mad = np.median(np.abs(yv - med)) * 1.4826
    if mad > 0:
        yv = np.clip(yv, med - 3 * mad, med + 3 * mad)
    X = np.column_stack([np.ones(len(xv)), xv])
    b = np.linalg.lstsq(X, yv, rcond=None)[0]
    resid = yv - X @ b
    mu, sd = resid.mean(), resid.std()
    if sd < 1e-9:
        return pd.Series(np.nan, index=group.index)
    out = np.full(len(y), np.nan)
    out[ok] = (resid - mu) / sd
    return pd.Series(out, index=group.index)


def main():
    print(f"[1] load {KLINE}")
    df = pd.read_csv(KLINE, dtype={"stock_code": str})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    df["stock_code"] = df["stock_code"].str.zfill(6)

    g = df.groupby("stock_code", sort=False)

    # --- prev close & TR ---
    df["prev_close"] = g["close"].shift(1)
    df["tr1"] = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - df["prev_close"]).abs(),
        (df["low"]  - df["prev_close"]).abs(),
    ], axis=1).max(axis=1)

    # --- typical price ---
    df["tp"] = (2.0 * df["close"] - df["high"] - df["low"]) / 2.0

    # --- 20d rolling VWAP(TP) and ATR ---
    def roll_num(sidx):
        sub = df.loc[sidx]
        amt = sub["amount"].values.astype(float)
        tp  = sub["tp"].values.astype(float)
        return pd.Series(
            (pd.Series(tp * amt, index=sidx).rolling(WINDOW, min_periods=14).sum()).values,
            index=sidx
        )

    df["vwap_num"] = 0.0
    df["atr20"]    = 0.0
    for code, idx in g.groups.items():
        idx = df.loc[idx].index  # already sorted
        sub = df.loc[idx]
        vnum = (sub["tp"] * sub["amount"]).rolling(WINDOW, min_periods=14).sum()
        vden = sub["amount"].rolling(WINDOW, min_periods=14).sum().clip(lower=1)
        df.loc[idx, "vwap_num"] = vnum.values
        df.loc[idx, "vwap_den"] = vden.values
        df.loc[idx, "atr20"]   = sub["tr1"].rolling(WINDOW, min_periods=14).mean().values

    df["vwap_approx"] = df["vwap_num"] / df["vwap_den"]
    df["vol_bias"]    = (df["close"] - df["vwap_approx"]) / df["atr20"].clip(lower=1e-6)

    # smoothed raw factor
    g = df.groupby("stock_code", sort=False)
    df["factor_raw"] = g["vol_bias"].transform(
        lambda s: s.rolling(WINDOW, min_periods=14).mean()
    )
    df["mean_amt_20d"] = g["amount"].transform(
        lambda s: s.rolling(WINDOW, min_periods=16).mean()
    )
    df["log_amount_20d"] = np.log(df["mean_amt_20d"].clip(lower=1))

    fdf = df[["date", "stock_code", "factor_raw", "log_amount_20d"]].dropna().copy()
    print(f"  raw rows = {len(fdf)}")

    print("[2] neutralize …")
    fdf["factor"] = (
        fdf.groupby("date", group_keys=False).apply(neutralize_cs).values
    )

    out = (fdf[["date", "stock_code", "factor"]]
           .dropna(subset=["factor"])
           .sort_values(["date", "stock_code"]))
    out.to_csv(OUT, index=False)
    print(f"[3] {OUT}  rows={len(out)}  "
          f"{out['date'].min().date()} ~ {out['date'].max().date()}  "
          f"avg stocks/date = {out.groupby('date')['stock_code'].count().mean():.0f}")
    print(f"    factor: mean={out['factor'].mean():+.4f}  std={out['factor'].std():.4f}")


if __name__ == "__main__":
    main()
