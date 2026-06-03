#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vwp_skew_v1 — VWAP Deviation Skewness (日内收益偏度因子)
===========================================================
构造：
  1. daily_dev_t = close_t / vwap_t − 1        # 单日 close 相对 VWAP 的偏离
  2. vwp_skew = skewness(daily_dev, 20d)        # 20 日滚动偏度
  3. OLS 截面中性化 (y=vwp_skew, X=log_amount) → MAD → z-score

假设：
  skew > 0 → 大部分时间 close 贴近 VWAP / 略低，偶发跳升到 VWAP 上方
           = 日内买方偶尔占优但整体仍被卖方主导 → 弱反转 / 买方修复
  skew < 0 → 大部分时间 close 贴近 VWAP / 略高，偶发跌穿 VWAP 下方
           = 日内卖方偶尔占优但整体仍被买方主导 → 反转

Barra Style: MICRO  /  Reversal
"""

import numpy as np, pandas as pd, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent   # alpha-factor-lab
WIN  = 20

def main():
    kline = pd.read_csv(BASE / "data" / "csi1000_kline_raw.csv",
                        usecols=["date","stock_code","close","volume","amount"])
    kline["date"] = pd.to_datetime(kline["date"])
    kline["stock_code"] = kline["stock_code"].astype(str).str.zfill(6)
    kline = kline.sort_values(["stock_code","date"]).reset_index(drop=True)

    kline["vwap"]    = kline["amount"] / (kline["volume"]*100).replace(0, np.nan)
    kline["vwp_dev"] = kline["close"] / kline["vwap"] - 1

    kline["vwp_skew"] = kline.groupby("stock_code")["vwp_dev"].transform(
        lambda s: s.rolling(WIN, min_periods=WIN).skew())

    kline = kline.dropna(subset=["vwp_skew"]).copy()

    # OLS amount-neutralize → MAD → z-score  per-date
    out = []
    for dt, g in kline.groupby("date"):
        arr = g[["stock_code","vwp_skew","amount"]].copy()
        amt_med = arr["amount"].median()
        arr["log_amount"] = np.log(arr["amount"].clip(amt_med*0.01, amt_med*100))
        X = np.column_stack([np.ones(len(arr)), arr["log_amount"].values])
        y = arr["vwp_skew"].values
        try:
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            continue
        resid = y - X @ b
        med = np.nanmedian(resid)
        mad = np.nanmedian(np.abs(resid - med)) + 1e-9
        z = np.clip((resid - med)/(1.4826*mad), -5.2, 5.2)
        z = (z - z.mean())/(z.std() + 1e-9)
        out.append(arr[["stock_code"]].assign(value=z, date=dt))

    out_df = pd.concat(out, ignore_index=True)[["date","stock_code","value"]]
    out_df.to_csv(BASE / "data" / "factor_vwp_skew_v1.csv", index=False)
    print(f"saved factor_vwp_skew_v1.csv : {len(out_df):,} rows | dates={out_df['date'].nunique()} | stocks={out_df['stock_code'].nunique()}")
    print(out_df["value"].describe())

if __name__ == "__main__":
    main()
