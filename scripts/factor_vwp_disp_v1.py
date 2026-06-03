#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vwp_disp_v1 — VWAP Position Consistency / Dispersion
=====================================================
因子构造：  sign(mean(close/vwap−1, 20d)) × std(close/vwap−1, 20d)
含义：方向明确且一致  ←→  符号表征"偏向哪一侧"，std表征"一致性/确定性"
正向使用：做多方向最一致型（正因子值=方向明确+低混沌=正alpha）

中性化：对数成交额 OLS + MAD 5.2σ 缩尾 + z-score
"""

import numpy as np, pandas as pd, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent        # alpha-factor-lab root
WIN  = 20

def main():
    kline = pd.read_csv(BASE / "data" / "csi1000_kline_raw.csv",
                        usecols=["date","stock_code","open","high","low",
                                 "close","volume","amount","turnover"])
    kline["date"]       = pd.to_datetime(kline["date"])
    kline["stock_code"] = kline["stock_code"].astype(str).str.zfill(6)
    kline = kline.sort_values(["stock_code","date"]).reset_index(drop=True)

    # VWAP  &  日度偏离
    kline["vwap"]     = kline["amount"] / (kline["volume"] * 100).replace(0, np.nan)
    kline["vwp_dev"]  = kline["close"] / kline["vwap"] - 1

    # 滚动 mean / std
    gb = kline.groupby("stock_code")["vwp_dev"]
    kline["vwp_mean"] = gb.transform(lambda s: s.rolling(WIN, min_periods=WIN).mean())
    kline["vwp_std"]  = gb.transform(lambda s: s.rolling(WIN, min_periods=WIN).std())
    kline["factor_raw_direction"] = np.sign(kline["vwp_mean"]) * kline["vwp_std"]

    kline = kline.dropna(subset=["factor_raw_direction"]).copy()

    # ——— 截面中性化 ———
    records = []
    for dt, grp in kline.groupby("date"):
        arr = grp[["stock_code","factor_raw_direction","amount"]].copy()
        amt_med = arr["amount"].median()
        arr["log_amount"] = np.log(arr["amount"].clip(amt_med*0.01, amt_med*100))

        X = np.column_stack([np.ones(len(arr)), arr["log_amount"].values])
        y = arr["factor_raw_direction"].values
        try:
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            arr["value"] = 0.0
            records.append(arr[["stock_code","value"]].assign(date=dt))
            continue

        resid = y - X @ b
        med   = np.nanmedian(resid)
        mad   = np.nanmedian(np.abs(resid - med)) + 1e-9
        z     = np.clip((resid - med) / (1.4826 * mad), -5.2, 5.2)
        z     = (z - z.mean()) / (z.std() + 1e-9)

        records.append(arr[["stock_code"]].assign(value=z, date=dt))

    out = pd.concat(records, ignore_index=True)[["date","stock_code","value"]]
    out.to_csv(BASE / "data" / "factor_vwp_disp_v1.csv", index=False)
    print(f"rows={len(out):,}  dates={out['date'].nunique()}  stocks={out['stock_code'].nunique()}")
    print(out.head(3))

if __name__ == "__main__":
    main()
