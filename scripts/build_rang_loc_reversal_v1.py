#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rang_loc_reversal_v1 -- 日内位置效率反转因子

  信号：ma20(rang_loc) - today_rang_loc
       >0  表示今日收盘低于过去20日平均日内位置 (=偏向日内低位，偏弱)，
            反转逻辑下"弱"蕴含次日/未来反弹预期。
  中性化：全截面 OLS 去 log(20d 成交额量) -> MAD 缩尾 -> 截面 z-score
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

BASE   = Path(__file__).resolve().parent.parent
DATA   = BASE / "data"
KLINE  = DATA / "csi1000_kline_raw.csv"
OUTCSV = DATA / "factor_rang_loc_reversal_v1.csv"


def main():
    print("=" * 55)
    print("  rang_loc_reversal_v1 -- intraday-location reversal")
    print("=" * 55)

    # 1. kline
    print("[1/4] load kline ...")
    kl = pd.read_csv(KLINE, dtype={"stock_code": str})
    kl["stock_code"] = kl["stock_code"].str.strip()
    kl["date"] = pd.to_datetime(kl["date"])
    kl = kl.sort_values(["stock_code", "date"]).reset_index(drop=True)

    kl["amt_20"] = (
        kl.groupby("stock_code")["amount"]
        .transform(lambda s: s.rolling(20, min_periods=5).mean())
    )
    kl["log_amt"] = np.log1p(kl["amt_20"].clip(lower=1))

    hl = kl["high"] - kl["low"]
    hl = hl.replace(0, np.nan)
    kl["rang_loc"] = (kl["close"] - kl["low"]) / hl  # 0==close at low, 1==close at high

    ma20loc = (
        kl.groupby("stock_code")["rang_loc"]
        .transform(lambda s: s.rolling(20, min_periods=10).mean())
    )

    # raw resid = rang_loc - ma20;  positive = today's close higher than recent average
    # reversal signal => use ma20 - rang_loc  (positive = close below trend)
    kl["factor_raw"] = ma20loc - kl["rang_loc"]

    print(f"  {kl.stock_code.nunique()} stk | {kl.date.min().date()} ~ {kl.date.max().date()}")

    # 2. pivot wide
    print("[2/4] pivot ...")
    FV = kl.pivot_table(index="date", columns="stock_code", values="factor_raw").sort_index()
    AM = kl.pivot_table(index="date", columns="stock_code", values="log_amt").sort_index()

    ccs = sorted(FV.columns.intersection(AM.columns))
    FV = FV.loc[:, ccs]
    AM = AM.loc[:, ccs]
    dates = sorted(FV.index)

    print(f"  shape = {FV.shape[0]} d x {FV.shape[1]} stk")

    # 3. cross-sectional OLS, MAD, z-score
    print("[3/4] cross-sectional OLS resid + MAD + z-score ...")
    recs = []

    for d in dates:
        fv = FV.loc[d].values.astype(float)
        av = AM.loc[d].values.astype(float)
        ok = np.isfinite(fv) & np.isfinite(av)
        if ok.sum() < 40:
            continue

        slope, intercept, _, _, _ = sp_stats.linregress(av[ok], fv[ok])
        ling = np.where(np.isfinite(fv), fv - (slope * av + intercept), np.nan)
        ok2 = np.isfinite(ling)
        if ok2.sum() < 30:
            continue

        x   = ling[ok2]
        med  = np.nanmedian(x)
        mad  = np.nanmedian(np.abs(x - med)) + 1e-12
        sc   = np.clip((x - med) / (mad * 1.4826), -5.0, 5.0)
        ok3  = np.isfinite(sc)
        if ok3.sum() < 30:
            continue

        s3  = sc[ok3]
        mu  = float(np.mean(s3))
        sd  = float(np.std(s3)) + 1e-12
        zz  = (s3 - mu) / sd

        for ik, scode in enumerate(np.array(ccs)[ok2][ok3]):
            recs.append({
                "date":         str(pd.Timestamp(d).date()),
                "stock_code":   scode,
                "factor_value": float(zz[ik]),
            })

    out = pd.DataFrame(recs, columns=["date", "stock_code", "factor_value"])
    print(f"  {len(out):,} rows, {out.stock_code.nunique()} stk, {out.date.nunique()} d")
    if len(out):
        print(f"  mean={out.factor_value.mean():.4f}  std={out.factor_value.std():.4f}  "
              f"[{out.factor_value.min():.2f}, {out.factor_value.max():.2f}]")

    # 4. write
    print("[4/4] write CSV ...")
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values(["date", "stock_code"]).to_csv(OUTCSV, index=False)
    print(f"\n{OUTCSV}")


if __name__ == "__main__":
    main()
