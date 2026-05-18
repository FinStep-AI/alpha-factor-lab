#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BPS增速加速度因子 v1 — 直接读csi1000_kline_raw.csv"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent.parent.parent   # alpha-factor-lab/
DATA    = PROJECT / "data"
KLINE   = DATA / "csi1000_kline_raw.csv"
FUND    = DATA / "csi1000_fundamental_cache.csv"
OUT     = DATA / "factor_bps_accel_v1.csv"

DISCLOSE_DELAY  = 25          # 财报发布后天数
NEUT_WINDOW     = 20          # 成交额均值窗口
MAD_K           = 5.5


# ══════════════════════════════════════════════════════
# 1. 读取行情
# ══════════════════════════════════════════════════════
def load_kline(path=KLINE):
    print("[1/5] 读行情 …")
    df = pd.read_csv(path, dtype={"stock_code": str})
    df["stock_code"] = df["stock_code"].str.strip()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["stock_code", "date"], inplace=True)
    print(f"  {df.stock_code.nunique()} 只 | {df.date.min().date()} ~ {df.date.max().date()} | {len(df):,} 行")
    return df


# ══════════════════════════════════════════════════════
# 2. 读取财报
# ══════════════════════════════════════════════════════
def load_fund(path=FUND):
    print("[2/5] 读财报 …")
    df = pd.read_csv(path, dtype={"stock_code": str})
    df["stock_code"] = df["stock_code"].str.strip()
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.dropna(subset=["bps"])
    df = df[df["bps"] > 0].copy()
    df.sort_values(["stock_code", "report_date"], inplace=True)
    print(f"  {df.stock_code.nunique()} 只 | {len(df):,} 条 | {df.report_date.min().date()} ~ {df.report_date.max().date()}")
    return df


# ══════════════════════════════════════════════════════
# 3. 计算 BPS YoY + 加速度（财报层面）
# ══════════════════════════════════════════════════════
def accel_report(df_fund):
    """
    accel = (bps_t/bps_{t-4Q} - 1) - (bps_{t-1Q}/bps_{t-5Q} - 1)
    即 YoY_t - YoY_{t-1Q}
    delay 25d → effective_date = report_date + DISCLOSE_DELAY
    """
    print("[3/5] 算 BPS YoY 加速度 …")
    rows = []

    for sc, g in df_fund.groupby("stock_code"):
        g = g.sort_values("report_date").reset_index(drop=True)
        for i in range(7, len(g)):   # 需要 7 期历史（t, t-1, ..., t-7）
            rep = g.iloc[i]["report_date"]
            bps_t   = g.iloc[i]["bps"]
            bps_m3  = g.iloc[i-3]["bps"]   # 上季度
            bps_m4  = g.iloc[i-4]["bps"]   # 去年同期 (YoY分母)
            bps_m7  = g.iloc[i-7]["bps"]   # 去年同Q前1季度 (上季YoY分母)

            if any(v <= 0 for v in [bps_t, bps_m3, bps_m4, bps_m7]):
                continue
            yoy_curr = bps_t   / bps_m4  - 1
            yoy_prev = bps_m3  / bps_m7  - 1
            accel    = yoy_curr - yoy_prev

            rows.append(dict(
                stock_code    = sc,
                report_date   = rep,
                effective_date= rep + pd.Timedelta(days=DISCLOSE_DELAY),
                bps_yoy_curr  = yoy_curr,
                bps_yoy_prev  = yoy_prev,
                bps_accel     = accel,
            ))

    df = pd.DataFrame(rows)
    print(f"  {len(df):,} 条 accel 记录, {df.stock_code.nunique()} 只 stocks")
    print(f"  有效日期: {df.effective_date.min().date()} ~ {df.effective_date.max().date()}")
    print(f"  accel 分位: {df['bps_accel'].quantile([.05,.25,.5,.75,.95]).round(4).to_dict()}")
    return df


# ══════════════════════════════════════════════════════
# 4. 展平到日频
# ══════════════════════════════════════════════════════
def expand_daily(df_accel, all_dates):
    print("[4/5] 展平到日频 …")
    ts = sorted(all_dates)
    recs = []

    for sc, g in df_accel.groupby("stock_code"):
        g = g.sort_values("effective_date").reset_index(drop=True)
        for j in range(len(g)):
            lo = g.iloc[j]["effective_date"]
            hi = (g.iloc[j+1]["effective_date"]
                  if j + 1 < len(g) else pd.Timestamp(ts[-1]) + pd.Timedelta(days=2))
            mask = (pd.to_datetime(ts) >= lo) & (pd.to_datetime(ts) < hi)
            dts   = pd.to_datetime(ts)[mask]
            if dts.empty:
                continue
            for d in dts:
                recs.append(dict(date=d, stock_code=sc,
                                 accel_raw=g.iloc[j]["bps_accel"]))

    df = pd.DataFrame(recs)
    print(f"  {len(df):,} 行, {df.stock_code.nunique()} 只")
    return df


# ══════════════════════════════════════════════════════
# 5. 截面 OLS 中性化 + MAD + z-score
# ══════════════════════════════════════════════════════
def neutralize(df_kline, df_daily):
    print("[5/5] 截面 OLS 中性化 …")

    # 20日成交额均值
    print("  算 量20unning 均值 …")
    df_kline["amt_20d"] = (
        df_kline.groupby("stock_code")["amount"]
        .transform(lambda x: x.rolling(NEUT_WINDOW, min_periods=5).mean())
    )
    df_kline["log_amt"] = np.log1p(df_kline["amt_20d"].clip(lower=1))

    amt_ref = df_kline[["date", "stock_code", "log_amt"]].drop_duplicates(
        ["date", "stock_code"])

    out = []
    loop_dates = sorted(df_daily.date.unique())

    # 跳过带有 Entry/Exit 的 dict_like attr:labels
    for d in loop_dates:
        acc = df_daily[df_daily["date"] == d]
        if acc.empty:
            continue
        m = acc.merge(amt_ref[amt_ref["date"] == d],
                      on="stock_code", how="inner")
        if len(m) < 30:
            continue
        y    = m["accel_raw"].values
        xlog = m["log_amt"].values

        try:
            sl, ic, _, _, _ = stats.linregress(xlog, y)
            resid = y - (sl * xlog + ic)
        except Exception:
            resid = y

        med   = np.median(resid)
        mad   = np.median(np.abs(resid - med))
        if mad < 1e-10:
            z = np.zeros_like(resid)
        else:
            scaled = np.clip((resid - med) / (mad * 1.4826), -MAD_K, MAD_K)
            mu, sd = scaled.mean(), scaled.std()
            z = (scaled - mu) / (sd + 1e-10)

        for idx_v, (_, row) in enumerate(m.iterrows()):
            out.append(dict(date=d, stock_code=row["stock_code"],
                            factor_value=z[idx_v]))

    df_out = pd.DataFrame(out)
    print(f"  最终: {len(df_out):,} 行, {df_out.stock_code.nunique()} 只")
    print(f"  分布: mean={df_out.factor_value.mean():.4f}  "
          f"std={df_out.factor_value.std():.4f}")
    return df_out


# ══════════════════════════════════════════════════════
def main():
    print("=" * 55)
    print("  BPS 增速加速度因子  v1")
    print("=" * 55)

    kline = load_kline()
    fund  = load_fund()
    all_dates = sorted(kline.date.unique())

    accel = accel_report(fund)
    daily = expand_daily(accel, all_dates)
    final = neutralize(kline, daily)

    out = Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    final.sort_values(["date", "stock_code"]).to_csv(out, index=False)
    print(f"\n✅ 因子输出 → {out}")


if __name__ == "__main__":
    main()
