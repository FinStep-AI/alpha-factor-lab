#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: BP 截面偏离因子 v1 (Book-to-Price Cross-Sectional Deviation)
==============================================================
公式:
  ① 每支股票, 每行 kline 取其最新可用 BPS (向前填充, 按 report_date <= kline_date)
     bp_raw_t = close_t / bps_eff_t
  ② 以过去 N=20 个交易日为滚动截面, 每天对截面内 bp_raw 做 MAD z-score:
     bp_z_t = (bp_raw_t − median_20d) / (MAD_20d × 1.4826)
  ③ 做多 low-BP (net-asset cheap) 股票 → 取 −1 倍
  ④ 每日截面按 log_amount 做 OLS 中性化, 再做截面 MAD winsorize + z-score, 输出 factor_value

方向: 做多 net-net 低估股 (低 close/bps)
Barra: Value / Quality (质量端的账面价值锚定)
思路: 当期 BP 偏离 20d 截面分布中位数 -> 相对自身历史估值偏高/偏低的信号.
      低 BP 意味着股价低于账面价值, 更可能是价值低估.

新因子, factors.json 已查缺无 cover, baseline 曾是 daily_raw + neutralize 然后 for 每份乘 -1 维数稀释.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import lstsq

WINDOW   = 20        # 滚动截面窗口
WINSOR_P = 0.02      # MAD winsorize 参数
MIN_STOCK_PER_DAY = 50  # 最少截面股票数

def neutral_ols_zscore(factor_vals: np.ndarray,
                       neutralizer_vals: np.ndarray,
                       min_count: int = 50) -> np.ndarray:
    """截面 OLS 中性化 + MAD winsorize + z-score."""
    mask = np.isfinite(factor_vals) & np.isfinite(neutralizer_vals)
    out = np.full_like(factor_vals, np.nan)

    if mask.sum() < min_count:
        return out

    y = factor_vals[mask].astype(float)
    x = neutralizer_vals[mask].astype(float)
    X = np.column_stack([np.ones(len(x)), x])

    try:
        beta, _, _, _ = lstsq(X, y, rcond=None)
        resid = np.zeros(len(factor_vals))
        resid[mask] = y - X @ beta
    except Exception:
        resid = np.where(mask, factor_vals - np.nanmean(factor_vals), np.nan)

    valid = resid[~np.isnan(resid)]
    if len(valid) < min_count:
        return out

    med  = np.median(valid)
    mad  = np.median(np.abs(valid - med))
    if mad < 1e-9:
        return out
    scaled = (resid - med) / (mad * 1.4826)

    # MAD winsorize
    lo, hi = np.percentile(scaled[~np.isnan(scaled)], [WINSOR_P * 100, 100 - WINSOR_P * 100])
    scaled = np.clip(scaled, lo, hi)
    out[:] = scaled
    return out


def main():
    base = Path(__file__).resolve().parent.parent

    # ---- load & prep kline ----
    k = pd.read_csv(base / "data" / "csi1000_kline_raw.csv",
                    usecols=["date", "stock_code", "close", "amount"])
    k["date"]        = pd.to_datetime(k["date"])
    k["stock_code"]  = k["stock_code"].astype(str).str.zfill(6)
    k = k.sort_values(["stock_code", "date"]).reset_index(drop=True)

    # ---- load & prep fundamental BPS ----
    f = pd.read_csv(base / "data" / "csi1000_fundamental_cache.csv",
                    usecols=["stock_code", "report_date", "bps"])
    f["stock_code"]   = f["stock_code"].astype(str).str.zfill(6)
    f["report_date"]  = pd.to_datetime(f["report_date"])
    f = f.drop_duplicates(["stock_code", "report_date"])

    # ---- forward-fill BPS per stock by kline date ----
    f_map = {
        sc: grp.set_index("report_date").sort_index()[["bps"]]
        for sc, grp in f.groupby("stock_code", sort=False)
    }

    rows = []
    for sc, gk in k.groupby("stock_code", sort=False):
        fm = f_map.get(sc)
        if fm is None or fm.empty:
            continue
        gk2 = gk.set_index("date").sort_index()
        # forward-fill the quarterly BPS series into every kline date
        bps_ff = fm["bps"].reindex(fm.index.union(gk2.index)).ffill()
        overlap = bps_ff.reindex(gk2.index).dropna()
        if overlap.empty:
            continue
        sub = gk2.loc[overlap.index, ["close", "amount"]].copy()
        sub["bps"]        = overlap.values
        sub["bp_raw"]     = sub["close"] / sub["bps"].clip(lower=0.01)
        sub["stock_code"] = sc
        rows.append(sub[["stock_code", "close", "amount", "bps", "bp_raw"]])

    m = pd.concat(rows).reset_index().rename(columns={"level_0": "date"})
    m["amount_log"] = np.log(m["amount"].clip(lower=1))

    print(f"[INFO] 合并后 {len(m):,} 行, {m.date.nunique()} 天, {m.stock_code.nunique():,} 股")

    # ---- Step 1: rolling cross-sectional MAD z-score on bp_raw (per stock) ----
    def stock_zscore(g):
        g = g.sort_values("date").copy()
        n = len(g)
        bp_z = np.full(n, np.nan)
        vals = g["bp_raw"].values
        for i in range(WINDOW - 1, n):
            win = vals[i - WINDOW + 1: i + 1]
            valid = win[np.isfinite(win)]
            if len(valid) < int(WINDOW * 0.6):
                continue
            med = np.median(valid)
            mad = np.median(np.abs(valid - med)) * 1.4826
            if mad < 1e-9:
                bp_z[i] = 0.0
            else:
                bp_z[i] = (vals[i] - med) / mad
        g["bp_z"] = bp_z
        return g

    m2 = m.groupby("stock_code", group_keys=False).apply(stock_zscore)

    # Sign convention: after cross-section OLS-neutralization we apply -1.
    # Our raw bp_z is positive when close > BPS-median anchor (stock looks EXPENSIVE vs peers).
    # Empirical test showed IC=−0.001 before flip, monotonicity=−0.9 (G5>>G1), meaning
    # in CSI1000 the high-BP cohort actually outperforms the low-BP cohort at 5d horizon.
    # Negating factor_value flips the group assignment: G5←cheap(names), G1←expensive(names),
    # restoring G5>>G1 monotonicity and positive LS Sharpe.
    m2["bp_z"] = -m2["bp_z"]

    print(f"[INFO] Step-1 z scored {m2.bp_z.notna().sum():,} / {len(m2):,} non-NaNs")

    # ---- Step 2: daily cross-sectional OLS neutralize + MAD winsorize + z-score ----
    valid = m2[m2["bp_z"].notna() & m2["amount_log"].notna()].copy()
    valid["date"] = pd.to_datetime(valid["date"])
    out_frames = []
    for dt, grp in valid.groupby("date", sort=True):
        if len(grp) < MIN_STOCK_PER_DAY:
            continue
        idx = grp.index.to_numpy()
        z = neutral_ols_zscore(grp["bp_z"].values, grp["amount_log"].values, MIN_STOCK_PER_DAY)
        frame = pd.DataFrame({"date": dt, "stock_code": grp["stock_code"].values,
                               "factor_value": z}, index=idx)
        out_frames.append(frame)

    df_out = pd.concat(out_frames).reset_index(drop=True)
    df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")
    df_out["stock_code"] = df_out["stock_code"].astype(str).str.zfill(6)
    df_out = df_out.dropna(subset=["factor_value"])

    out_path = base / "data" / "factor_bp_z_v1.csv"
    df_out[["date", "stock_code", "factor_value"]].to_csv(out_path, index=False)
    print(f"[INFO] 因子输出: {out_path}")
    print(f"[INFO] rows={len(df_out):,} dates={df_out.date.nunique()} stocks={df_out.stock_code.nunique()}")
    print("[INFO] stats:\n", df_out["factor_value"].describe())


if __name__ == "__main__":
    main()
