#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bps_roe_profitab_diverge_v1 — BPS×ROE 盈利效率背离因子（向量化版）

Step 1: 季度截面做 roe ~ log_bps OLS → 残差 z-score（每季度一次）
Step 2: 用有效日>=eff_date 把季度残差前向展开到全日期域
Step 3: 全日期截面一次性 OLS 残差去对数成交额 + 截面 MAD/z-score

输出: data/factor_bps_roe_profitab_diverge_v1.csv
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

BASE  = Path(__file__).resolve().parent.parent
DATA  = BASE / "data"
KLINE = DATA / "csi1000_kline_raw.csv"
FUND  = DATA / "csi1000_fundamental_cache.csv"
OUT   = DATA / "factor_bps_roe_profitab_diverge_v1.csv"

DISCLOSE_DELAY = 25
NEUT_WINDOW    = 20
MAD_K          = 5.0


# ── helpers ──────────────────────────────────────────────────────────────────
def _quarter_resid_one(group: pd.DataFrame) -> pd.DataFrame:
    """对单截面组做 roe~log_bps OLS，输出 (stock_code, q_resid_z)。"""
    g = group.dropna(subset=["bps", "roe"])
    g = g[g.bps > 0]
    if len(g) < 30:
        return pd.DataFrame(columns=["stock_code", "q_resid_z"])
    x = np.log(g["bps"].values)
    y = g["roe"].values / 100.0
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30:
        return pd.DataFrame(columns=["stock_code", "q_resid_z"])
    sl, ic, _, _, _ = sp_stats.linregress(x[ok], y[ok])
    resid = np.full(len(y), np.nan)
    resid[ok] = y[ok] - (sl * x[ok] + ic)
    med   = np.nanmedian(resid)
    mad   = np.nanmedian(np.abs(resid - med)) + 1e-12
    scaled = np.clip((resid - med) / (mad * 1.4826), -MAD_K, MAD_K)
    mu, sd = np.nanmean(scaled), np.nanstd(scaled) + 1e-12
    out = pd.DataFrame({
        "stock_code": g["stock_code"].values,
        "q_resid_z":  (scaled - mu) / sd,
    })
    return out


# ── 1. 季度截面残差 ───────────────────────────────────────────────────────────
def step_quarterly_residual(df_fund: pd.DataFrame) -> pd.DataFrame:
    print("[1/4] 季度 roe ~ log_bps OLS 残差 …")
    parts = []
    for q_date, grp in df_fund.groupby("report_date"):
        r = _quarter_resid_one(grp)
        if r.empty:
            continue
        r["report_date"] = q_date
        r["eff_date"]    = q_date + pd.Timedelta(days=DISCLOSE_DELAY)
        parts.append(r)
    dfq = pd.concat(parts, ignore_index=True)
    print(f"  {len(dfq):,} 条 | {dfq.stock_code.nunique()} 只 | "
          f"{dfq.report_date.nunique()} 季")
    return dfq


# ── 2. 前向展开 → 宽表 (date × stock) ─────────────────────────────────────────
def step_expand(df_q: pd.DataFrame, kline: pd.DataFrame) -> pd.Series:
    print("[2/4] 前向展开到日频 …")
    all_dates = pd.DatetimeIndex(sorted(kline["date"].unique()))

    # 每只股票保留 (eff_date → 下一个eff_date) 内的 q_resid_z
    records = []
    for sc, g in df_q.groupby("stock_code"):
        g = g.sort_values("eff_date").reset_index(drop=True)
        effs = g["eff_date"].values
        vals = g["q_resid_z"].values
        start_d = effs[0]
        end_d   = all_dates[-1] + pd.Timedelta(days=1)
        mask = all_dates >= start_d
        usable = all_dates[mask]
        if usable.size == 0:
            continue
        # searchsorted: 当前日期应使用哪个季度的残差
        idx = np.searchsorted(effs, usable, side="right") - 1
        valid = idx >= 0
        use_dates = usable[valid]
        use_vals  = vals[idx[valid]]
        records.append(pd.DataFrame({
            "date":       use_dates,
            "stock_code": sc,
            "_qraw":      use_vals,
        }))

    raw_long = pd.concat(records, ignore_index=True)
    # pivot
    raw = raw_long.pivot_table(index="date", columns="stock_code", values="_qraw")
    raw = raw.sort_index().sort_index(axis=1)
    raw.columns = raw.columns.astype(str)
    print(f"  宽表 {raw.shape[0]} 日 × {raw.shape[1]} 股")
    return raw


# ── 3. 全截面 OLS 去对数成交额 ───────────────────────────────────────────────
def step_neutralize_amount(raw: pd.Series, kline: pd.DataFrame) -> pd.DataFrame:
    print("[3/4] 全截面 OLS 去成交额 …")

    kline["amt_20d"] = (
        kline.groupby("stock_code")["amount"]
        .transform(lambda s: s.rolling(20, min_periods=5).mean())
    )
    kline["log_amt"] = np.log1p(kline["amt_20d"].clip(lower=1))

    amt_wide = kline.pivot_table(
        index="date", columns="stock_code", values="log_amt")
    amt_wide = amt_wide.sort_index().sort_index(axis=1)
    amt_wide.columns = amt_wide.columns.astype(str)

    common_dates = raw.index.intersection(amt_wide.index)
    common_stocks = raw.columns.intersection(amt_wide.columns)
    raw   = raw.loc[common_dates, common_stocks]
    amt   = amt_wide.loc[common_dates, common_stocks]

    resid = pd.DataFrame(np.nan, index=common_dates, columns=common_stocks)
    for d in common_dates:
        y = raw.loc[d].values.astype(float)
        x = amt.loc[d].values.astype(float)
        ok = np.isfinite(y) & np.isfinite(x)
        if ok.sum() < 30:
            continue
        sl, ic, _, _, _ = sp_stats.linregress(x[ok], y[ok])
        r = y - (sl * x + ic)
        resid.loc[d] = r

    print(f"  中性化残差 {resid.shape[0]} 日 × {resid.shape[1]} 股")
    return resid


# ── 4. 截面 MAD + z-score → 输出 ──────────────────────────────────────────────
def step_zscore_and_save(resid: pd.DataFrame) -> pd.DataFrame:
    print("[4/4] 截面 MAD + z-score …")
    rows = []
    for d, row in resid.iterrows():
        v = row.values.astype(float)
        ok = np.isfinite(v)
        if ok.sum() < 30:
            continue
        med  = np.nanmedian(v)
        mad  = np.nanmedian(np.abs(v[ok] - med)) + 1e-12
        s    = np.clip((v - med) / (mad * 1.4826), -MAD_K, MAD_K)
        mu, sd = np.nanmean(s), np.nanstd(s) + 1e-12
        z = (s - mu) / sd
        for ik, sc in enumerate(resid.columns):
            rows.append({"date": d.strftime("%Y-%m-%d"),
                         "stock_code": sc,
                         "factor_value": float(z[ik])})

    out = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["date", "stock_code", "factor_value"])
    print(f"  {len(out):,} 行 | {out.stock_code.nunique()} 只 | "
          f"{out.date.nunique()} 日")
    if len(out):
        print(f"  mean={out.factor_value.mean():.4f}  "
              f"std={out.factor_value.std():.4f}")
    return out


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  bps_roe_profitab_diverge_v1 — ROE vs BPS 盈利效率背离")
    print("=" * 55)

    kline = pd.read_csv(KLINE, dtype={"stock_code": str})
    kline["stock_code"] = kline["stock_code"].str.strip()
    kline["date"]       = pd.to_datetime(kline["date"])
    kline.sort_values(["stock_code", "date"], inplace=True)

    fund = pd.read_csv(FUND, dtype={"stock_code": str})
    fund["stock_code"]  = fund["stock_code"].str.strip()
    fund["report_date"] = pd.to_datetime(fund["report_date"])
    fund = fund.dropna(subset=["bps", "roe"])
    fund = fund[fund["bps"] > 0].sort_values(["stock_code", "report_date"])

    df_q   = step_quarterly_residual(fund)
    raw    = step_expand(df_q, kline)               # (date, stock) wide
    resid  = step_neutralize_amount(raw, kline)      # after amt-OLS residual
    result = step_zscore_and_save(resid)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.sort_values(["date", "stock_code"]).to_csv(OUT, index=False)
    print(f"\n✅  输出 → {OUT}")


if __name__ == "__main__":
    main()
