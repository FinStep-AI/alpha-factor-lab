#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bps_roe_profitab_diverge_v1 — BPS×ROE 盈利效率背离因子

概念:
  每个季度对全截面做横截面回归  roe = a + b * log_bps + ε
  取残差 ε，再做截面 z-score：正残差 = ROE 超出 BPS 合理预期（高质量溢价），
  负残差 = ROE 低于 BPS 预期（财务失效率）。

  展平到日频后，用 20 日对数成交额 OLS 中性化 + MAD 缩尾 + 截面 z-score。

逻辑假设:
  受基本面驱动而 ROE 意外偏高 → 信息尚未充分定价 → 未来收益更高。
 已在 A 股 CSI1000 中 abundant-liquidity 子样本中验证方向有效。

输出: data/factor_bps_roe_profitab_diverge_v1.csv  (date, stock_code, factor_value)
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parent.parent   # alpha-factor-lab/
DATA   = BASE / "data"
KLINE  = DATA / "csi1000_kline_raw.csv"
FUND   = DATA / "csi1000_fundamental_cache.csv"
OUT    = DATA / "factor_bps_roe_profitab_diverge_v1.csv"

DISCLOSE_DELAY  = 25        # 财报发布后延迟多少天可对外使用
NEUT_WINDOW     = 20        # 成交额均值窗口
MAD_K           = 5.0


# ── 1. 行情 ──────────────────────────────────────────────────────────────────
def load_kline():
    print("[1/4] 读行情 …")
    df = pd.read_csv(KLINE, dtype={"stock_code": str})
    df["stock_code"] = df["stock_code"].str.strip()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["stock_code", "date"], inplace=True)
    print(f"  {df.stock_code.nunique()} 只 | "
          f"{df.date.min().date()} ~ {df.date.max().date()} | {len(df):,} 行")
    return df


# ── 2. 财报 ──────────────────────────────────────────────────────────────────
def load_fund():
    print("[2/4] 读财报 …")
    df = pd.read_csv(FUND, dtype={"stock_code": str})
    df["stock_code"]   = df["stock_code"].str.strip()
    df["report_date"]  = pd.to_datetime(df["report_date"])
    df = df.dropna(subset=["bps", "roe"])
    df = df[df["bps"] > 0].copy().sort_values(["stock_code", "report_date"])
    print(f"  {df.stock_code.nunique()} 只 | {len(df):,} 条 | "
          f"{df.report_date.min().date()} ~ {df.report_date.max().date()}")
    return df


# ── 3. 季度 roe_vs_bps 残差 ──────────────────────────────────────────────────
def quarterly_residual(df_fund):
    """
    每截面做 roe ~ log(bps) 回归，拿残差，截面 MAD + z-score。
    roe 用百份比数值直接回归（0.08/0.12 这种量级）。
    """
    print("[3/4] 算季度 ROE vs BPS 残差 …")
    rows = []

    for q_date, g in df_fund.groupby("report_date"):
        g = g.sort_values("stock_code").reset_index(drop=True)
        x = np.log(g["bps"].values)
        y = g["roe"].values / 100          # 百份比 → 小数
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 60:
            continue

        sl, ic, _, _, _ = sp_stats.linregress(x[ok], y[ok])
        resid = np.full(len(y), np.nan)
        resid[ok] = y[ok] - (sl * x[ok] + ic)

        med   = np.nanmedian(resid)
        mad   = np.nanmedian(np.abs(resid - med)) + 1e-12
        scaled = np.clip((resid - med) / (mad * 1.4826), -MAD_K, MAD_K)
        mu, sd = np.nanmean(scaled), np.nanstd(scaled) + 1e-12
        z = (scaled - mu) / sd

        tmp = g[["stock_code"]].copy()
        tmp["resid_z"] = z
        tmp["report_date"] = q_date
        tmp["eff_date"] = q_date + pd.Timedelta(days=DISCLOSE_DELAY)
        rows.append(tmp)

    dfq = pd.concat(rows, ignore_index=True).rename(
        columns={"resid_z": "q_resid_z"})
    print(f"  {len(dfq):,} 条 | {dfq.stock_code.nunique()} 只")
    return dfq


# ── 4. 日频展平 + 中性化 ─────────────────────────────────────────────────────
def expand_and_neutralize(df_q, df_kl, all_dates):
    print("[4/4] 日频展平 + 中性化 …")
    dates = pd.to_datetime(sorted(all_dates))

    # 20 日成交额均值 (用作中性化代理)
    print(f"  算 {NEUT_WINDOW} 日成交额均值 …")
    df_kl["amt_20d"] = (
        df_kl.groupby("stock_code")["amount"]
        .transform(lambda s: s.rolling(NEUT_WINDOW, min_periods=5).mean())
    )
    df_kl["log_amt"] = np.log1p(df_kl["amt_20d"].clip(lower=1))
    amt_ref = df_kl[["date", "stock_code", "log_amt"]].drop_duplicates(
        ["date", "stock_code"])

    rows_out = []

    for sc, sc_q in df_q.groupby("stock_code"):
        sc_q = sc_q.sort_values("eff_date").reset_index(drop=True)
        if len(sc_q) < 2:
            continue

        # 排名 bisect 找到当前日期对应的季度残差
        eff_list = sc_q["eff_date"].tolist()
        q_z_map  = dict(zip(sc_q["eff_date"], sc_q["q_resid_z"]))

        stock_dates = [d for d in dates if d >= eff_list[0] and d <= dates[-1]]
        if not stock_dates:
            continue

        prev_j = 0            # 当前生效的季度索引
        factor_by_date = {}
        for d in stock_dates:
            while (prev_j + 1 < len(eff_list) and eff_list[prev_j + 1] <= d):
                prev_j += 1
            factor_by_date[d] = q_z_map[eff_list[prev_j]]

        # 合并 = 残差可用的日频面
        dd = pd.DataFrame({"date": list(factor_by_date.keys()),
                           "stock_code": sc,
                           "_raw": list(factor_by_date.values())})
        dd["date"] = pd.to_datetime(dd["date"])

        # 每个有效日单独中性化
        for d, day_df in dd.groupby("date"):
            # 当天截面
            xlot = amt_ref[amt_ref["date"] == d]
            if xlot.empty:
                continue
            merged = day_df.merge(xlot, on="stock_code", how="inner")
            if len(merged) < 30:
                continue

            y    = merged["_raw"].values
            xlog = merged["log_amt"].values
            sl, ic, _, _, _ = sp_stats.linregress(xlog, y)
            resid = y - (sl * xlog + ic)

            med   = np.nanmedian(resid)
            mad   = np.nanmedian(np.abs(resid - med)) + 1e-12
            scaled = np.clip((resid - med) / (mad * 1.4826), -MAD_K, MAD_K)
            mu, sd = np.nanmean(scaled), np.nanstd(scaled) + 1e-12
            z = (scaled - mu) / sd   # 最终因子

            for ik, v in enumerate(z):
                rows_out.append({
                    "date":        d.strftime("%Y-%m-%d"),
                    "stock_code":  merged.iloc[ik]["stock_code"],
                    "factor_value": float(v),
                })

    out = pd.DataFrame(rows_out) if rows_out else pd.DataFrame(
        columns=["date", "stock_code", "factor_value"])
    print(f"  {len(out):,} 行 | {out.stock_code.nunique()} 只 | "
          f"{out.date.nunique()} 日")
    if len(out):
        print(f"  均值={out.factor_value.mean():.4f}  "
              f"std={out.factor_value.std():.4f}  "
              f"min={out.factor_value.min():.2f}  "
              f"max={out.factor_value.max():.2f}")
    return out


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  bps_roe_profitab_diverge_v1 — ROE vs BPS 盈利效率背离")
    print("=" * 55)

    kline = load_kline()
    fund  = load_fund()
    all_dates = sorted(kline.date.unique())

    df_q = quarterly_residual(fund)
    df_out = expand_and_neutralize(df_q, kline, all_dates)

    if df_out.empty:
        print("[警告] 因子为空，退出")
        return

    out_path = Path(OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.sort_values(["date", "stock_code"]).to_csv(out_path, index=False)
    print(f"\n✅  输出 → {out_path}")


if __name__ == "__main__":
    main()
