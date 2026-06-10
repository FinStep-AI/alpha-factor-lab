"""
勇攀高峰因子 v1 — 日线近似版
方正证券 2022-05-30《个股波动率的变动及"勇攀高峰"因子构建——多因子选股系列研究之三》

原文用分钟级 OHLC（5根→20个价 → 当期改良波动率），受限于日线数据无法完全复现。
我们用近5个交易日的 OHLC（= 20个价）做等权近似，期望捕获同方向信号。

构造逻辑：
  1. 日线改良波动率 vol_t  = std(OHLC过去5根) / mean(OHLC过去5根)
  2. 收益波动比      rw_t   = |ret_t| / max(vol_t, eps)
  3. 过去 20 个交易日，求 rw 序列与 vol 序列的逐日协方差 → cov_t
     （与原文更接近：每月末用20天协方差均/当时20天协方差标准差）
  4. 月均攀登 = 最近20天 cov 均值（高＝异常高波动时段收益能趁机上涨）
  5. 月稳攀登 = 最近20天 cov 的 z ＝ (cov_t - μ20) / σ20
  6. 勇攀高峰因子 = 0.60 × 月均攀登 + 0.40 × 月稳攀登
  7. 截面对数市值 OLS 中性化，MAD 缩尾，z-score

输出: data/factor_climb_peak_v1_neutral.csv
"""

import sys, os, json
import numpy as np
import pandas as pd
from pathlib import Path

LABEL = "climb_peak_v1"
BASE  = Path(__file__).resolve().parent.parent
DATA  = BASE / "data"
OUT   = DATA / f"factor_{LABEL}.csv"


def load_kline():
    df = pd.read_csv(DATA / "csi1000_kline_raw.csv",
                     parse_dates=["date"], dtype={"stock_code": str})
    df["stock_code"] = df["stock_code"].str.zfill(6)
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    # ret
    df["ret"] = df.groupby("stock_code")["close"].pct_change()
    return df


def improved_vol(row_group_ohlc: np.ndarray) -> float:
    """改良波动率 = std(20个价) / mean(20个价)。"""
    vals = row_group_ohlc.ravel()
    mu, sig = vals.mean(), vals.std(ddof=0)
    if mu == 0:
        return np.nan
    return sig / mu


def build_factor(df: pd.DataFrame, win_cov: int = 20, win_vol: int = 5):
    """每只股票单独并行（pandas apply per group）。"""

    records = []

    for code, g in df.groupby("stock_code", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        if n < win_vol + 2:
            continue

        ohlc = g[["open", "high", "low", "close"]].values  # (T, 4)

        # 1. 改良波动率：过去 win_vol 根K线的 OHLC 均值/std
        vol_roll = pd.Series(np.nan, index=g.index)
        for i in range(win_vol - 1, n):
            vol_roll.iloc[i] = improved_vol(ohlc[i - win_vol + 1: i + 1])

        # 2. 收益波动比
        rw = g["ret"].abs() / vol_roll.clip(lower=1e-12)

        # 3. 滚动 win_cov 日：rw 与 vol 的逐日协方差
        cov = pd.Series(np.nan, index=g.index)
        for i in range(win_cov - 1, n):
            x = rw.iloc[i - win_cov + 1: i + 1]
            y = vol_roll.iloc[i - win_cov + 1: i + 1]
            if x.notna().sum() < win_cov and y.notna().sum() < win_cov:
                continue
            cov.iloc[i] = x.cov(y)

        cov = cov.fillna(0.0)

        # 4-5. 再用 win_cov 窗口聚合月均攀登 + 月稳攀登
        avg_peak = pd.Series(np.nan, index=g.index)
        sta_peak = pd.Series(np.nan, index=g.index)

        for i in range(win_cov - 1, n):
            w = cov.iloc[i - win_cov + 1: i + 1]
            mu20, sig20 = w.mean(), w.std(ddof=0)
            avg_peak.iloc[i] = mu20
            sta_peak.iloc[i] = (w.iloc[-1] - mu20) / sig20 if sig20 > 0 else 0.0

        # 6. 合成
        raw = 0.60 * avg_peak.fillna(0) + 0.40 * sta_peak.fillna(0)
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0)

        tmp = pd.DataFrame({
            "date":  g["date"].values,
            "stock_code": code,
            "factor_raw": raw.values,
        })
        records.append(tmp)

    return pd.concat(records, ignore_index=True)


def neutralize_market_cap(raw_df: pd.DataFrame, kline_df: pd.DataFrame) -> pd.Series:
    """对数成交额 OLS 中性化 + MAD 缩尾 + z-score。"""
    mkt = kline_df[["date", "stock_code", "amount"]].copy()
    mkt["stock_code"] = mkt["stock_code"].str.zfill(6)
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt["ln_amount"] = np.log(mkt["amount"].clip(lower=1))

    raw_df = raw_df.copy()
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    out = []
    for dt, g in raw_df.groupby("date"):
        g = g.merge(mkt[mkt["date"] == dt][["stock_code", "ln_amount"]],
                    on="stock_code", how="left")
        valid = g.dropna(subset=["factor_raw", "ln_amount"])
        if len(valid) < 30:
            out.append(pd.DataFrame({"date": g["date"], "stock_code": g["stock_code"],
                                      "factor": np.nan}))
            continue

        X = np.column_stack([np.ones(len(valid)), valid["ln_amount"].values])
        y = valid["factor_raw"].values
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            med, iqr = np.median(resid), np.percentile(resid, 75) - np.percentile(resid, 25)
            lo, hi = med - 5 * iqr, med + 5 * iqr
            resid = np.clip(resid, lo, hi)
            mu, s = resid.mean(), resid.std()
            z = (resid - mu) / s if s > 0 else resid * 0
        except Exception:
            z = np.zeros(len(valid))

        out.append(pd.DataFrame({
            "date": g["date"].values[:len(valid)],
            "stock_code": valid["stock_code"].values,
            "factor": z,
        }))

    result = pd.concat(out, ignore_index=True)
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    result = result.sort_values(["date", "stock_code"]).reset_index(drop=True)
    return result


def main():
    print(f"[{LABEL}] loading kline …")
    df = load_kline()
    print(f"[{LABEL}] kline shape {df.shape}, date range "
          f"{df['date'].min().date()} ~ {df['date'].max().date()}")

    print(f"[{LABEL}] building raw factor …")
    raw = build_factor(df)
    print(f"[{LABEL}] raw rows: {len(raw)}")

    print(f"[{LABEL}] neutralizing …")
    out = neutralize_market_cap(raw, df)
    print(f"[{LABEL}] neutralized rows: {len(out)}")

    out.to_csv(OUT, index=False)
    print(f"[{LABEL}] saved → {OUT}")
    print(out.groupby("date")["factor"].count().tail(5))


if __name__ == "__main__":
    main()
