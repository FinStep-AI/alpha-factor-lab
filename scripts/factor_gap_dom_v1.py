#!/usr/bin/env python3
"""
Overnight Gap Dominance Factor (gap_dom_v1)
=============================================
Concept: proportion of a stock's daily price displacement that occurs at the
         overnight gap (open vs prev close) rather than during continuous trading
         (open vs close intraday).

Per-day measure (pre-smoothing):
    gap_abs  = |open_t / close_{t-1} - 1|
    intra_abs = |close_t / open_t - 1|
    gap_dom_daily = gap_abs / (gap_abs + intra_abs + eps)

The two absolute moves form contiguous price intervals: prev_close → open → close,
so gap_dom_daily ∈ [0, 1] geometrically represents the proportion of the total
|close_t - close_{t-1}| shift that occurs at the boundary between sessions.

After 20-day rolling mean: the stock's average gap_dom over the past month.

Background (academic)
---------------------
Lou, Polk & Skouras (2019) "A Tug of War: Overnight versus Intraday Expected
Returns", Journal of Financial Economics, show that overnight returns are driven
by slow-moving capital; intraday returns by fast-moving liquidity demanders.
If the overnight gap persistently dominates, it suggests sustained slow-moving
(semi-informed) capital commitment at the open.

A-share micro context: the 9:15-9:25 call auction concentrates liquidity;
subsequent retail noise tends to produce intraday reversals on low-gap days.
High gap_ratio stocks = large, price-revealing gaps that set the day's tone;
low gap_ratio stocks = open under-informs, intraday information arrives and
reverses.

Direction: NEGATIVE (low overnight gap dominance → price is intraday driven →
           associated with retail/noise trading → REVERSE, buy HIGH gap_dom)

Neutralization: amount 20d rolling mean OLS (market-cap proxy)
Z-score: YES (cross-sectional)
"""

import os, sys
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(HERE, "..", "data")
OUT_DIR = os.path.join(HERE, "..", "output", "gap_dom_v1")
os.makedirs(OUT_DIR, exist_ok=True)

EPS    = 1e-9
WIN    = 20        # rolling mean window for gap_dom
LAG_PC = 1         # one-bar lagged close

def load_data():
    df = pd.read_csv(os.path.join(DATA, "csi1000_kline_raw.csv"),
                     dtype={"stock_code": str})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    return df


def compute_gap_dom(df: pd.DataFrame) -> pd.DataFrame:
    """Compute |gap|/(|gap|+|intraday|) per stock per day, then 20d rolling mean."""

    # lagged close within each stock
    df["close_lag"] = df.groupby("stock_code")["close"].shift(LAG_PC)

    # absolute overnight gap: |open/prev_close - 1|
    df["gap_abs"] = ((df["open"] / df["close_lag"]) - 1.0).abs()

    # absolute intraday return: |close/open - 1|
    df["intra_abs"] = ((df["close"] / df["open"]) - 1.0).abs()

    # gap dominance per day  (constrained to [0, 1])
    df["gap_dom_daily"] = df["gap_abs"] / (df["gap_abs"] + df["intra_abs"] + EPS)
    df["gap_dom_daily"] = df["gap_dom_daily"].clip(0.0, 1.0)

    # 20d rolling mean
    df["gap_dom"] = (
        df.groupby("stock_code")["gap_dom_daily"]
          .transform(lambda s: s.rolling(WIN, min_periods=max(WIN // 2, 5))
                               .mean())
    )
    return df


def neutralize_cross_section(df: pd.DataFrame, fac_col: str, neu_col: str) -> pd.Series:
    """OLS residuals per day: fac ~ log(neu_col)."""
    def _ols(group):
        y = group[fac_col].values.astype(float)
        x = np.log(group[neu_col].clip(lower=1).values.astype(float) + 1)
        m = np.isfinite(y) & np.isfinite(x)
        out = np.full(len(y), np.nan)
        if m.sum() < 30:
            return pd.Series(out, index=group.index)
        X = np.column_stack([np.ones(m.sum()), x[m]])
        try:
            b = np.linalg.lstsq(X, y[m], rcond=None)[0]
            out[m] = y[m] - X @ b
        except Exception:
            out[m] = y[m] - np.nanmean(y[m])
        return pd.Series(out, index=group.index)

    return df.groupby("date", group_keys=False).apply(_ols)


def zscore_cross_section(df: pd.DataFrame, col: str) -> pd.Series:
    """Rank → z-score per day."""
    def _zs(group):
        v = group[col].values.astype(float)
        m = np.isfinite(v)
        out = np.full(len(v), np.nan)
        if m.sum() < 10:
            return pd.Series(out, index=group.index)
        from scipy.stats import rankdata
        ranks = rankdata(v, method="average")          # 1..n, ties averaged
        n = m.sum()
        cdf = ranks / (n + 1)
        from scipy.stats import norm
        out[:] = norm.ppf(cdf)
        return pd.Series(out, index=group.index)
    return df.groupby("date", group_keys=False).apply(_zs)


def main():
    print("=" * 60)
    print("Overnight Gap Dominance Factor (gap_dom_v1)")
    print("  gap_dom = |gap| / (|gap| + |intraday|),  20d rolling mean")
    print("  Direction: NEGATIVE  (buy LOW gap_dom)")
    print("=" * 60)

    df = load_data()
    print(f"Loaded {len(df):,} rows  |  {df['stock_code'].nunique()} stocks  "
          f"|  {df['date'].min().date()} → {df['date'].max().date()}")

    df = compute_gap_dom(df)

    # neutralize by amount-20d-mean proxy
    df["amount_ma20"] = (
        df.groupby("stock_code")["amount"]
          .transform(lambda s: s.rolling(WIN, min_periods=WIN // 2).mean())
    )

    # full row count before neutralize
    before = len(df)
    df["factor"] = neutralize_cross_section(df, "gap_dom", "amount_ma20")

    # transform: take NEGATIVE direction (buy low gap-dominance → high factor value)
    df["factor"] = -df["factor"]

    # z-score
    df["factor"] = zscore_cross_section(df, "factor")

    out = (
        df[["date", "stock_code", "factor"]]
        .dropna(subset=["factor"])
        .copy()
    )
    out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
    out["date"]      = out["date"].dt.strftime("%Y-%m-%d")

    path = os.path.join(DATA, "factor_gap_dom_v1.csv")
    out.to_csv(path, index=False)
    print(f"\nFactor saved → {path}")
    print(f"  rows={len(out):,}  stocks={out['stock_code'].nunique()}  "
          f"dates={out['date'].nunique()}")
    print(f"  mean={out['factor'].mean():.4f}  std={out['factor'].std():.4f}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
