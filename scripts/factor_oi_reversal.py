"""
Overnight-Intraday Reversal v1
================================
SSRN 2730304 — "Overnight-Intraday Reversal Everywhere"
( motivated by Jiayan thesis + Lou et al. overnight-intraday decomposition )

Construction
------------
Factor = sum_{t=1..20} [ sign(overnight_ret_t) × intraday_ret_t ]
         ─────────────────────────────────────────────────────────
                mean_{t=1..20} |intraday_ret_t|

Intuition
---------
overnight_ret  = close_{t-1} / open_t  - 1   （隔夜信息驱动）
intraday_ret   = close_t  / open_t  - 1         （日内交易驱动）

High factor = 隔夜涨 → 日内跌 的频率/幅度高 → Overnight-Intraday 反转效应强
Low  factor = 隔夜涨 → 日内也涨  → OIP 是纯粹的动量延续

Direction
---------
正向使用：高因子 = 信息释放后日内走得越偏反向 → 反转延续 → 次日正向收益
（类似 close_low_v1：日内消化完隔夜利好后，卖压释放完毕，次日反弹）
"""

import numpy as np
import pandas as pd
import sys, os, warnings
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────
LABEL = "oi_reversal_v1"
WORK  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KLINE = os.path.join(WORK, "data", "csi1000_kline_raw.csv")
OUT   = os.path.join(WORK, "data", f"factor_{LABEL}.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ── load ───────────────────────────────────────────────────────────
df = pd.read_csv(KLINE, parse_dates=["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

# ── compute overnight & intraday returns ───────────────────────────
grp = df.groupby("stock_code", sort=False)

df["open_prev"] = grp["open"].shift(1)
df["close_prev"] = grp["close"].shift(1)

df["overnight_ret"] = df["open"] / df["close_prev"] - 1   # close_{t-1} → open_t
df["intraday_ret"]  = df["close"] / df["open"] - 1        # open_t → close_t

# winsorize extreme gaps caused by missing prev_close
eps = 1e-6
for col in ["overnight_ret", "intraday_ret"]:
    upper = df[col].quantile(0.995)
    lower = df[col].quantile(0.005)
    df[col] = df[col].clip(lower, upper)

print(f"Dates available: {df['date'].min()} → {df['date'].max()}")
print(f"Stocks: {df['stock_code'].nunique()}")

# ── rolling 20-day factor ──────────────────────────────────────────
W = 20

def roll_oi_reversal(g):
    """Compute OI-reversal for one stock."""
    on  = g["overnight_ret"].values
    inr = g["intraday_ret"].values

    # signed product: sign(overnight) * intraday
    signed = np.sign(on) * inr
    # running 20d sum
    roll_sum = pd.Series(signed, index=g.index).rolling(W, min_periods=W).sum().values
    # running 20d mean |intraday|
    roll_ampd = pd.Series(np.abs(inr), index=g.index).rolling(W, min_periods=W).mean().values

    factor_raw = np.where(roll_ampd > eps, roll_sum / roll_ampd, np.nan)
    return pd.Series(factor_raw, index=g.index)

print("Computing rolling O/I reversal…")
df["factor_raw"] = grp.apply(roll_oi_reversal).reset_index(level=0, drop=True)
df = df.dropna(subset=["factor_raw"]).copy()

# ── neutralization: OLS on log_amount_20d ──────────────────────────
df["log_amount_20d"] = grp["amount"].transform(lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1))
df = df.dropna(subset=["log_amount_20d"]).copy()

def neutralize_cross_section(series, control):
    """OLS neutralize series on control, return residual z-score."""
    out = pd.Series(np.nan, index=series.index)
    for dt, idx in series.groupby(series.index).groups.items():
        if len(idx) < 30:
            continue
        y = series.loc[idx].values
        x = control.loc[idx].values
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            continue
        x_ = np.column_stack([np.ones(mask.sum()), x[mask]])
        try:
            beta, _, _, _ = np.linalg.lstsq(x_, y[mask], rcond=None)
            resid = np.full(mask.sum(), np.nan)
            resid[mask] = y[mask] - x_ @ beta
            mu = np.nanmean(resid); sd = np.nanstd(resid)
            if pd.notna(sd) and sd > eps:
                out.loc[idx[mask]] = (resid - mu) / sd
        except Exception:
            pass
    return out

print("Cross-sectional neutralization…")
df["factor_neutral"] = df.groupby("date", group_keys=False).apply(
    lambda x: neutralize_cross_section(x["factor_raw"], x["log_amount_20d"])
).reset_index(level=0, drop=True)

# ── winsorize ±3σ then re-zscore ───────────────────────────────────
def winsorize_zscore(series, n_sig=3.0):
    lo = series.quantile(0.01)
    hi = series.quantile(0.99)
    s  = series.clip(lo, hi)
    mu, sd = s.mean(), s.std()
    if pd.notna(sd) and sd > eps:
        return (s - mu) / sd
    return s

df["factor_final"] = df.groupby("date")["factor_neutral"].transform(winsorize_zscore)

# ── save ───────────────────────────────────────────────────────────
out = df[["date","stock_code","factor_final"]].dropna(subset=["factor_final"])
out = out.rename(columns={"factor_final": "factor_value"})
out.to_csv(OUT, index=False)
print(f"Saved {len(out)} rows → {OUT}")
print(f"Date range: {out['date'].min()} → {out['date'].max()}")
print(f"\nFactor stats:")
print(out["factor_value"].describe())
