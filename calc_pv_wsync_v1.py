#!/usr/bin/env python3
"""
因子 pv_wsync_v1 — 量价信号持续自相关
Volume-Weighted Signed Return 60d rolling lag-1 autocorrelation.

理论来源:
  Chordia, Roll & Subrahmanyam (2001), JOF, "Market Liquidity and Trading Activity"
  Llorente et al. (2002), JFQA, "Dynamic volume-return relation of individual stocks"
  北大光华 – 中国股票市场的信息传导与流动性需求

构造:
  每日有方向量能信号
    svret_t = sign(ret_{t-1}) * ln(1 + turnover_t / MA20(turnover))
  60日前视滚动窗口的 lag-1 自相关 (min_periods=40)
  对截面 ln(amount) OLS 中性化 → MAD 缩尾 → z-score

输出: data/factor_pv_wsync_v1.csv
"""

import numpy as np
import pandas as pd
import sys
from numba import njit

DATA_DIR = "data"
OUT = f"{DATA_DIR}/factor_pv_wsync_v1.csv"

LOOKBACK = 60          # 滚动窗口
MA_TO_WIN = 20          # 换手率均值窗口
MIN_PERIODS = max(LOOKBACK // 3 * 2, 40)   # 最少有效样本


def lag1_autocorr(s: np.ndarray, min_p: int) -> float:
    """快速 lag-1 ACF, 输入为一维 float64 array。"""
    n = s.shape[0]
    if n < min_p:
        return np.nan
    x = s   # current
    y = np.empty(n)
    y[0] = np.nan
    y[1:] = s[:-1]
    mask = np.isfinite(x) & np.isfinite(y)
    k = mask.sum()
    if k < min_p:
        return np.nan
    xm = x[mask] - np.nanmean(x[mask])
    ym = y[mask] - np.nanmean(y[mask])
    denom = np.sqrt(np.sum(xm * xm) * np.sum(ym * ym))
    if denom == 0:
        return np.nan
    return float(np.sum(xm * ym) / denom)


def compute_factor(df: pd.DataFrame) -> pd.DataFrame:
    print(f"输入: {len(df)} 行, {df['stock_code'].nunique()} 股票, "
          f"{df['date'].nunique()} 个交易日")

    df = df.sort_values(["stock_code", "date"]).copy()
    results = []

    for scode, g in df.groupby("stock_code", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        if n < LOOKBACK + 5:
            continue

        date_arr = g["date"].values.astype(str)
        amt_arr  = g["amount"].values.astype(np.float64)
        turn_arr = g["turnover"].values.astype(np.float64)

        # prev-day close return: shift(close).pct_change()
        prev_ret = np.empty(n, dtype=np.float64)
        prev_ret[:] = np.nan
        if "close" in g.columns:
            close = g["close"].values.astype(np.float64)
            prev_close = np.empty(n)
            prev_close[:] = np.nan
            prev_close[1:] = close[:-1]
            np.divide(close - prev_close, prev_close,
                      out=prev_ret, where=np.isfinite(prev_close))

        # MAP 20-day mean turnover
        ma20 = np.empty(n)
        ma20[:] = np.nan
        rv = np.copy(turn_arr)
        csum = 0.0
        cnt = 0
        buf = []
        for i in range(n):
            v = rv[i]
            if np.isfinite(v):
                buf.append(v)
                if len(buf) > MA_TO_WIN:
                    buf.pop(0)
                ma20[i] = np.mean(buf)
            else:
                ma20[i] = np.nan

        # signed volume signal: sign(prev_ret) * ln(1 + to/MA20)
        sv = np.empty(n, dtype=np.float64)
        sv[:] = np.nan
        for i in range(n):
            if not (np.isfinite(prev_ret[i]) and np.isfinite(turn_arr[i])
                    and np.isfinite(ma20[i]) and ma20[i] > 0):
                continue
            ratio = turn_arr[i] / ma20[i]
            sv[i] = np.sign(prev_ret[i]) * np.log1p(ratio)

        # rolling lag-1 ACF
        raw = np.empty(n, dtype=np.float64)
        raw[:] = np.nan
        # warm up with cumulative
        for i in range(LOOKBACK - 1, n):
            sl = sv[i - LOOKBACK + 1: i + 1]
            raw[i] = lag1_autocorr(sl, MIN_PERIODS)

        raw[pd.isna(g["amount"])] = np.nan

        block = pd.DataFrame({
            "date": date_arr,
            "stock_code": np.full(n, scode),
            "_raw": raw,
            "_amt": amt_arr,
        })
        results.append(block)

    big = pd.concat(results, ignore_index=True)

    # Section cross-sectional: OLS residual on log_amount
    big["_amt_log"] = np.log(big["_amt"].clip(lower=1))
    final_vals = np.empty(len(big))
    for dt, idx in big.groupby("date").groups.items():
        rows = big.loc[idx]
        y = rows["_raw"].values.astype(float)
        xcol = rows["_amt_log"].values.astype(float)
        mask = np.isfinite(y) & np.isfinite(xcol)
        if mask.sum() < 10:
            final_vals[idx] = np.nan
            continue
        X = np.column_stack([np.ones(mask.sum()), xcol[mask]])
        beta = np.linalg.lstsq(X, y[mask], rcond=None)[0]
        res = np.full(len(y), np.nan)
        res[mask] = y[mask] - X @ beta
        res = _winsorize_mad_1d(res)
        res = _zscore_1d(res)
        final_vals[idx] = res

    out = pd.DataFrame({
        "date": big["date"],
        "stock_code": big["stock_code"].astype(int),
        "factor_pv_wsync_v1": final_vals,
    })
    return out.dropna(subset=["factor_pv_wsync_v1"])


@njit
def _mad_1d(x):
    # not used directly — kept as doc
    return 0.0


def _winsorize_mad_1d(arr: np.ndarray, n_mad: float = 3.0) -> np.ndarray:
    valid = arr[np.isfinite(arr)]
    if len(valid) < 5:
        return arr
    med = np.median(valid)
    mad = np.median(np.abs(valid - med)) * 1.4826
    if mad == 0 or np.isnan(mad):
        return arr
    lo = med - n_mad * mad
    hi = med + n_mad * mad
    return np.clip(arr, lo, hi)


def _zscore_1d(arr: np.ndarray) -> np.ndarray:
    valid = arr[np.isfinite(arr)]
    mu, sd = valid.mean(), valid.std()
    if sd == 0 or np.isnan(sd):
        return arr
    out = arr.copy()
    m = np.isfinite(arr)
    out[m] = (arr[m] - mu) / sd
    return out


if __name__ == "__main__":
    raw_path = f"{DATA_DIR}/csi1000_kline_raw.csv"
    print(f"读取: {raw_path}")
    df = pd.read_csv(raw_path)
    df["date"] = pd.to_datetime(df["date"])
    out = compute_factor(df)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\n输出: {OUT}  |  {len(out)} 行")
    fv = out["factor_pv_wsync_v1"]
    print(f"  valid={fv.notna().sum()}  mean={fv.mean():.4f}  std={fv.std():.4f}"
          f"  [{fv.min():.2f}, {fv.max():.2f}]")
