#!/usr/bin/env python3
"""
因子：日内收益率偏度v1 (Intraday Return Skewness) - 向量化版本

逻辑：每只股票在过去 20 个交易日内 intraday_ret 的滚动偏度 → 取当日截面值
  然后对 log_amount_20d 截面OLS中性化 + MAD winsorize + z-score

输出: data/factor_intraday_ret_skew_v1.csv
"""
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
import warnings
warnings.filterwarnings("ignore")

KLINE = "data/csi1000_kline_raw.csv"
ROLLING_WINDOW = 20
WINSORIZE_N = 5.0
OUTPUT = "data/factor_intraday_ret_skew_v1.csv"

def main():
    print("[INFO] 读取K线数据...")
    df = pd.read_csv(KLINE, parse_dates=["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    # prev_close + intraday_ret
    df["prev_close"] = df.groupby("stock_code")["close"].shift(1)
    df["intraday_ret"] = (df["close"] - df["open"]) / df["prev_close"]

    # per-stock rolling skewness (20d width window)
    def _skew20(x):
        s = pd.Series(x)
        return s.rolling(ROLLING_WINDOW, min_periods=6).skew()
        # pandas rolling skewness ⬆️

    print("[INFO] 滚动偏度计算（向量化）...")
    df["factor_raw"] = df.groupby("stock_code")["intraday_ret"].transform(
        lambda s: _skew20(s).values
    )

    # log_amount_20d 做中性化目标
    df["log_amount_20d"] = df.groupby("stock_code")["amount"].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )

    # 截面OLS中性化（每日截面做 y ~ log_amount_20d）
    cross_infos = df.groupby("date", sort=False)
    valid_rows = df.dropna(subset=["factor_raw", "log_amount_20d"]).copy()
    print(f"[INFO] 原始因子值: {len(valid_rows)} 行")

    def _ols_neutralize(grp):
        codes   = grp["stock_code"].values
        y       = grp["factor_raw"].values
        x       = grp["log_amount_20d"].values
        mask    = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 10:
            return pd.Series(np.nan, index=grp.index)
        X = np.column_stack([np.ones(mask.sum()), x[mask]])
        idx = grp.index
        resid = pd.Series(np.nan, index=idx)
        _, ss, _, _ = lstsq(X, y[mask], rcond=None)
        for k, m in zip(np.where(mask)[0], (y[mask] - X @ (_, ss, _, _)[0])):
            resid.iloc[k] = m
        # Compact: use numpy directly
        # re-calculate (cleaner)
        X2 = np.column_stack([np.ones(valid.sum()), x[valid]])
        beta, _, _, _ = lstsq(X2, y[valid], rcond=None)
        out = np.full(len(grp), np.nan)
        out_valid = y[valid] - X2 @ beta
        out[valid] = out_valid
        return pd.Series(out, index=grp.index)
        # Actually let's use a simpler closed form:
        # y_centered = y - mean(y)
        # x_centered = x - mean(x)
        # beta = sum(xc*yc) / (sum(xc^2)+eps)
        # resid = y_centered - beta * x_centered
        yc = y - np.nanmean(y)
        xc = x - np.nanmean(x)
        denom = np.nansum(xc**2)
        if denom < 1e-10:
            return pd.Series(0.0, index=grp.index)
        beta = np.nansum(xc * yc) / denom
        resid = yc - beta * xc
        return pd.Series(resid, index=grp.index)

    print("[INFO] 截面OLS中性化...")
    df["factor_neutralized"] = np.nan
    
    for dt, grp in cross_infos:
        codes = grp["stock_code"].values
        y = grp["factor_raw"].values.astype(float)
        x = grp["log_amount_20d"].values.astype(float)
        valid = np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 20:
            continue
        yc = y - np.mean(y[valid])
        xc = x - np.mean(x[valid])
        denom = np.sum(xc[valid]**2)
        if denom < 1e-10:
            df.loc[grp.index[valid], "factor_neutralized"] = 0.0
            continue
        beta = np.sum(xc[valid] * yc[valid]) / denom
        resid_full = np.full(len(grp), np.nan)
        resid_full[valid] = yc[valid] - beta * xc[valid]
        df.loc[grp.index, "factor_neutralized"] = resid_full

    print(f"[INFO] 中性化后有效数: {df['factor_neutralized'].notna().sum()}")

    # MAD winsorize + z-score (per cross-section)
    print("[INFO] MAD winsorize + z-score...")

    def _mad_zscore(arr):
        v = arr[~np.isnan(arr)]
        if len(v) < 20:
            return np.full(len(arr), np.nan)
        med = np.median(v)
        mad = np.median(np.abs(v - med)) * 1.4826
        if mad < 1e-10:
            return np.zeros(len(arr))
        hi, lo = med + WINSORIZE_N*mad, med - WINSORIZE_N*mad
        v2 = np.clip(v, lo, hi)
        m2, s2 = v2.mean(), v2.std()
        if s2 < 1e-10:
            return np.zeros(len(arr))
        result = np.full(len(arr), np.nan)
        result[~np.isnan(arr)] = (v2 - m2) / s2
        return result

    dt_groups = df.groupby("date", sort=False)
    fn_series = []
    for dt, grp in dt_groups:
        vals = grp["factor_neutralized"].values
        fn_series.append(pd.Series(_mad_zscore(vals), index=grp.index))

    # 展平回df
    all_fn = pd.concat(fn_series).sort_index()
    df["factor_value"] = all_fn[df.index].values

    out = df[["date", "stock_code", "factor_value"]].dropna()
    out.to_csv(OUTPUT, index=False)
    print(f"[DONE] 保存 {OUTPUT}")
    print(f"  有效记录: {len(out)} | 股票数: {out['stock_code'].nunique()}")
    print(f"  日期跨度: {out['date'].min().date()} ~ {out['date'].max().date()}")


if __name__ == "__main__":
    main()
