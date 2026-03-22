#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Price Delay Factor (Hou & Moskowitz, 2005 RFS) — Vectorized

Delay = 1 - R²_restricted / R²_unrestricted
- R²_restricted: regress stock_ret on mkt_ret (contemporaneous only)
- R²_unrestricted: regress stock_ret on mkt_ret + 4 lags

Uses vectorized rolling regression (no per-stock loop).
"""

import numpy as np
import pandas as pd
from pathlib import Path

def rolling_r2(y_arr, X_arr, window, min_obs):
    """
    Compute rolling OLS R² for arrays.
    y_arr: (T,) target
    X_arr: (T, k) regressors (including intercept)
    Returns: (T,) array of R² (NaN where insufficient data)
    """
    T, k = X_arr.shape
    r2 = np.full(T, np.nan)
    
    for i in range(window - 1, T):
        start = i - window + 1
        y_w = y_arr[start:i+1]
        X_w = X_arr[start:i+1]
        
        mask = np.all(np.isfinite(X_w), axis=1) & np.isfinite(y_w)
        if mask.sum() < min_obs:
            continue
        
        y_m = y_w[mask]
        X_m = X_w[mask]
        
        ss_tot = np.sum((y_m - y_m.mean()) ** 2)
        if ss_tot < 1e-15:
            continue
        
        try:
            beta = np.linalg.lstsq(X_m, y_m, rcond=None)[0]
            resid = y_m - X_m @ beta
            ss_res = np.sum(resid ** 2)
            r2[i] = 1.0 - ss_res / ss_tot
        except Exception:
            continue
    
    return r2

def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    
    print("Loading kline data...")
    df = pd.read_csv(data_dir / "csi1000_kline_raw.csv", parse_dates=["date"])
    df = df.sort_values(["date", "stock_code"]).reset_index(drop=True)
    
    # Daily returns
    df = df.sort_values(["stock_code", "date"])
    df["ret"] = df.groupby("stock_code")["close"].pct_change()
    
    # Equal-weighted market return
    mkt = df.groupby("date")["ret"].mean().rename("mkt_ret").reset_index()
    
    # Add lagged market returns
    mkt = mkt.sort_values("date").reset_index(drop=True)
    for lag in range(1, 5):
        mkt[f"mkt_lag{lag}"] = mkt["mkt_ret"].shift(lag)
    
    df = df.merge(mkt, on="date", how="left")
    
    # Parameters
    window = 60
    min_obs = 40
    
    print(f"Computing Price Delay per stock (window={window})...")
    
    stocks = df["stock_code"].unique()
    total = len(stocks)
    results = []
    
    for idx, stock in enumerate(stocks):
        if (idx + 1) % 200 == 0:
            print(f"  {idx+1}/{total}...")
        
        sdf = df[df["stock_code"] == stock].sort_values("date").reset_index(drop=True)
        n = len(sdf)
        if n < window:
            continue
        
        y = sdf["ret"].values
        ones = np.ones(n)
        mkt_r = sdf["mkt_ret"].values
        lags = [sdf[f"mkt_lag{l}"].values for l in range(1, 5)]
        
        # Restricted: [1, mkt_ret]
        X_r = np.column_stack([ones, mkt_r])
        # Unrestricted: [1, mkt_ret, lag1, lag2, lag3, lag4]
        X_u = np.column_stack([ones, mkt_r] + lags)
        
        r2_r = rolling_r2(y, X_r, window, min_obs)
        r2_u = rolling_r2(y, X_u, window, min_obs)
        
        for i in range(n):
            if np.isnan(r2_r[i]) or np.isnan(r2_u[i]):
                continue
            if r2_u[i] < 1e-10:
                delay = 0.0
            else:
                delay = 1.0 - r2_r[i] / r2_u[i]
            delay = np.clip(delay, 0.0, 1.0)
            
            results.append({
                "date": sdf.iloc[i]["date"],
                "stock_code": stock,
                "factor_value": delay,
            })
    
    print(f"  Raw observations: {len(results)}")
    factor_df = pd.DataFrame(results)
    
    # Merge log_amount_20d for neutralization
    print("Neutralizing by log_amount_20d...")
    df_sorted = df.sort_values(["stock_code", "date"])
    df_sorted["log_amount_20d"] = np.log(
        df_sorted.groupby("stock_code")["amount"].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        ) + 1
    )
    amt_lk = df_sorted[["date", "stock_code", "log_amount_20d"]].drop_duplicates(["date", "stock_code"])
    factor_df = factor_df.merge(amt_lk, on=["date", "stock_code"], how="left")
    
    def neutralize_and_normalize(grp):
        y = grp["factor_value"].values.copy()
        x = grp["log_amount_20d"].values
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            grp["factor_value"] = np.nan
            return grp
        
        ym, xm = y[mask], x[mask]
        X = np.column_stack([np.ones(len(xm)), xm])
        try:
            beta = np.linalg.lstsq(X, ym, rcond=None)[0]
            resid = ym - X @ beta
        except Exception:
            grp["factor_value"] = np.nan
            return grp
        
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        if mad < 1e-10:
            grp["factor_value"] = np.nan
            return grp
        z = (resid - med) / (1.4826 * mad)
        z = np.clip(z, -3, 3)
        mu, sigma = z.mean(), z.std()
        if sigma < 1e-10:
            grp["factor_value"] = np.nan
            return grp
        z = (z - mu) / sigma
        
        out = np.full(len(y), np.nan)
        out[mask] = z
        grp["factor_value"] = out
        return grp
    
    factor_df = factor_df.groupby("date", group_keys=False).apply(neutralize_and_normalize)
    
    out = factor_df[["date", "stock_code", "factor_value"]].dropna()
    out_path = data_dir / "factor_price_delay_v1.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"  Dates: {out['date'].min()} ~ {out['date'].max()}")
    print(f"  Stocks/date: ~{out.groupby('date')['stock_code'].count().median():.0f}")
    print(f"  Total rows: {len(out)}")

if __name__ == "__main__":
    main()
