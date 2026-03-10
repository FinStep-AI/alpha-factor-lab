#!/usr/bin/env python3
"""
Factor: Intraday Return Skewness (intra_skew_v1)
=======================================================
Compute skewness of INTRADAY returns (open→close) over 20 days.
Different from realized_skew (which used close→close returns and failed).

Intraday returns strip out overnight/gap component, giving cleaner signal
about actual trading session dynamics.

Hypothesis: Negative intraday skewness = frequent small gains but occasional 
large intraday drops = hidden selling pressure → continue to underperform.
OR: Positive intraday skew = strong intraday upside → momentum continues.

Test both directions. Neutralize by log(20d avg amount).
"""

import pandas as pd
import numpy as np

def compute_factor():
    kline = pd.read_csv("data/csi1000_kline_raw.csv", parse_dates=["date"])
    kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    # Intraday return: open → close
    kline["intra_ret"] = (kline["close"] - kline["open"]) / kline["open"]
    
    # Rolling 20d skewness of intraday returns
    kline["intra_skew_20d"] = kline.groupby("stock_code")["intra_ret"].transform(
        lambda x: x.rolling(20, min_periods=15).skew()
    )
    
    # Neutralization variable
    kline["avg_amount_20d"] = kline.groupby("stock_code")["amount"].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )
    kline["log_amount_20d"] = np.log(kline["avg_amount_20d"].clip(lower=1))
    
    kline["raw_factor"] = kline["intra_skew_20d"]
    
    def neutralize_cs(group):
        y = group["raw_factor"]
        x = group["log_amount_20d"]
        mask = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 50:
            return pd.Series(np.nan, index=group.index)
        y_m, x_m = y[mask], x[mask]
        med = y_m.median()
        mad = (y_m - med).abs().median() * 1.4826
        if mad < 1e-10:
            return pd.Series(np.nan, index=group.index)
        y_clipped = y_m.clip(med - 3*mad, med + 3*mad)
        x_dm = x_m - x_m.mean()
        denom = (x_dm**2).sum()
        beta = (x_dm * y_clipped).sum() / denom if denom > 0 else 0
        alpha = y_clipped.mean() - beta * x_m.mean()
        residual = y_clipped - (alpha + beta * x_m)
        std = residual.std()
        if std < 1e-10:
            return pd.Series(np.nan, index=group.index)
        z = (residual - residual.mean()) / std
        result = pd.Series(np.nan, index=group.index)
        result[mask] = z
        return result
    
    print("Computing cross-sectional neutralization...")
    kline["factor_value"] = kline.groupby("date").apply(
        neutralize_cs
    ).reset_index(level=0, drop=True)
    
    output = kline[["date", "stock_code", "factor_value"]].dropna()
    output = output.rename(columns={"factor_value": "factor"})
    
    out_path = "data/factor_intra_skew_v1.csv"
    output.to_csv(out_path, index=False)
    print(f"Saved {len(output)} rows to {out_path}")
    print(f"Date range: {output['date'].min()} to {output['date'].max()}")
    print(f"Stocks per date: {output.groupby('date').size().mean():.0f}")
    print(f"\nFactor stats:")
    print(output["factor"].describe())

if __name__ == "__main__":
    compute_factor()
