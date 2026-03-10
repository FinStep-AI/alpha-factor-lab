#!/usr/bin/env python3
"""
Factor: High-Volume Close Strength (hv_close_strength_v1)
=============================================================
CLV (close location value) measured ONLY on high-volume days.

On days with above-median volume (past 20d), where does the stock close
in its daily range? Consistently closing near the high on heavy volume = 
strong institutional demand signal.

This is a refinement of close_location_v1 (IC=0.018 t=1.37 not significant)
by conditioning on volume.

Neutralize: log(20d avg amount).
Direction: Positive (high value = closes near high on volume days = bullish).
"""

import pandas as pd
import numpy as np

def compute_factor():
    kline = pd.read_csv("data/csi1000_kline_raw.csv", parse_dates=["date"])
    kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    # CLV = (close - low) / (high - low)
    range_hl = kline["high"] - kline["low"]
    kline["clv"] = np.where(range_hl > 0, (kline["close"] - kline["low"]) / range_hl, 0.5)
    
    # Rolling 20d median volume
    kline["vol_median_20d"] = kline.groupby("stock_code")["volume"].transform(
        lambda x: x.rolling(20, min_periods=15).median()
    )
    
    # High volume flag (above rolling median)
    kline["is_high_vol"] = (kline["volume"] > kline["vol_median_20d"]).astype(float)
    
    # CLV only on high-volume days, 0 otherwise
    kline["clv_hv"] = kline["clv"] * kline["is_high_vol"]
    
    # Rolling mean of CLV on high-vol days = sum(clv_hv) / sum(is_high_vol)
    kline["sum_clv_hv_20d"] = kline.groupby("stock_code")["clv_hv"].transform(
        lambda x: x.rolling(20, min_periods=15).sum()
    )
    kline["count_hv_20d"] = kline.groupby("stock_code")["is_high_vol"].transform(
        lambda x: x.rolling(20, min_periods=15).sum()
    )
    
    kline["raw_factor"] = kline["sum_clv_hv_20d"] / kline["count_hv_20d"].clip(lower=1)
    
    # Neutralize
    kline["avg_amount_20d"] = kline.groupby("stock_code")["amount"].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )
    kline["log_amount_20d"] = np.log(kline["avg_amount_20d"].clip(lower=1))
    
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
    
    print("Computing neutralization...")
    kline["factor_value"] = kline.groupby("date").apply(
        neutralize_cs
    ).reset_index(level=0, drop=True)
    
    output = kline[["date", "stock_code", "factor_value"]].dropna()
    output = output.rename(columns={"factor_value": "factor"})
    
    out_path = "data/factor_hv_close_strength_v1.csv"
    output.to_csv(out_path, index=False)
    print(f"Saved {len(output)} rows to {out_path}")
    print(f"Date range: {output['date'].min()} to {output['date'].max()}")
    print(f"Stocks per date: {output.groupby('date').size().mean():.0f}")
    print(f"\nFactor stats:")
    print(output["factor"].describe())

if __name__ == "__main__":
    compute_factor()
