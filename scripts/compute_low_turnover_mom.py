#!/usr/bin/env python3
"""
Factor: Low-Turnover Momentum (low_turnover_mom_v1)
=======================================================
Idea: Cumulative 20d return / average 20d turnover (log-scaled).
High value = stock went up (or down less) with low trading activity = "quiet accumulation".
Low turnover + positive return suggests informed/institutional buying rather than retail frenzy.

This is a Quality/Momentum hybrid factor.

Direction: Positive (high factor = high expected return).
Neutralize: market cap (OLS residual).
"""

import pandas as pd
import numpy as np
import sys
import os

def compute_factor():
    # Load data
    kline = pd.read_csv("data/csi1000_kline_raw.csv", parse_dates=["date"])
    kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    # Need: close for returns, turnover for trading activity, amount for neutralization
    # Compute daily returns
    kline["ret"] = kline.groupby("stock_code")["close"].pct_change()
    
    # Rolling 20-day cumulative return (using log returns for additivity)
    kline["log_ret"] = np.log1p(kline["ret"])
    kline["cum_ret_20d"] = kline.groupby("stock_code")["log_ret"].transform(
        lambda x: x.rolling(20, min_periods=15).sum()
    )
    
    # Rolling 20-day average turnover
    kline["avg_turnover_20d"] = kline.groupby("stock_code")["turnover"].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )
    
    # Rolling 20-day average amount (for neutralization)
    kline["avg_amount_20d"] = kline.groupby("stock_code")["amount"].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )
    kline["log_amount_20d"] = np.log(kline["avg_amount_20d"].clip(lower=1))
    
    # Factor: cumulative return / average turnover
    # Add small epsilon to avoid division by zero
    # Higher = more return per unit of turnover = "efficient momentum"
    kline["raw_factor"] = kline["cum_ret_20d"] / (kline["avg_turnover_20d"].clip(lower=0.01))
    
    # Market cap proxy: use amount as proxy (as done in other factors)
    # Neutralize via OLS residual
    def neutralize_cross_section(group):
        """OLS neutralize factor by log_amount."""
        y = group["raw_factor"]
        x = group["log_amount_20d"]
        mask = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 50:
            return pd.Series(np.nan, index=group.index)
        y_m, x_m = y[mask], x[mask]
        # Winsorize with MAD before regression
        med = y_m.median()
        mad = (y_m - med).abs().median() * 1.4826
        if mad < 1e-10:
            return pd.Series(np.nan, index=group.index)
        y_clipped = y_m.clip(med - 3*mad, med + 3*mad)
        # OLS
        x_dm = x_m - x_m.mean()
        beta = (x_dm * y_clipped).sum() / (x_dm**2).sum() if (x_dm**2).sum() > 0 else 0
        alpha = y_clipped.mean() - beta * x_m.mean()
        residual = y_clipped - (alpha + beta * x_m)
        # Z-score
        std = residual.std()
        if std < 1e-10:
            return pd.Series(np.nan, index=group.index)
        z = (residual - residual.mean()) / std
        result = pd.Series(np.nan, index=group.index)
        result[mask] = z
        return result
    
    print("Computing cross-sectional neutralization...")
    kline["factor_value"] = kline.groupby("date").apply(
        neutralize_cross_section
    ).reset_index(level=0, drop=True)
    
    # Output
    output = kline[["date", "stock_code", "factor_value"]].dropna()
    output = output.rename(columns={"factor_value": "factor"})
    
    out_path = "data/factor_low_turnover_mom_v1.csv"
    output.to_csv(out_path, index=False)
    print(f"Saved {len(output)} rows to {out_path}")
    print(f"Date range: {output['date'].min()} to {output['date'].max()}")
    print(f"Stocks per date (mean): {output.groupby('date').size().mean():.0f}")
    print(f"\nFactor stats:")
    print(output["factor"].describe())
    
    return output

if __name__ == "__main__":
    compute_factor()
