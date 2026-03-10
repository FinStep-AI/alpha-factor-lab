#!/usr/bin/env python3
"""
Factor: Information Discreteness / Frog-in-the-Pan (fip_v1)
=============================================================
Based on: Da, Gurun & Warachka (2014) "Frog in the Pan: 
Continuous Information and Momentum", Review of Financial Studies.

Formula:
  ID = sign(sum(ret, 20d)) × (frac_positive_days - frac_negative_days)

Low ID = return achieved through few large discrete moves → investors underreact
High ID = return achieved through many small continuous moves → investors notice

On CSI1000 (small-cap reversal-dominant), we test NEGATIVE direction:
- Low ID (discrete moves) → underreaction → momentum (might continue)
- High ID (continuous moves) → already noticed → reversal

We test both directions. Amount-neutralized.
"""

import pandas as pd
import numpy as np

def compute_factor():
    kline = pd.read_csv("data/csi1000_kline_raw.csv", parse_dates=["date"])
    kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    kline["ret"] = kline.groupby("stock_code")["close"].pct_change()
    
    # Rolling 20d
    window = 20
    min_obs = 15
    
    # Sum of returns
    kline["sum_ret_20d"] = kline.groupby("stock_code")["ret"].transform(
        lambda x: x.rolling(window, min_periods=min_obs).sum()
    )
    
    # Fraction of positive and negative days
    kline["pos_day"] = (kline["ret"] > 0).astype(float)
    kline["neg_day"] = (kline["ret"] < 0).astype(float)
    
    kline["frac_pos_20d"] = kline.groupby("stock_code")["pos_day"].transform(
        lambda x: x.rolling(window, min_periods=min_obs).mean()
    )
    kline["frac_neg_20d"] = kline.groupby("stock_code")["neg_day"].transform(
        lambda x: x.rolling(window, min_periods=min_obs).mean()
    )
    
    # Information Discreteness
    kline["raw_factor"] = np.sign(kline["sum_ret_20d"]) * (kline["frac_pos_20d"] - kline["frac_neg_20d"])
    
    # Neutralize by amount
    kline["avg_amount_20d"] = kline.groupby("stock_code")["amount"].transform(
        lambda x: x.rolling(window, min_periods=min_obs).mean()
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
    
    # Save POSITIVE direction first (high ID → high factor value)
    output = kline[["date", "stock_code", "factor_value"]].dropna()
    output = output.rename(columns={"factor_value": "factor"})
    
    out_path = "data/factor_fip_v1.csv"
    output.to_csv(out_path, index=False)
    print(f"Saved {len(output)} rows to {out_path}")
    print(f"Date range: {output['date'].min()} to {output['date'].max()}")
    print(f"Stocks per date: {output.groupby('date').size().mean():.0f}")
    print(f"\nFactor stats:")
    print(output["factor"].describe())
    
    # Also save negated version
    output_neg = output.copy()
    output_neg["factor"] = -output_neg["factor"]
    out_path_neg = "data/factor_fip_neg_v1.csv"
    output_neg.to_csv(out_path_neg, index=False)
    print(f"\nAlso saved negated version to {out_path_neg}")

if __name__ == "__main__":
    compute_factor()
