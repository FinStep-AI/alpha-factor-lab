#!/usr/bin/env python3
"""
Factor: Down-Day Volume Ratio (down_vol_ratio_v1)
=======================================================
Idea: mean(volume on down days) / mean(volume on up days), 20d rolling window.
High ratio = heavy volume accompanies declines (distribution/selling pressure).
Low ratio = selling dries up quickly, volume mainly on up days (accumulation).

Use NEGATIVE direction: do multi on low ratio (accumulation), short high ratio (distribution).

Barra Style: 微观结构 / Quality proxy
Neutralize: log(20d avg amount) via OLS residual.
"""

import pandas as pd
import numpy as np

def compute_factor():
    kline = pd.read_csv("data/csi1000_kline_raw.csv", parse_dates=["date"])
    kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    # Daily return
    kline["ret"] = kline.groupby("stock_code")["close"].pct_change()
    
    # Flag up/down days
    kline["is_down"] = (kline["ret"] < 0).astype(float)
    kline["is_up"] = (kline["ret"] > 0).astype(float)
    
    # Volume on down days (0 if up day)
    kline["vol_down"] = kline["volume"] * kline["is_down"]
    kline["vol_up"] = kline["volume"] * kline["is_up"]
    
    # Rolling sums
    kline["sum_vol_down_20d"] = kline.groupby("stock_code")["vol_down"].transform(
        lambda x: x.rolling(20, min_periods=15).sum()
    )
    kline["sum_vol_up_20d"] = kline.groupby("stock_code")["vol_up"].transform(
        lambda x: x.rolling(20, min_periods=15).sum()
    )
    kline["count_down_20d"] = kline.groupby("stock_code")["is_down"].transform(
        lambda x: x.rolling(20, min_periods=15).sum()
    )
    kline["count_up_20d"] = kline.groupby("stock_code")["is_up"].transform(
        lambda x: x.rolling(20, min_periods=15).sum()
    )
    
    # Average volume per down day / Average volume per up day
    kline["avg_vol_down"] = kline["sum_vol_down_20d"] / kline["count_down_20d"].clip(lower=1)
    kline["avg_vol_up"] = kline["sum_vol_up_20d"] / kline["count_up_20d"].clip(lower=1)
    
    # Ratio: log scale for normality
    kline["raw_factor"] = np.log(kline["avg_vol_down"].clip(lower=1) / kline["avg_vol_up"].clip(lower=1))
    
    # Neutralization variable: log average 20d amount
    kline["avg_amount_20d"] = kline.groupby("stock_code")["amount"].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )
    kline["log_amount_20d"] = np.log(kline["avg_amount_20d"].clip(lower=1))
    
    # Cross-sectional neutralization
    def neutralize_cs(group):
        y = group["raw_factor"]
        x = group["log_amount_20d"]
        mask = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 50:
            return pd.Series(np.nan, index=group.index)
        y_m, x_m = y[mask], x[mask]
        # Winsorize MAD 3x
        med = y_m.median()
        mad = (y_m - med).abs().median() * 1.4826
        if mad < 1e-10:
            return pd.Series(np.nan, index=group.index)
        y_clipped = y_m.clip(med - 3*mad, med + 3*mad)
        # OLS
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
    
    # Negate for "low ratio = accumulation = good" direction
    kline["factor_value"] = -kline["factor_value"]
    
    output = kline[["date", "stock_code", "factor_value"]].dropna()
    output = output.rename(columns={"factor_value": "factor"})
    
    out_path = "data/factor_down_vol_ratio_v1.csv"
    output.to_csv(out_path, index=False)
    print(f"Saved {len(output)} rows to {out_path}")
    print(f"Date range: {output['date'].min()} to {output['date'].max()}")
    print(f"Stocks per date: {output.groupby('date').size().mean():.0f}")
    print(f"\nFactor stats:")
    print(output["factor"].describe())

if __name__ == "__main__":
    compute_factor()
