#!/usr/bin/env python3
"""
Factor: VWAP Deviation Trend (vwap_trend_v1)
=============================================================
The SLOPE of (close-VWAP)/VWAP over the past 20 days.

close_vwap_dev (level) already works well (IC=0.04, Sharpe=1.49).
This factor captures the CHANGE in that relationship:
- Positive slope = VWAP deviation improving = increasing buying pressure
- Negative slope = VWAP deviation deteriorating = increasing selling pressure

Uses linear regression slope of daily (close-VWAP)/VWAP values over 20d.
Neutralize by log(20d avg amount).

This should capture MOMENTUM of the microstructure signal, 
complementing the LEVEL captured by close_vwap_dev.
"""

import pandas as pd
import numpy as np

def compute_factor():
    kline = pd.read_csv("data/csi1000_kline_raw.csv", parse_dates=["date"])
    kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    # VWAP = amount / volume (total money / total shares)
    # But VWAP needs to be daily average price = amount / volume / 100 
    # Actually VWAP = sum(price×volume)/sum(volume) ≈ amount/volume for daily
    kline["vwap"] = kline["amount"] / (kline["volume"] * 100).clip(lower=1)
    # Note: volume is in 手 (100 shares), amount is in 元
    
    # Daily VWAP deviation
    kline["vwap_dev"] = (kline["close"] - kline["vwap"]) / kline["vwap"]
    
    # Rolling 20d slope of VWAP deviation
    window = 20
    min_obs = 15
    
    def rolling_slope(group):
        """Compute rolling slope of vwap_dev over 20d window."""
        vals = group["vwap_dev"].values
        n = len(vals)
        result = np.full(n, np.nan)
        
        t = np.arange(window, dtype=float)
        t_mean = t.mean()
        t_dm = t - t_mean
        ss_t = (t_dm ** 2).sum()
        
        for i in range(window - 1, n):
            start = i - window + 1
            y = vals[start:i+1]
            valid = ~np.isnan(y)
            if valid.sum() < min_obs:
                continue
            
            # Use only valid points
            y_valid = y[valid]
            t_valid = t[valid]
            t_v_mean = t_valid.mean()
            t_v_dm = t_valid - t_v_mean
            y_v_mean = y_valid.mean()
            y_v_dm = y_valid - y_v_mean
            
            ss_t_v = (t_v_dm ** 2).sum()
            if ss_t_v < 1e-15:
                continue
            
            beta = (t_v_dm * y_v_dm).sum() / ss_t_v
            result[i] = beta
        
        return pd.Series(result, index=group.index)
    
    print("Computing rolling slopes per stock...")
    kline["raw_factor"] = kline.groupby("stock_code").apply(
        rolling_slope
    ).reset_index(level=0, drop=True)
    
    # Neutralize by amount
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
    
    out_path = "data/factor_vwap_trend_v1.csv"
    output.to_csv(out_path, index=False)
    print(f"Saved {len(output)} rows to {out_path}")
    print(f"Date range: {output['date'].min()} to {output['date'].max()}")
    print(f"Stocks per date: {output.groupby('date').size().mean():.0f}")
    print(output["factor"].describe())

if __name__ == "__main__":
    compute_factor()
