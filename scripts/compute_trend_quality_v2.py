#!/usr/bin/env python3
"""
Factor: Trend Quality Score (trend_quality_v1)
=======================================================
Idea: sign(20d cumulative return) × R²(linear regression fit of cumulative returns over 20d)

Stocks with CLEAN UPTRENDS get high positive values.
Stocks with CLEAN DOWNTRENDS get high negative values.  
Stocks with CHOPPY/NOISY action get near-zero values regardless of direction.

This is a Momentum × Quality hybrid:
- Pure momentum = direction of return
- Quality adjustment = consistency/smoothness of the path
- High trend quality + positive momentum = institutional/conviction buying

Direction: Positive (high factor = clean uptrend = expected outperformance)
Neutralize: log(20d avg amount) via OLS residual.
"""

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def compute_factor():
    kline = pd.read_csv("data/csi1000_kline_raw.csv", parse_dates=["date"])
    kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    # Daily return and log return
    kline["ret"] = kline.groupby("stock_code")["close"].pct_change()
    kline["log_ret"] = np.log1p(kline["ret"])
    
    # For each stock, compute rolling 20d trend quality
    def compute_trend_quality(group):
        log_rets = group["log_ret"].values
        n = len(log_rets)
        result = np.full(n, np.nan)
        
        window = 20
        min_obs = 15
        
        if n < min_obs:
            return pd.Series(result, index=group.index)
        
        # Compute cumulative log returns
        cum_log_ret = np.nancumsum(log_rets)
        
        for i in range(window - 1, n):
            start = i - window + 1
            rets_window = log_rets[start:i+1]
            valid = ~np.isnan(rets_window)
            if valid.sum() < min_obs:
                continue
            
            # Cumulative returns within window
            cum_rets = np.nancumsum(rets_window)
            valid_cum = ~np.isnan(cum_rets)
            
            # Time index
            t = np.arange(len(cum_rets), dtype=float)
            t_valid = t[valid_cum]
            y_valid = cum_rets[valid_cum]
            
            if len(y_valid) < min_obs:
                continue
            
            # Linear regression: y = a + b*t
            t_mean = t_valid.mean()
            y_mean = y_valid.mean()
            t_dm = t_valid - t_mean
            y_dm = y_valid - y_mean
            
            ss_t = (t_dm ** 2).sum()
            if ss_t < 1e-15:
                continue
            
            beta = (t_dm * y_dm).sum() / ss_t
            y_hat = y_mean + beta * t_dm
            
            ss_res = ((y_dm - beta * t_dm) ** 2).sum()
            ss_tot = (y_dm ** 2).sum()
            
            if ss_tot < 1e-15:
                result[i] = 0.0
                continue
            
            r_squared = 1.0 - ss_res / ss_tot
            r_squared = max(0, min(1, r_squared))
            
            # Total return over window
            total_ret = cum_rets[-1] if not np.isnan(cum_rets[-1]) else 0
            direction = np.sign(total_ret)
            
            # Trend quality = direction × R² (weighted by magnitude for better separation)
            # Use sqrt(R²) for less extreme values
            # Trend quality = direction × R² (clean trend only)
            # Range bound: [0, 1] before standardization
            result[i] = direction * r_squared
        
        return pd.Series(result, index=group.index)
    
    print("Computing trend quality per stock...")
    kline["raw_factor"] = kline.groupby("stock_code").apply(
        compute_trend_quality
    ).reset_index(level=0, drop=True)
    
    # Neutralization
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
    
    print("Computing cross-sectional neutralization...")
    kline["factor_value"] = kline.groupby("date").apply(
        neutralize_cs
    ).reset_index(level=0, drop=True)
    
    output = kline[["date", "stock_code", "factor_value"]].dropna()
    output = output.rename(columns={"factor_value": "factor"})
    
    out_path = "data/factor_trend_quality_v1.csv"
    output.to_csv(out_path, index=False)
    print(f"Saved {len(output)} rows to {out_path}")
    print(f"Date range: {output['date'].min()} to {output['date'].max()}")
    print(f"Stocks per date: {output.groupby('date').size().mean():.0f}")
    print(f"\nFactor stats:")
    print(output["factor"].describe())

if __name__ == "__main__":
    compute_factor()
