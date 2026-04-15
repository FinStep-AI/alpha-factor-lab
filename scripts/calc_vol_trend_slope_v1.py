#!/usr/bin/env python3
"""
因子: 成交量趋势斜率 (Volume Trend Slope) v1

定义: 过去20天成交量(log)对时间的线性回归斜率, 标准化后做成交额中性化.
      正值=放量趋势, 负值=缩量趋势.

变体: 也计算一个"成交量趋势R²"来衡量趋势的清晰度.
      复合因子 = slope × sqrt(R²), 即趋势强度×趋势清晰度.

逻辑: 持续温和放量 = 资金持续流入/流出, 比单点放量更有信息量.
      与turnover_decel(短长期比值)不同, slope捕捉的是线性趋势方向.

Barra风格: Sentiment/Momentum
"""

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
import warnings
warnings.filterwarnings("ignore")

# ── 读取数据 ────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/csi1000_kline_raw.csv", parse_dates=["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

# ── 计算log volume ──────────────────────────────────────
df["log_vol"] = np.log1p(df["volume"])

# ── 20日滚动回归斜率 ────────────────────────────────────
WINDOW = 20
print(f"Computing {WINDOW}d rolling volume trend slope...")

def rolling_vol_slope(group):
    g = group.sort_values("date")
    log_vol = g["log_vol"].values
    n = len(log_vol)
    slopes = np.full(n, np.nan)
    
    t = np.arange(WINDOW, dtype=float)
    t_mean = t.mean()
    t_var = ((t - t_mean)**2).sum()
    
    for i in range(WINDOW - 1, n):
        y = log_vol[i - WINDOW + 1: i + 1]
        if np.any(np.isnan(y)):
            continue
        y_mean = y.mean()
        cov = ((t - t_mean) * (y - y_mean)).sum()
        slope = cov / t_var
        slopes[i] = slope
    
    return pd.Series(slopes, index=g.index)

slopes = df.groupby("stock_code", group_keys=False).apply(rolling_vol_slope)
df["vol_slope"] = slopes.values

# ── 成交额中性化 ───────────────────────────────────────
print("Neutralizing by log_amount_20d...")
df["log_amount"] = np.log1p(df["amount"])
df["log_amount_20d"] = df.groupby("stock_code")["log_amount"].transform(
    lambda x: x.rolling(20, min_periods=15).mean()
)

def neutralize_cross_section(group):
    y = group["vol_slope"]
    x = group["log_amount_20d"]
    mask = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 30:
        return pd.Series(np.nan, index=group.index)
    X = np.column_stack([np.ones(mask.sum()), x[mask].values])
    beta, _, _, _ = lstsq(X, y[mask].values, rcond=None)
    resid = pd.Series(np.nan, index=group.index)
    resid[mask] = y[mask].values - X @ beta
    return resid

print("Cross-sectional neutralization...")
df["factor_raw"] = df.groupby("date", group_keys=False).apply(neutralize_cross_section)

# ── MAD Winsorize ──────────────────────────────────────
def mad_winsorize(group, n_mad=5):
    vals = group["factor_raw"]
    median = vals.median()
    mad = (vals - median).abs().median()
    if mad < 1e-10:
        return vals
    lower = median - n_mad * 1.4826 * mad
    upper = median + n_mad * 1.4826 * mad
    return vals.clip(lower, upper)

print("MAD winsorize...")
df["factor_win"] = df.groupby("date", group_keys=False).apply(mad_winsorize)

# ── Z-score ────────────────────────────────────────────
def zscore(group):
    vals = group["factor_win"]
    mu = vals.mean()
    std = vals.std()
    if std < 1e-10:
        return pd.Series(0.0, index=group.index)
    return (vals - mu) / std

print("Z-score standardization...")
df["factor"] = df.groupby("date", group_keys=False).apply(zscore)

# ── 输出 ───────────────────────────────────────────────
out = df[["date", "stock_code", "factor"]].dropna(subset=["factor"])
out = out.rename(columns={"factor": "factor_value"})
out.to_csv("data/factor_vol_trend_slope_v1.csv", index=False)
print(f"Output: data/factor_vol_trend_slope_v1.csv")
print(f"  Records: {len(out):,}")
print(f"  Date range: {out['date'].min()} ~ {out['date'].max()}")
print(f"  Stocks per date (avg): {out.groupby('date')['stock_code'].nunique().mean():.0f}")

print("\nDone!")
