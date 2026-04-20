#!/usr/bin/env python3
"""
因子 vwsr_amp_v1: Volume-Weighted Price Direction (Amplitude-Adjusted)
构造: 20日窗口内成交量加权的日内价格变动方向
升级 vs signed-only版: 用 pct_change 幅度加权，不仅方向

公式:
  amp_vol_ratio = pct_change(close) * volume / MA20(|volume| * |pct_change|)
  factor_raw = MA20(amp_vol_ratio, 20d)
  
理论: Campbell, Grossman & Wang (1993) + 幅度加权增强信号
  - 大幅上涨+放量 → 强烈动量信号
  - 小幅上涨+放量 → 弱信号
  - 下跌+放量 → 反转信号（强度由跌幅×量决定）

中性化: OLS neutralize(factor_raw, log(MA20(amount)))
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent
kline = pd.read_csv(DATA_PATH / "csi1000_kline_raw.csv")
kline["date"] = pd.to_datetime(kline["date"])
kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)

# returns
kline["ret"] = kline.groupby("stock_code")["close"].pct_change()

# abs_vol_ret_ma20 = MA20(|volume| * |ret|) — 加权绝对量
kline["abs_vol_ret"] = (kline["volume"] * kline["ret"].abs()).abs()
kline["abs_vol_ret_ma20"] = kline.groupby("stock_code")["abs_vol_ret"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

# amp_vol_ratio = ret * volume / abs_vol_ret_ma20
# Retains sign of return, magnitude scaled by "typical" dollar volume-weight
kline["amp_vol_ratio"] = np.where(
    kline["abs_vol_ret_ma20"] > 0,
    kline["ret"] * kline["volume"] / kline["abs_vol_ret_ma20"],
    np.nan
)

# 20d rolling mean
kline["factor_raw"] = kline.groupby("stock_code")["amp_vol_ratio"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

# amount_ma20 for neutralization
kline["amount_ma20"] = kline.groupby("stock_code")["amount"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

# ====== Neutralization: OLS ======
def neutralize_ols(group):
    """OLS neutralize factor_raw against log(amount_ma20), per date cross-section."""
    y = group["factor_raw"].values.astype(float)
    X_col = "log_amount_ma20"
    X = group[X_col].values.astype(float)
    
    valid = np.isfinite(y) & np.isfinite(X)
    if valid.sum() < 10:
        group["factor_neutral"] = np.nan
        return group
    
    # OLS with intercept
    X_design = np.column_stack([np.ones(valid.sum()), X[valid]])
    y_clean = y[valid]
    try:
        beta = np.linalg.lstsq(X_design, y_clean, rcond=None)[0]
        residual = np.full_like(y, np.nan)
        residual[valid] = y_clean - X_design @ beta
    except Exception:
        residual = y.copy()
    
    group["factor_neutral"] = residual
    return group

kline["log_amount_ma20"] = np.log(kline["amount_ma20"].replace(0, np.nan))
kline = kline.groupby("date", group_keys=False).apply(neutralize_ols)

# MAD winsorize + z-score per date
def mad_winsorize_zscore(group):
    vals = group["factor_neutral"]
    valid = vals.notna()
    if valid.sum() < 10:
        group["factor"] = np.nan
        return group
    
    med = vals[valid].median()
    mad = (vals[valid] - med).abs().median()
    if mad == 0 or np.isnan(mad):
        group["factor"] = np.nan
        return group
    
    scaled_mad = 1.4826 * mad
    lower = med - 3.0 * scaled_mad
    upper = med + 3.0 * scaled_mad
    clipped = vals.clip(lower, upper)
    
    std = clipped[valid].std()
    if std > 0:
        group["factor"] = (clipped - clipped[valid].mean()) / std
    else:
        group["factor"] = 0.0
    return group

kline = kline.groupby("date", group_keys=False).apply(mad_winsorize_zscore)

# Output
out = kline[["date", "stock_code", "factor"]].copy()
out["date"] = out["date"].dt.strftime("%Y-%m-%d")
out.to_csv(DATA_PATH / "factor_vwsr_amp_v1.csv", index=False, encoding="utf-8")

print(f"[OK] factor_vwsr_amp_v1 saved")
print(f"  shape: {out.shape}")
print(f"  factor stats: mean={out['factor'].mean():.4f}, std={out['factor'].std():.4f}")
print(f"  non-null: {out['factor'].notna().sum()}")

# Quick debugging
print(f"\n[DEBUG] factor_raw range: [{kline['factor_raw'].min():.4f}, {kline['factor_raw'].max():.4f}]")
print(f"[DEBUG] factor_neutral range: [{kline['factor_neutral'].min():.4f}, {kline['factor_neutral'].max():.4f}]")
print(f"[DEBUG] IC potential: corr(raw, fwd_ret) check needed")
