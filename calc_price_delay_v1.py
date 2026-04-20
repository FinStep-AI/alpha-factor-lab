#!/usr/bin/env python3
"""
因子 price_delay_v1: 市场同步延迟因子 (Price Delay)
构建: 20日滚动窗口个股收益率对CSI1000等权市场收益回归的R²
      低R² = 价格同步延迟高 = 信息效率低 = 未被充分定价 = 未来超额收益

公式:
  r_i,t = alpha + beta * r_m,t + eps
  delay_score = 1 - R² (反向: 高延迟 = 高因子值)
  中性化: OLS neutralize(delay_score, log(MA20(amount)))

理论来源:
  - Hou & Moskowitz (2024) "Stock price delay and the cross-section of expected returns", RFS
  - 核心发现: 价格延迟(R²)能预测未来1个月收益，低R²股票未来收益更高
  - 机制: 信息摩擦小/交易摩擦小的股票更快反映信息 → 高R² → 可能已充分定价
  - A股CSI1000小盘股摩擦更大 → 延迟效应可能更强

中性化: OLS neutralize(factor_raw, log(MA20(amount)))
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Output to root level (matching factors_vwsr_amp_v1.py convention)
DATA_PATH = Path(__file__).resolve().parent / "data"

# ====== 1. Load data ======
print("[1/6] Loading data...")
kline = pd.read_csv(DATA_PATH / "csi1000_kline_raw.csv")
kline["date"] = pd.to_datetime(kline["date"])
kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)

# daily returns
kline["ret"] = kline.groupby("stock_code")["close"].pct_change()

# ====== 2. Market return (CSI1000 equal-weight proxy) ======
print("[2/6] Computing market return...")
# Use equal-weight average of all stocks' daily returns as market proxy
market_ret = kline.groupby("date")["ret"].mean().reset_index()
market_ret.columns = ["date", "mkt_ret"]
kline = kline.merge(market_ret, on="date", how="left")

# ====== 3. Rolling R² of stock vs market ======
print("[3/6] Computing rolling R² (20d window)...")
window = 20

def calc_rolling_r2(group):
    """For each window, regress stock ret on market ret and compute R²"""
    stock_ret = group["ret"].values
    mkt_ret = group["mkt_ret"].values
    n = len(stock_ret)
    
    r2_vals = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        y = stock_ret[i - window + 1:i + 1]
        x = mkt_ret[i - window + 1:i + 1]
        
        valid = np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 10:
            continue
        
        y_v = y[valid]
        x_v = x[valid]
        
        # Simple OLS: R² = corr²
        if np.std(y_v) > 0 and np.std(x_v) > 0:
            corr = np.corrcoef(x_v, y_v)[0, 1]
            if not np.isnan(corr):
                r2_vals[i] = corr ** 2
    
    group["r2_20d"] = r2_vals
    return group

kline = kline.groupby("stock_code", group_keys=False).apply(calc_rolling_r2)

# ====== 4. Factor: 1 - R² (high delay = high factor value) ======
print("[4/6] Building delay factor...")
kline["factor_raw"] = 1.0 - kline["r2_20d"]

# amount_ma20 for neutralization
kline["amount_ma20"] = kline.groupby("stock_code")["amount"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)
kline["log_amount_ma20"] = np.log(kline["amount_ma20"].replace(0, np.nan))

# ====== 5. Neutralization: OLS ======
print("[5/6] Neutralizing...")
def neutralize_ols(group):
    """OLS neutralize factor_raw against log(amount_ma20), per date cross-section."""
    y = group["factor_raw"].values.astype(float)
    X_col = "log_amount_ma20"
    X = group[X_col].values.astype(float)
    
    valid = np.isfinite(y) & np.isfinite(X)
    if valid.sum() < 10:
        group["factor_neutral"] = np.nan
        return group
    
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

# ====== 6. Output ======
print("[6/6] Outputting...")
out = kline[["date", "stock_code", "factor"]].copy()
out["date"] = out["date"].dt.strftime("%Y-%m-%d")
out.to_csv(DATA_PATH / "factor_price_delay_v1.csv", index=False, encoding="utf-8")

print(f"[OK] factor_price_delay_v1 saved")
print(f"  shape: {out.shape}")
print(f"  factor stats: mean={out['factor'].mean():.4f}, std={out['factor'].std():.4f}")
print(f"  non-null: {out['factor'].notna().sum()}")

# Quick debug
valid_df = kline[kline["factor"].notna()]
print(f"\n[DEBUG] r2_20d range: [{kline['r2_20d'].min():.4f}, {kline['r2_20d'].max():.4f}]")
print(f"[DEBUG] factor_raw range: [{kline['factor_raw'].min():.4f}, {kline['factor_raw'].max():.4f}]")
print(f"[DEBUG] factor_neutral range: [{kline['factor_neutral'].min():.4f}, {kline['factor_neutral'].max():.4f}]")
print(f"[DEBUG] Valid stock-days: {len(valid_df)}")

# Per-stock R² distribution
stock_r2 = kline.groupby("stock_code")["r2_20d"].mean()
print(f"\n[DEBUG] Avg R² per stock: mean={stock_r2.mean():.4f}, std={stock_r2.std():.4f}")
print(f"[DEBUG] R² distribution: 5%={stock_r2.quantile(0.05):.4f}, 50%={stock_r2.median():.4f}, 95%={stock_r2.quantile(0.95):.4f}")
