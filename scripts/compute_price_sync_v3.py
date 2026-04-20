#!/usr/bin/env python3
"""
因子猎人 2026-04-19 周末挖矿 - 价格同步性 v3 (Rank 变换版)
Hou & Moskowitz (2005) R² 变体

关键改进: rank变换后中性化
  - 截面rank变换可以放大极端信号差异
  - 帮助G4/G5更清晰分离 (v1/v2中G4≈G5)
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

print("Loading data ...")
kline = pd.read_csv("data/csi1000_kline_raw.csv")
kline["date"] = pd.to_datetime(kline["date"])
kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)
kline["ret_cc"] = kline.groupby("stock_code")["close"].pct_change()
kline["amount_ma20"] = kline.groupby("stock_code")["amount"].transform(
    lambda x: x.rolling(20, min_periods=15).mean()
)

# Market return (amount-weighted)
dd = kline[["date", "stock_code", "ret_cc", "amount", "amount_ma20"]].dropna(subset=["ret_cc"])
dd_valid = dd[dd["amount"] > 0].copy()
mkt = dd_valid.groupby("date").apply(
    lambda g: pd.Series({"mkt_ret": np.average(g["ret_cc"].values, weights=g["amount"].values)})
).reset_index()
dd = dd.merge(mkt[["date", "mkt_ret"]], on="date", how="inner")
dd = dd.sort_values(["stock_code", "date"]).reset_index(drop=True)

def rolling_r2(group, window=20):
    vals = group[["ret_cc", "mkt_ret"]].values
    n = len(vals)
    r2_vals = np.full(n, np.nan)
    for i in range(window - 1, n):
        win = vals[i - window + 1:i + 1, :]
        x, y = win[:, 1], win[:, 0]
        if np.sum(np.isfinite(x)) >= max(window // 2, 10) and np.sum(np.isfinite(y)) >= max(window // 2, 10):
            mask = np.isfinite(x) & np.isfinite(y)
            x_clean, y_clean = x[mask], y[mask]
            if len(x_clean) >= 10:
                try:
                    r_val, _, _, _, _ = stats.linregress(x_clean, y_clean)
                    r2_vals[i] = r_val ** 2
                except:
                    pass
    return r2_vals

print("Rolling R² (20-day window) ...")
results = []
for code, grp in dd.groupby("stock_code"):
    r2 = rolling_r2(grp, window=20)
    grp = grp.copy()
    grp["r2"] = r2
    results.append(grp[["date", "stock_code", "r2", "amount_ma20"]])

dd = pd.concat(results, ignore_index=True)

# === KEY CHANGE: Rank transform R² BEFORE neutralization ===
# This amplifies the relative ordering and should help G4/G5 separation
print("Rank transforming R² ...")
dd["factor_raw"] = dd.groupby("date")["r2"].rank(pct=True)  # [0, 1], higher = more synchronous

# Neutralize: OLS on log(amount_ma20)
print("Neutralizing ...")
neutralized_records = []
for date, grp in dd.groupby("date"):
    vals = grp[["date", "stock_code", "factor_raw", "amount_ma20"]].dropna()
    if len(vals) < 30:
        continue
    log_amt = np.log(vals["amount_ma20"].values + 1)
    y = vals["factor_raw"].values
    X = np.column_stack([np.ones(len(log_amt)), log_amt])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
    except:
        continue
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    if mad < 1e-10:
        continue
    residuals = np.clip(residuals, med - 5.5*mad, med + 5.5*mad)
    mu, sigma = residuals.mean(), residuals.std()
    if sigma < 1e-10:
        continue
    z = (residuals - mu) / sigma
    out = pd.DataFrame({
        "date": vals["date"].values,
        "stock_code": vals["stock_code"].values,
    })
    out["factor_value"] = z
    neutralized_records.append(out)

factor_df = pd.concat(neutralized_records, ignore_index=True)
output_path = "data/factor_price_sync_v3.csv"
factor_df.to_csv(output_path, index=False)
print(f"\nFactor saved: {output_path}, shape={factor_df.shape}")
print(f"Range: {factor_df['date'].min()} ~ {factor_df['date'].max()}")
print(f"Stocks/date median: {factor_df.groupby('date')['stock_code'].nunique().median():.0f}")
