#!/usr/bin/env python3
"""
因子猎人 2026-04-19 周末挖矿
因子: 价格同步性 (Price Synchronicity / Market Model R²)

论文: Hou & Moskowitz (2005) "Market Frictions, Price Delay, and the Cross-Section of Expected Returns"
      The Review of Financial Studies, 18(3), 981-1020

理论: 价格反应市场信息的快慢程度 = 市场摩擦/信息不对称程度的代理变量。
      R² = 个股收益率对市场收益率的回归拟合优度。
      低R² = 价格同步性低 = 信息延迟大 = 市场摩擦大 = 应该有更高风险溢价。

构造:
  1. 计算每日市场等权收益率 (CSI1000成分股简单平均)
  2. 用20日滚动窗口对每只股票做: r_i,t = α + β·r_m,t + ε
  3. 提取滚动R²
  4. 取R²的负值 (负相关性: 低R² = 因子值高 = 预期收益高)
  5. 成交额OLS中性化 + MAD缩尾 + z-score

中性化: log(MA20_amount) 成交额代理市值

回测查阅: --forward-days 20 (中频因子)
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. Load data
# ============================================================
print("Loading data ...")
kline = pd.read_csv("data/csi1000_kline_raw.csv")
returns = pd.read_csv("data/csi1000_returns.csv")

kline["date"] = pd.to_datetime(kline["date"])
returns["date"] = pd.to_datetime(returns["date"])

print(f"K-line: {kline.shape}, Returns: {returns.shape}")
print(f"Date range: {kline['date'].min()} ~ {kline['date'].max()}")

# Precompute close-to-close daily return
kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)
kline["ret_cc"] = kline.groupby("stock_code")["close"].pct_change()

# Compute 20-day average amount
kline["amount_ma20"] = kline.groupby("stock_code")["amount"].transform(
    lambda x: x.rolling(20, min_periods=15).mean()
)

# ============================================================
# 2. Compute daily market-cap-weighted market return proxy
#    (use amount = market_cap x return_proxy as weight proxy)
# ============================================================
print("Computing market returns ...")
kline["abs_ret"] = kline["ret_cc"].abs()

dd = kline[["date", "stock_code", "ret_cc", "amount", "amount_ma20"]].dropna(subset=["ret_cc"])

# Market return = amount-weighted average of individual returns (skip zero-amount)
dd_valid = dd[dd["amount"] > 0].copy()
mkt = dd_valid.groupby("date").apply(
    lambda g: pd.Series({
        "mkt_ret": np.average(g["ret_cc"].values, weights=g["amount"].values),
    })
).reset_index()
mkt = mkt.sort_values("date").dropna()

# ============================================================
# 3. Rolling R² from market model
#    model: r_i = α + β·r_m + ε
#    R² measures how synchronous stock is with market
# ============================================================
print("Computing rolling market-model R² ...")

# Merge market return with individual returns
dd = dd.merge(mkt[["date", "mkt_ret"]], on="date", how="inner")
dd = dd.sort_values(["stock_code", "date"]).reset_index(drop=True)

def rolling_r2(group, window=20):
    """Compute rolling R² of stock return on market return."""
    vals = group[["ret_cc", "mkt_ret"]].values
    n = len(vals)
    r2_vals = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        win = vals[i - window + 1:i + 1, :]
        x = win[:, 1]  # market return
        y = win[:, 0]  # stock return
        if np.sum(np.isfinite(x)) >= max(window // 2, 10) and np.sum(np.isfinite(y)) >= max(window // 2, 10):
            mask = np.isfinite(x) & np.isfinite(y)
            x_clean = x[mask]
            y_clean = y[mask]
            if len(x_clean) >= 10:
                try:
                    slope, intercept, r_val, p_val, std_err = stats.linregress(x_clean, y_clean)
                    r2_vals[i] = r_val ** 2
                except:
                    pass
    return r2_vals

# Compute rolling R² for each stock
print("  Running rolling regressions (this may take a few minutes) ...")
results = []
for code, grp in dd.groupby("stock_code"):
    r2 = rolling_r2(grp, window=60)  # v2: 60-day window for more stable estimate
    grp = grp.copy()
    grp["r2"] = r2
    results.append(grp[["date", "stock_code", "r2", "amount_ma20"]])

dd = pd.concat(results, ignore_index=True)

# Test 1 (-R²): IC=-0.049(t=-3.86), mono=-1.0
#   In CSI1000: LOW R² → LOW returns (high factor → low return = negative IC)
#   CSI1000 behaves opposite: HIGH R² stocks perform BETTER
# => Use R² directly (no negation)
dd["factor_raw"] = dd["r2"]

# Neutralize: OLS regression on log(amount_ma20)
print("Neutralizing: OLS on log(amount_ma20) ...")

neutralized_records = []
for date, grp in dd.groupby("date"):
    vals = grp[["date", "stock_code", "factor_raw", "amount_ma20"]].dropna()
    if len(vals) < 30:
        continue
    
    # log transform amount
    log_amt = np.log(vals["amount_ma20"].values + 1)
    y = vals["factor_raw"].values
    
    # OLS regression
    X = np.column_stack([np.ones(len(log_amt)), log_amt])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
    except:
        continue
    
    # MAD winsorize
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    if mad < 1e-10:
        continue
    upper = med + 5.5 * mad
    lower = med - 5.5 * mad
    residuals = np.clip(residuals, lower, upper)
    
    # z-score
    mu = residuals.mean()
    sigma = residuals.std()
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
factor_df["date"] = pd.to_datetime(factor_df["date"])
factor_df["stock_code"] = factor_df["stock_code"].astype(str)

# Save
output_path = "data/factor_price_sync_v2.csv"
factor_df.to_csv(output_path, index=False)
print(f"\nFactor saved: {output_path}")
print(f"Shape: {factor_df.shape}")
print(f"Date range: {factor_df['date'].min()} ~ {factor_df['date'].max()}")
print(f"Stocks per date (median): {factor_df.groupby('date')['stock_code'].nunique().median():.0f}")

# Quick stats
print(f"\nFactor value stats:")
print(f"  mean={factor_df['factor_value'].mean():.4f}, std={factor_df['factor_value'].std():.4f}")
print(f"  min={factor_df['factor_value'].min():.4f}, max={factor_df['factor_value'].max():.4f}")

print("\nDone! Factor computed successfully.")
print(f"Run backtest with: python3 skills/alpha-factor-lab/scripts/factor_backtest.py "
      f"--factor {output_path} --returns data/csi1000_returns.csv --groups 5 --forward-days 20 --cost 0.002 --output output/price_sync_v1/")
