#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子核心计算脚本（独立运行，写入中间数据）：

目标因子: roe_growth_v1  — ROE同比增长率（动态均值）中性化
方向：Growth/Quality

逻辑：
  1. 将基本面报告数据映射到交易日
     → 季报披露后N天首次可用（N=info_lag_days=30）
  2. 取最近两个"已披露"季度的 ROE
     → 若 t 附近有 Q(i-4) 和 Q(i)，用 annualized_growth = (ROE_i - ROE_i-4)
  3. 在 t 的截面，对 annualized_roe_change 成交额中性化 + MAD winsorize + z-score
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
KLINE_PATH = BASE_DIR / "data" / "csi1000_kline_raw.csv"
FUND_PATH  = BASE_DIR / "data" / "csi1000_fundamental_cache.csv"
OUTPUT_DIR = BASE_DIR / "data"
INFO_LAG_DAYS = 30  # 季报披露后可用的滞后天数（30天）

# ─────────────────────────────────────────
# 1. 加载数据
# ─────────────────────────────────────────
print("[1] 加载 K 线数据...")
kline = pd.read_csv(KLINE_PATH)
kline["date"] = pd.to_datetime(kline["date"])
kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)

print("[2] 加载基本面数据...")
fund = pd.read_csv(FUND_PATH)
fund["report_date"] = pd.to_datetime(fund["report_date"])
fund = fund.sort_values(["stock_code", "report_date"]).reset_index(drop=True)

# ─────────────────────────────────────────
# 2. 构建"可交易日 × 可用的ROE"面板
# ─────────────────────────────────────────
print("[3] 将季报映射到交易日（信息解锁日）...")

# 信息解锁日 = 报告日期 + 30天（典型季报披露滞后）
fund = fund.copy()
fund["info_date"] = fund["report_date"] + pd.Timedelta(days=INFO_LAG_DAYS)

# 向前填充：每个交易日 t，对股票 i，找到 info_date <= t 的最新 record
# 注意：fund 可能有多个季度 -> 取最新的已披露季度
all_dates       = kline["date"].unique()
all_stocks      = kline["stock_code"].unique()

# expand to (date, stock) grid
grid = pd.MultiIndex.from_product([all_dates, all_stocks], names=["date", "stock_code"])
panel = pd.DataFrame(index=grid).reset_index()

# Merge latest info for each (date, stock)
fund_indexed = fund.set_index("info_date")[["stock_code", "bps", "roe"]].rename(
    columns={"stock_code": "sc"}
)

# Sort and forward-fill (但没有连续索引时，groupby-based 方法更好)
# Instead, use apply per stock with merge_asof
panel = panel.sort_values(["stock_code", "date"])

# Split by stock
results = []
for sc, grp in panel.groupby("stock_code"):
    f = fund[fund["stock_code"] == sc].sort_values("info_date")
    if len(f) == 0:
        continue
    # merge_asof: left join, take latest info_date <= date
    merged = pd.merge_asof(
        grp.sort_values("date"),
        f.rename(columns={"stock_code": "sc_f"})[["info_date", "bps", "roe"]].sort_values("info_date"),
        left_on="date", right_on="info_date",
        direction="backward", by=None
    )
    results.append(merged)

panel_merged = pd.concat(results, ignore_index=True)
print(f"  面板大小: {panel_merged.shape}")
print(f"  有效 ROE 记录: {panel_merged['roe'].notna().sum():,}")

# ─────────────────────────────────────────
# 3. 计算季度 ROE 同比变化
# ─────────────────────────────────────────
print("[4] 计算 ROE 年度化同比增长率...")

def get_prev_quarter_roe(row):
    """取同一股票的报告期前后各一列的 avalue，最复杂，这里用更 efficient 方法"""
    pass

# Efficent: sort by (stock, info_date), group by stock
panel_merged = panel_merged.sort_values(["stock_code", "date"]).reset_index(drop=True)

# For each stock, shift ROE by 4 quarters (4 periods = 1 year)
# But information is sparse (quarterly).  
# Strategy: build a quarterly panel of roe values, 
# then apply 4-step lag within stock.
quarterly_roe = (
    fund.sort_values(["stock_code", "report_date"])
        .groupby("stock_code")["roe"]
        .apply(lambda x: x.shift(-4))
        .reset_index()
)
# Actually simpler: for each stock, get the last two ROEs OBSERVED
# Use rolling windows of the available quarter history.

# More practical: within each stock, make a wide table of available quarters
# then at each trading date, get the last 2 ROE available (current and 1 year before)
# For a stock with quarterly roe_2022Q1, roe_2022Q2, ..., roe_2025Q2
# at date t, the latest available quarter <= t+30d is current quarter
# the 4-quarter-ago available quarter is the growth reference

# Build per-stock quarterly ROE wide-form
# Using fund, already sorted by (stock_code, report_date)
fund_sorted = fund.sort_values(["stock_code", "report_date"]).copy()

# For each stock, shift ROE by 4 quarters back to align "current" vs "4Q ago"
# Note: the "current" ROE available at date t is in fund[sc, quarter <= t+30d]
# The "4Q ago" ROE = current quarter ROE shifted by 4 quarters
# Both are station-based, then merge back to trading dates.

# Step 3.1: align to dates -> take latest available
fund_sorted["info_date"] = fund_sorted["report_date"] + pd.Timedelta(days=INFO_LAG_DAYS)

# Per stock, do merge_asof then compute yoy_growth
panel_final = []
for sc, grp in panel_merged.groupby("stock_code"):
    # Get quarterly roe for this stock
    f_quarterly = fund_sorted[fund_sorted["stock_code"] == sc][["report_date", "info_date", "roe"]].copy()
    f_quarterly = f_quarterly.sort_values("info_date").drop_duplicates("info_date", keep="last")
    
    if len(f_quarterly) < 5:
        continue  # not enough quarters
    
    # Build "current" roe and "4Q ago" roe columns aligned to info_date
    f_quarterly = f_quarterly.copy()
    f_quarterly["roe_lag4"] = f_quarterly["roe"].shift(4)
    
    # For each quarter in f_quarterly, we can compute roe growth only if lag4 exists
    f_quarterly = f_quarterly.dropna(subset=["roe_lag4"])
    
    # Choose the "current" quarter's info_date as the validity start date
    # At this info_date, both current roe and lag4 roe are known
    # But since this quarter becomes "current" only AFTER info_date,
    # we associate growth value with info_date.
    
    growth_map = f_quarterly.set_index("info_date")[["roe_lag4", "roe"]].rename(
        columns={"roe_lag4": "roe_prev", "roe": "roe_curr"}
    )
    
    # Map each trading date in grp to the latest growth available (info_date <= date)
    grp_sorted = grp.sort_values("date").copy()
    # merge_asof: for each date, get growth if a quarter has been disclosed
    # since quarter info_date starts being available AFTER info_date
    grp_mapped = pd.merge_asof(
        grp_sorted, growth_map.reset_index().rename(columns={"info_date": "growth_info_date"}),
        left_on="date", right_on="growth_info_date",
        direction="backward"
    )
    
    panel_final.append(grp_mapped)

panel_final = pd.concat(panel_final, ignore_index=True)
panel_final["roe_growth"] = panel_final["roe_curr"] - panel_final["roe_prev"]
panel_final["roe_growth"] = panel_final["roe_growth"].clip(-50, 50)  # sanity cap

print(f"  最终截面大小: {panel_final.shape}")
print(f"  ROE有增长值: {panel_final['roe_growth'].notna().sum():,}")
print(f"  ROE Growth Descr:\n{panel_final['roe_growth'].describe().round(2)}")

# ─────────────────────────────────────────
# 5. 市值中性化 + winsorize + z-score
# ─────────────────────────────────────────
print("[5] 截面中性化 & 标准化...")

def mad_winsorize(s: pd.Series, n_mad: float = 3.0) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return s
    scaled = 1.4826 * mad
    return s.clip(med - n_mad * scaled, med + n_mad * scaled)

def ols_residual(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < X.shape[1] + 2:
        return y
    y_c = y[mask]; X_c = X[mask]
    try:
        XtX = X_c.T @ X_c + 1e-10 * np.eye(X_c.shape[1])
        beta = np.linalg.solve(XtX, X_c.T @ y_c)
        res = np.full_like(y, np.nan)
        res[mask] = y_c - X_c @ beta
        return res
    except np.linalg.LinAlgError:
        return y

panel_final = panel_final.reset_index(drop=True)
factor_vals = pd.Series(np.nan, index=panel_final.index)

for date, group in panel_final.groupby("date"):
    idx = group.index
    y = group["roe_growth"].values.astype(float)
    valid_mask = np.isfinite(y)
    if valid_mask.sum() < 10:
        continue
    
    # 市值代理: log(amount * turnover), amount as proxy
    if "amount" in group.columns and "turnover" in group.columns:
        amt = group["amount"].values.astype(float)
        trn = group["turnover"].values.astype(float)
        # 用成交额均值作市值代理（or take log_amount if available ≈ proxy）
        log_amt = np.log(np.maximum(amt, 1))
        mkt_proxy = log_amt
    else:
        mkt_proxy = np.zeros(len(y))
    
    X_stack = np.column_stack([np.ones(len(y)), mkt_proxy.reshape(-1, 1)])
    
    # Winsorize
    y_series = pd.Series(y)
    y_winsorized = mad_winsorize(y_series[valid_mask]).values
    
    y_temp = y.copy()
    y_temp[valid_mask] = y_winsorized
    y_full = y_temp
    
    # neutralizing: remove log_amt exposure
    residual = ols_residual(y_full, X_stack)
    
    # z-score
    valid2 = np.isfinite(residual)
    if valid2.sum() > 0:
        mu    = residual[valid2].mean()
        sigma = residual[valid2].std()
        if sigma > 0:
            factor_vals.loc[idx] = (residual - mu) / sigma
        else:
            factor_vals.loc[idx] = residual

panel_final["factor_value"] = factor_vals
panel_final = panel_final.dropna(subset=["factor_value"])

# Also compute log_amt for future talks
amt_safe = panel_final["amount"].clip(lower=1)
panel_final["log_amount"] = np.log(amt_safe)

print(f"  有效截面日期数: {panel_final['date'].nunique()}")
print(f"  因子值有记录: {panel_final['factor_value'].notna().sum():,}")

# ─────────────────────────────────────────
# 6. 输出 factor CSV
# ─────────────────────────────────────────
out_path = OUTPUT_DIR / "factor_roe_growth_v1.csv"
out = panel_final[["date", "stock_code", "factor_value", "roe_growth"]].copy()
out["date"] = out["date"].dt.strftime("%Y-%m-%d")
out["stock_code"] = out["stock_code"].astype(str)
out.to_csv(out_path, index=False)
print(f"\n因子值已保存: {out_path}")
print(f"  形状: {out.shape}")
print(f"  样本:\n{out.head()}")
