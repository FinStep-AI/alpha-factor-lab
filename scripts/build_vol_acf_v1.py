#!/usr/bin/env python3
"""
因子: volume_acf_v1 (成交量自相关因子)
构造: 过去20日成交量的一阶自相关系数（lag-1 ACF）
      衡量成交量序列的记忆性 / 持续性
      高自相关=成交量有持续趋势=有连续资金介入=信息流稳定
      低自相关=成交量随机分散=噪音交易主导

逻辑来源: Chordia, Roll & Subrahmanyam (2001) "Market Liquidity and Trading Activity",
          JOF 通过对交易量自相关性的研究捕获市场信息效率。

bias方向: 与既有的 turnover_level（换手率水平）冗不相关，捕捉的是流量的动态持续性而非水平。
"""

import sys
sys.path.insert(0, "skills/alpha-factor-lab/scripts")

import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------------------------------------
# 路径与数据加载
# -----------------------------------------------------------
BASE = Path(__file__).parent
WORKSPACE = BASE.parent
DATA = WORKSPACE / "data"
KLINE_FILE = DATA / "csi1000_kline_raw.csv"
RETURNS_FILE = DATA / "csi1000_returns.csv"

print(f"[1/5] Loading K-line data: {KLINE_FILE.name}")
df = pd.read_csv(KLINE_FILE, parse_dates=["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
print(f"     Shape: {df.shape}, date range: {df['date'].min()} ~ {df['date'].max()}")
print(f"     Columns: {list(df.columns)}")

# -----------------------------------------------------------
# 衍生字段计算
# -----------------------------------------------------------
print("[2/5] Computing derived fields...")

# 20日成交量序列（自然对数）
df["vol_log"] = np.log(df["volume"] + 1)

# 前1日成交量（用于滞后差分）
df["vol_log_lag1"] = df.groupby("stock_code")["vol_log"].shift(1)

# 20日滚动均值/标准差
df["vol_log_mean20"] = df.groupby("stock_code")["vol_log"].transform(
    lambda s: s.rolling(20, min_periods=15).mean()
)
df["vol_log_std20"] = df.groupby("stock_code")["vol_log"].transform(
    lambda s: s.rolling(20, min_periods=15).std()
)

# 20日成交量的一阶自相关系数（简化版: corr(vol[t], vol[t-1]) 在20日窗口内）
# 使用 rolling corr: 先算每天的 corr，再20日平均
def rolling_vol_acf(group, window=20, lag=1):
    """计算组内滚动lag-1自相关"""
    s = group["vol_log"].values
    result = np.full(len(s), np.nan)
    for i in range(window - 1 + lag, len(s)):
        seg_x = s[i - window + 1:i + 1]
        seg_y = s[i - window + 1 - lag:i + 1 - lag]
        mask = ~(np.isnan(seg_x) | np.isnan(seg_y))
        if mask.sum() >= 10:
            result[i] = np.corrcoef(seg_x[mask], seg_y[mask])[0, 1]
    return pd.Series(result, index=group.index)

print("[3/5] Computing 20d lag-1 volume ACF (this may take ~60s)...")
acf_series = (
    df.groupby("stock_code", group_keys=False)
    .apply(rolling_vol_acf, window=20, lag=1)
)
df["vol_acf_20d"] = acf_series
print(f"     vol_acf_20d stats:\n{df['vol_acf_20d'].describe()}")

# -----------------------------------------------------------
# 成交额中性化
# -----------------------------------------------------------
print("[4/5] Neutralizing (amount_20d)...")
df["amount_20d"] = df.groupby("stock_code")["amount"].transform(
    lambda s: s.rolling(20, min_periods=15).mean()
)
df["log_amount_20d"] = np.log(df["amount_20d"].clip(1))

# 截面 OLS 中性化
from numpy.linalg import lstsq

def neutralize_cross_section(raw: pd.Series, neutralizer: pd.Series) -> pd.Series:
    """截面OLS: raw ~ 1 + neutralizer, 返回残差"""
    out = pd.Series(np.nan, index=raw.index)
    for dt, grp in raw.groupby(level=0 if raw.index.nlevels > 1 else raw.index):
        pass
    # 按日期分组中性化
    valid = ~(raw.isna() | neutralizer.isna())
    dates = df.loc[valid, "date"].unique()
    for d in sorted(dates):
        mask = (df["date"] == d) & valid
        if mask.sum() < 30:
            continue
        x = neutralizer[mask].values
        y = raw[mask].values
        ones = np.ones(len(x))
        X = np.column_stack([ones, x])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
        out.loc[mask] = residuals
    return out

# 每天截面中性化
print("     Cross-sectional OLS neutralize on log_amount_20d ...")
raw = df["vol_acf_20d"]
neutralizer = df["log_amount_20d"]

raw_vals = raw.values
neu_vals = neutralizer.values
dates = df["date"].values
out_vals = np.full(len(raw_vals), np.nan)

unique_dates = sorted(pd.unique(dates))
for d in unique_dates:
    mask = dates == d
    r_sub = raw_vals[mask]
    n_sub = neu_vals[mask]
    non_na = ~(np.isnan(r_sub) | np.isnan(n_sub))
    if non_na.sum() < 30:
        continue
    x = n_sub[non_na].reshape(-1, 1)
    y = r_sub[non_na]
    X = np.column_stack([np.ones(len(x)), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = np.full(len(r_sub), np.nan)
    residuals[non_na] = y - X @ beta
    out_vals[mask] = residuals

df["vol_acf_neutral"] = out_vals

# MAD winsorize + z-score
def mad_zscore(arr):
    valid = arr[~np.isnan(arr)]
    if len(valid) < 10:
        return arr
    med = np.median(valid)
    mad = np.median(np.abs(valid - med)) + 1e-8
    upper = med + 5.2 * 1.4826 * mad
    lower = med - 5.2 * 1.4826 * mad
    arr = np.clip(arr, lower, upper)
    mu, sigma = np.nanmean(arr), np.nanstd(arr) + 1e-8
    return (arr - mu) / sigma

df["vol_acf_factor"] = df.groupby("date")["vol_acf_neutral"].transform(
    lambda x: mad_zscore(x.values) if len(x) > 10 else x.values
)

print(f"     vol_acf_factor stats:\n{df['vol_acf_factor'].describe()}")

# -----------------------------------------------------------
# 输出因子CSV
# -----------------------------------------------------------
OUT_FILE = DATA / "factor_volume_acf_v1.csv"
out_df = df[["date", "stock_code", "vol_acf_factor"]].dropna(subset=["vol_acf_factor"])
out_df = out_df.rename(columns={"vol_acf_factor": "factor_value"})
out_df["date"] = out_df["date"].astype(str)
out_df.to_csv(OUT_FILE, index=False)
print(f"[5/5] ✅  Factor CSV saved: {OUT_FILE}  ({len(out_df)} rows, {out_df['date'].nunique()} dates)")
print(f"     Latest date: {out_df['date'].max()}, stocks per date: {out_df.groupby('date')['stock_code'].count().median():.0f}")
