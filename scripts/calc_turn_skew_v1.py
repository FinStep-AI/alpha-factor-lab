"""
因子: 换手率偏度 (Turnover Skewness, turn_skew_v1)
向量化版本，避免逐行循环
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
kline = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv", parse_dates=["date"])
kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)
print(f"数据: {kline.shape[0]} 行, {kline['stock_code'].nunique()} 只股票")

# ── 20日滚动换手率偏度（向量化）──────────────────────────
window = 20
min_periods = 12

def rolling_skew(group):
    group["turn_skew_raw"] = group["turnover"].rolling(window, min_periods=min_periods).skew()
    return group

print("计算换手率偏度 (20d滚动, 向量化)...")
kline = kline.groupby("stock_code", group_keys=False).apply(rolling_skew)
valid_count = kline["turn_skew_raw"].notna().sum()
print(f"  非空因子值: {valid_count} / {len(kline)}")

# ── 成交额中性化 ────────────────────────────────────────
kline["log_amount_20d"] = kline.groupby("stock_code")["amount"].transform(
    lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
)

def neutralize_ols(df):
    mask = df["turn_skew_raw"].notna() & df["log_amount_20d"].notna() & np.isfinite(df["turn_skew_raw"])
    if mask.sum() < 30:
        df["factor"] = np.nan
        return df
    y = df.loc[mask, "turn_skew_raw"].values
    x = df.loc[mask, "log_amount_20d"].values
    X = np.column_stack([np.ones(len(x)), x])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
    except:
        df["factor"] = np.nan
        return df
    df["factor"] = np.nan
    df.loc[mask, "factor"] = resid
    return df

print("截面中性化...")
kline = kline.groupby("date", group_keys=False).apply(neutralize_ols)

# ── MAD winsorize + z-score ──────────────────────────────
def mad_zscore(df):
    vals = df["factor"]
    mask = vals.notna()
    if mask.sum() < 30:
        df["factor"] = np.nan
        return df
    med = vals[mask].median()
    mad = (vals[mask] - med).abs().median() * 1.4826
    if mad < 1e-10:
        df["factor"] = np.nan
        return df
    lower = med - 3 * mad
    upper = med + 3 * mad
    clipped = vals.clip(lower, upper)
    mu = clipped[mask].mean()
    sigma = clipped[mask].std()
    if sigma < 1e-10:
        df["factor"] = np.nan
        return df
    df["factor"] = (clipped - mu) / sigma
    return df

print("MAD winsorize + z-score...")
kline = kline.groupby("date", group_keys=False).apply(mad_zscore)

output = kline[["date", "stock_code", "factor"]].dropna(subset=["factor"])
output = output.rename(columns={"factor": "turn_skew_v1"})
out_path = DATA_DIR / "factor_turn_skew_v1.csv"
output.to_csv(out_path, index=False)
print(f"\n输出: {out_path}")
print(f"  行数: {len(output)}")
print(f"  分布: mean={output['turn_skew_v1'].mean():.4f}, std={output['turn_skew_v1'].std():.4f}")
