#!/usr/bin/env python3
"""
因子: 日内方向确认率 (Intraday Direction Confirmation Rate) v1

定义: 过去20天中, gap方向(open vs prev_close)与日内走势方向(close vs open)
      同向的天数占比. 剔除gap或日内变动极小的天数(绝对值<0.001).

逻辑: 高确认率 = 集合竞价信号被日内交易确认 = 信息传播有效, 价格发现质量高
      低确认率 = 集合竞价与日内相反(gap被频繁回补) = 噪声交易多, 信息混乱

中性化: 成交额OLS中性化 + MAD winsorize + z-score

Barra风格: Quality/微观结构
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── 读取数据 ────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/csi1000_kline_raw.csv", parse_dates=["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

# ── 计算gap和日内方向 ──────────────────────────────────
print("Computing gap and intraday directions...")
df["prev_close"] = df.groupby("stock_code")["close"].shift(1)
df["gap"] = df["open"] - df["prev_close"]
df["intraday"] = df["close"] - df["open"]

# 方向确认: gap方向和日内方向同号(都涨或都跌)
# 剔除gap或intraday极小的情况(noise)
MIN_MOVE = 0.001  # 最小有意义变动
df["gap_sign"] = np.sign(df["gap"])
df["intra_sign"] = np.sign(df["intraday"])

# 标记有效天: gap和intraday绝对值都大于阈值
df["valid"] = (df["gap"].abs() > MIN_MOVE) & (df["intraday"].abs() > MIN_MOVE)
# 确认=同向
df["confirmed"] = (df["gap_sign"] == df["intra_sign"]) & df["valid"]

# ── 20日滚动确认率 ─────────────────────────────────────
WINDOW = 20
print(f"Computing {WINDOW}d rolling confirmation rate...")

def rolling_confirm_rate(group):
    """计算滚动确认率 = confirmed天数 / valid天数"""
    g = group.sort_values("date").copy()
    confirmed_sum = g["confirmed"].astype(float).rolling(WINDOW, min_periods=15).sum()
    valid_sum = g["valid"].astype(float).rolling(WINDOW, min_periods=15).sum()
    # 避免除以零
    rate = confirmed_sum / valid_sum.replace(0, np.nan)
    return rate

confirm_rates = df.groupby("stock_code", group_keys=False).apply(
    lambda g: rolling_confirm_rate(g)
)
df["confirm_rate"] = confirm_rates.values

# ── 成交额中性化 ───────────────────────────────────────
print("Neutralizing by log_amount_20d...")
df["log_amount"] = np.log1p(df["amount"])
df["log_amount_20d"] = df.groupby("stock_code")["log_amount"].transform(
    lambda x: x.rolling(20, min_periods=15).mean()
)

def neutralize_cross_section(group):
    """OLS成交额中性化"""
    y = group["confirm_rate"]
    x = group["log_amount_20d"]
    mask = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 30:
        return pd.Series(np.nan, index=group.index)
    from numpy.linalg import lstsq
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
out.to_csv("data/factor_gap_confirm_v1.csv", index=False)
print(f"Output: data/factor_gap_confirm_v1.csv")
print(f"  Records: {len(out):,}")
print(f"  Date range: {out['date'].min()} ~ {out['date'].max()}")
print(f"  Stocks per date (avg): {out.groupby('date')['stock_code'].nunique().mean():.0f}")

# 快速描述统计
print("\nFactor stats per date (mean of cross-section stats):")
for stat in ["mean", "std", "min", "max"]:
    val = out.groupby("date")["factor_value"].agg(stat).mean()
    print(f"  {stat}: {val:.4f}")

print("\nDone!")
