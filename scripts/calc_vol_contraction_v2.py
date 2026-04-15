#!/usr/bin/env python3
"""
因子: 波动率收缩因子 (Volatility Contraction) v2

之前amp_compress_v1: -log(MA5_amp/MA20_amp) IC=0.022 t=2.06 mono=0.7 (待优化)

改进思路:
1. 用更长的基准窗口(40d/60d)代替20d, 让"正常水平"更稳定
2. 用realized vol(std of returns)代替振幅, 信号更纯粹
3. 复合: -log(std_5d / std_40d), 短期波动率收缩=蓄势突破

实际做: -log(realized_vol_5d / realized_vol_40d), 成交额中性化

高因子值 = 短期波动率远低于长期 = 波动率收缩/蓄势状态
低因子值 = 短期波动率远高于长期 = 波动率爆发

假说: 波动率收缩后的股票酝酿新趋势, 但方向未定 → 需看是否有偏向
      实测之前vol_term_structure(负log)失败了(IC=0.003 t=0.52)
      但那是5d/20d窗口, 5d/40d可能不同

更好方案: 直接做 std_5d / std_60d 的水平, 不取负 (让数据说话)
"""

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
import warnings
warnings.filterwarnings("ignore")

print("Loading data...")
df = pd.read_csv("data/csi1000_kline_raw.csv", parse_dates=["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

df["ret"] = df.groupby("stock_code")["close"].pct_change()

# ── 计算多窗口realized vol ──────────────────────────────
print("Computing rolling volatilities...")
for w in [5, 10, 20, 40, 60]:
    df[f"vol_{w}d"] = df.groupby("stock_code")["ret"].transform(
        lambda x: x.rolling(w, min_periods=max(w-2, 3)).std()
    )

# ── 波动率比率 ──────────────────────────────────────────
# 试多组组合
CONFIGS = {
    "vol_ratio_5_40": ("vol_5d", "vol_40d"),
    "vol_ratio_5_60": ("vol_5d", "vol_60d"),
    "vol_ratio_10_60": ("vol_10d", "vol_60d"),
}

for name, (short_col, long_col) in CONFIGS.items():
    df[name] = np.log(df[short_col] / df[long_col].replace(0, np.nan))

# 成交额
df["log_amount"] = np.log1p(df["amount"])
df["log_amount_20d"] = df.groupby("stock_code")["log_amount"].transform(
    lambda x: x.rolling(20, min_periods=15).mean()
)

def neutralize_cs(group, col):
    y = group[col]
    x = group["log_amount_20d"]
    mask = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 30:
        return pd.Series(np.nan, index=group.index)
    X = np.column_stack([np.ones(mask.sum()), x[mask].values])
    beta, _, _, _ = lstsq(X, y[mask].values, rcond=None)
    resid = pd.Series(np.nan, index=group.index)
    resid[mask] = y[mask].values - X @ beta
    return resid

def mad_win(group, col, n_mad=5):
    vals = group[col]
    median = vals.median()
    mad = (vals - median).abs().median()
    if mad < 1e-10:
        return vals
    lower = median - n_mad * 1.4826 * mad
    upper = median + n_mad * 1.4826 * mad
    return vals.clip(lower, upper)

def zsc(group, col):
    vals = group[col]
    mu = vals.mean()
    std = vals.std()
    if std < 1e-10:
        return pd.Series(0.0, index=group.index)
    return (vals - mu) / std

# 处理所有变体
for name in CONFIGS:
    print(f"\nProcessing {name}...")
    df["f_raw"] = df.groupby("date", group_keys=False).apply(lambda g: neutralize_cs(g, name))
    df["f_win"] = df.groupby("date", group_keys=False).apply(lambda g: mad_win(g, "f_raw"))
    df["f_z"] = df.groupby("date", group_keys=False).apply(lambda g: zsc(g, "f_win"))
    
    out = df[["date", "stock_code", "f_z"]].dropna(subset=["f_z"])
    out = out.rename(columns={"f_z": "factor_value"})
    out.to_csv(f"data/factor_{name}_v1.csv", index=False)
    print(f"  Output: data/factor_{name}_v1.csv ({len(out):,} records)")
    print(f"  Date range: {out['date'].min()} ~ {out['date'].max()}")

print("\nDone!")
