#!/usr/bin/env python3
"""
因子：信号一致性 v1 (Binary Signal Consistency)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
从短期收益方向一致性与成交量方向一致性的角度改进 pv_corr_v1。

并非对每日收益率滚动 signal_consistency = sign(ret) × sign(vol_chg)
取20日均值，稳定反映量价同向信号的有效性。

阳性值 = 信号方向一致 = 量价信息同步 = 预期更高收益
构造逻辑：Brennan, Chordia & Subrahmanyam (1998) 量价协方差因子的一种二进制近似变体。

操作顺序：
  1. sign_ret × sign_vol → 每日信号一致性指标（二进制，∈{-1,0,1}）
  2. 20日滚动均值（信号连续性）
  3. 成交额OLS中性化
  4. MAD winsorize (3σ)
  5. z-score截面标准化

因子名称：bin_consistency_v1

Barra风格: MICRO（与 pv_corr_v1 同属极致，但角度互补：后者幅度相关 vs 本因子方向一致）
"""

import pandas as pd
import numpy as np
import sys, os
from scipy import stats

def winsorize_mad(series, n_mad=3.0):
    """MAD方法去极值"""
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return series
    scaled_mad = 1.4826 * mad
    lower = med - n_mad * scaled_mad
    upper = med + n_mad * scaled_mad
    return series.clip(lower, upper)

def ols_residual(y, X):
    """OLS残差"""
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < X.shape[1] + 2:
        return y
    y_clean = y[mask]
    X_clean = X[mask]
    try:
        XtX = X_clean.T @ X_clean
        Xty = X_clean.T @ y_clean
        beta = np.linalg.solve(XtX + 1e-10 * np.eye(XtX.shape[0]), Xty)
        residual = np.full_like(y, np.nan)
        residual[mask] = y_clean - X_clean @ beta
        return residual
    except np.linalg.LinAlgError:
        return y

print("=== 因子挖掘：bin_consistency_v1 ===", flush=True)
DATA = "/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab/data/csi1000_kline_raw.csv"

# ── 1. 加载数据 ────────────────────────────────────────────────────────────────
print("[1/5] 加载数据 ...", flush=True)
df = pd.read_csv(DATA, parse_dates=["date"])
print(f"原始: {df.shape}，{df['stock_code'].nunique()} 只股票，"
      f"{df['date'].min()} ~ {df['date'].max()}", flush=True)

df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

# ── 2. 计算基础字段 ────────────────────────────────────────────────────────────
print("[2/5] 计算信号一致性 ...", flush=True)
g = df.groupby("stock_code")
df["ret"] = g["close"].pct_change()
df["vol_chg"] = g["volume"].pct_change()

# sign(ret) × sign(vol_chg) → ∈{-1,0,1}，二进制方向一致性
df["sign_ret"] = np.sign(df["ret"])
df["sign_vol"] = np.sign(df["vol_chg"])
df["sign_product"] = (df["sign_ret"] * df["sign_vol"]).clip(-1, 1)

# 20日滚动均值（连续交易的成本与贡献）
df["factor_raw"] = g["sign_product"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

valid = df["factor_raw"].notna().sum()
total = len(df)
print(f"  原始因子有效值: {valid}/{total} ({valid/total*100:.1f}%)", flush=True)

# ── 3. 成交额中性化 ────────────────────────────────────────────────────────────
print("[3/5] 成交额OLS中性化 ...", flush=True)

df["log_amount_20d"] = g["amount"].transform(
    lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
)

resid_list = []  # list of (date_list, resid_list) to reconstruct
df["_resid"] = np.nan

for dt, grp in df.groupby("date", sort=False):
    sub = grp.copy()
    mask = sub["factor_raw"].notna() & sub["log_amount_20d"].notna()
    if mask.sum() < 100:
        df.loc[sub.index, "_resid"] = np.nan
        continue
    y = sub.loc[mask, "factor_raw"].values.astype(float)
    X = sub.loc[mask, ["log_amount_20d"]].values.astype(float)
    X = np.column_stack([np.ones(len(X)), X])
    resid = ols_residual(y, X)
    df.loc[sub.index[mask], "_resid"] = resid

# MAD Winsorize
df["_resid"] = df.groupby("date")["_resid"].transform(
    lambda x: winsorize_mad(x, n_mad=3.0)
)

# Z-score截面标准化
df["_z"] = df.groupby("date")["_resid"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-10)
)

# ── 4. 保存因子 + 做IC初验 ──────────────────────────────────────────────────────
print("[4/5] 保 MIC因子并做回测 ...", flush=True)

OUT_CSV = "data/factor_bin_consistency_v1.csv"
df[["date", "stock_code", "_z"]].rename(columns={"_z": "bin_consistency_v1"}).to_csv(OUT_CSV, index=False)
print(f"  已保存: {OUT_CSV}", flush=True)

# 读取收益率做短期IC初验
rtn = pd.read_csv("data/csi1000_returns.csv", parse_dates=["date"])
merged = df[["date","stock_code","_z"]].rename(columns={"_z":"factor"})\
    .merge(rtn.rename(columns={"return":"fwd_ret"}), on=["date","stock_code"], how="inner")

# 取5日内的fwd_ret
merged = merged.sort_values(["date","stock_code"]).reset_index(drop=True)
merged["fwd_ret_5d"] = merged.groupby("stock_code")["fwd_ret"].transform(
    lambda x: x.shift(-5) if len(x)>5 else x
)
valid = merged[["factor","fwd_ret_5d"]].dropna()
if len(valid) > 0:
    ic, pval = stats.spearmanr(valid["factor"], valid["fwd_ret_5d"])
    print(f"\n  [初步IC检验 | 5日前瞻]")
    print(f"  IC_mean ≈ {ic:.4f}  (p={pval:.4f})")
else:
    print("  警告: 有效合并数据不足，跳过IC初验")

print("\n[5/5] 因子构造完毕，执行因子生成完毕。", flush=True)
print(f"输出路径: {OUT_CSV}")
