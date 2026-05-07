#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: vol_skew_v1 — 波动率偏度因子 (Yield Skewness)
======================================================
方向: 波动率偏度（收益率分布尾部不对称）

构造:
  1. 过去20日日收益率的 Fisher-Pearson 偏度
  2. 成交额20日均值OLS中性化
  3. 5%截尾Winsorize

逻辑:
  正偏（右尾肥）的股票具有"彩票特征"——小概率大涨、大概率微跌。A股散户占比高，
  彩票偏好更强，高偏度股票常被散户追捧→高估→后续收益差。
  负偏（左尾肥）的股票频繁小涨偶尔大跌→风险暴露大→要求补偿→后续收益高。

理论:
  - Boyer, Mitton & Vorkink (2010) "Expected Idiosyncratic Skewness" RFS
  - Bali, Cakici & Whitelaw (2011) MAX Effect → JFE
  - Harvey & Siddique (2000) Conditional Skewness → JFE
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ────────────────── 参数 ──────────────────
WINDOW = 20
DATA_CUTOFF = "2026-05-01"
FACTOR_ID = "vol_skew_v1"

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = BASE_DIR.parent / "data" / "csi1000_kline_raw.csv"
SCRIPTS_DIR = BASE_DIR.parent / "skills" / "alpha-factor-lab" / "scripts"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output" / FACTOR_ID
FACTOR_CSV_PATH = Path(__file__).resolve().parent.parent.parent / "data" / f"factor_{FACTOR_ID}.csv"

# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据 (cutoff: {DATA_CUTOFF})...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] <= DATA_CUTOFF]
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")

ret_piv = close_piv.pct_change()
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股, 数据截至 {max(dates).strftime('%Y-%m-%d')}")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 计算 {WINDOW}日收益率偏度...")
factor_raw = ret_piv.rolling(WINDOW, min_periods=15).skew()

print(f"   非空率: {factor_raw.notna().mean().mean():.2%}")
print(f"   均值: {factor_raw.stack().mean():.4f}, std: {factor_raw.stack().std():.4f}")
print(f"   min:  {factor_raw.stack().min():.4f}, max: {factor_raw.stack().max():.4f}")

# ────────────────── 缩尾 ──────────────────
print(f"[3] 5%截尾Winsorize...")
for date in dates:
    row = factor_raw.loc[date].dropna()
    if len(row) < 10:
        continue
    lo = row.quantile(0.05)
    hi = row.quantile(0.95)
    factor_raw.loc[date] = factor_raw.loc[date].clip(lo, hi)

# ────────────────── 中性化 ──────────────────
print(f"[4] 成交额OLS中性化...")
factor_neutral = factor_raw.copy()
for date in dates:
    f = factor_raw.loc[date].dropna()
    m = log_amt.loc[date].reindex(f.index).dropna()
    common = f.index.intersection(m.index)
    if len(common) < 30:
        continue
    f_c = f[common].values
    m_c = m[common].values
    X = np.column_stack([np.ones(len(m_c)), m_c])
    try:
        beta = np.linalg.lstsq(X, f_c, rcond=None)[0]
        factor_neutral.loc[date, common] = f_c - X @ beta
    except Exception:
        pass

print(f"   中性化后非空率: {factor_neutral.notna().mean().mean():.2%}")

# ────────────────── 输出 ──────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
factor_daily = factor_neutral.copy()

# 输出到 CSV
rows = []
for date in factor_daily.index:
    s = factor_daily.loc[date].dropna()
    for code, val in s.items():
        rows.append({"date": date.strftime("%Y-%m-%d"), "stock_code": code, "factor_value": round(float(val), 6)})

out_df = pd.DataFrame(rows)


print(f"[5] 写入文件...")
factor_csv = FACTOR_CSV_PATH
out_df.to_csv(factor_csv, index=False)
print(f"   因子CSV: {factor_csv} ({len(rows)} 行)")

# 保存一些统计信息
stats_path = OUTPUT_DIR / "factor_stats.json"
with open(stats_path, "w") as f:
    json.dump({
        "factor_id": FACTOR_ID,
        "n_dates": len(dates),
        "n_stocks": len(stocks),
        "data_cutoff": DATA_CUTOFF,
        "non_null_rate": float(factor_daily.notna().mean().mean()),
        "raw_mean": float(factor_raw.stack().mean()),
        "raw_std": float(factor_raw.stack().std()),
        "neutral_mean": float(factor_daily.stack().mean()),
        "neutral_std": float(factor_daily.stack().std()),
        "rows_written": len(rows),
    }, f, indent=2, ensure_ascii=False)
print(f"   统计: {stats_path}")
print(f"\n[完成] 波动率偏度因子计算完毕 ✓")
