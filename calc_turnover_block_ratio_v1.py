#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turnover Block Ratio (TBR) v1 — 换手率断点集中度

定义: max(turnover_20d) / MA(turnover_20d)
      衡量过去20日内最高换手日的相对放大程度

逻辑:
  成交极度集中于某几日 => 短期活跃日信号
  高 => 说明在极个别交易日换手率大幅超出常态 = 特定事件驱动
  极低 (<1.0): 换手率高度均衡 => 无事件
  极高 (>2.5-3.0): 异常放量交易日集中
  
  此因子与 turnover_level(水平)、turnover_decel(减速)、vol_cv_neg(CV稳定性)
  角度不同: 这三个是统计 moments(均值/趋势/分布形状), 
  TBR 是距离感知的(Wasserstein-like)波动率代理

公式: factor_raw = max(turnover,20) / ma20(turnover)
构造: 成交额OLS中性化 + MAD winsorize + z-score
方向: 待定(先单侧测试)

引用: 灵感来自 GMM / block effect 文献
      Barndorff-Nielsen & Shephard "Econometrics of Testing for jumps in economic"
      (但中国A股换手率集中度聚焦在市场日内信息扩散)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
KLINE_FILE = DATA_DIR / "csi1000_kline_raw.csv"
OUTPUT_FILE = DATA_DIR / "factor_tbr_v1.csv"
WINDOW = 20

def mad_zscore(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if mad < 1e-8:
        return s * 0.0
    return ((s - med) / mad).clip(-5, 5)

print("=== Turnover Block Ratio (TBR) v1 ===")
print(f"读取: {KLINE_FILE}")
df = pd.read_csv(KLINE_FILE, parse_dates=["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
df = df.dropna(subset=["date", "stock_code", "turnover", "amount"])

print(f"股票池: {df['stock_code'].nunique()} 只, 日期 {df['date'].min()} ~ {df['date'].max()}")

# ---- 因子计算 ----
print("计算因子...")
results = []  # type: ignore

grouped = df.groupby("stock_code")
for code, g in grouped:
    g = g.sort_values("date").copy()

    # rolling max / mean
    tbr = g["turnover"].rolling(WINDOW, min_periods=15).max() / \
          g["turnover"].rolling(WINDOW, min_periods=15).mean()

    g["tbr_raw"] = tbr
    # 成交额代理 (本因子用 amount 中性化)
    g["log_amount"] = np.log1p(g["amount"].clip(lower=1))

    df_code = g.dropna(subset=["tbr_raw", "log_amount"]).copy()
    if len(df_code) < 30:
        continue

    # 截面OLS中性化
    xs = df_code["log_amount"].values
    ys = df_code["tbr_raw"].values
    X = np.column_stack([np.ones(len(xs)), xs])
    try:
        beta = np.linalg.lstsq(X, ys, rcond=None)[0]
        resid = ys - X @ beta
        df_code["tbr_neutral"] = resid
    except Exception:
        df_code["tbr_neutral"] = np.nan

    results.append(df_code[["date", "stock_code", "tbr_raw", "tbr_neutral"]])

print("合并截面...")
out = pd.concat(results, ignore_index=True).dropna(subset=["tbr_neutral"])

print("MAD winsorize → z-score (by cross-section)...")
out["factor_value"] = out.groupby("date")["tbr_neutral"].transform(mad_zscore)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
out[["date", "stock_code", "factor_value"]].to_csv(OUTPUT_FILE, index=False)

ndates = out["date"].nunique()
nstocks_mean = out.groupby("date")["stock_code"].count().mean()
print(f"\n✅ 完成")
print(f"  截面天数: {ndates}")
print(f"  均持仓: {int(nstocks_mean)}")
print(f"  输出: {OUTPUT_FILE}")
print(f"  描述统计:\n{out['factor_value'].describe().to_string()}")
