#!/usr/bin/env python3
"""
因子 vwsr_v1: Volume-Weighted Signed Return
构造: 20日窗口内成交量加权价格方向一致性
公式: MA20(sign(ret) * volume / MA20(|volume|))
理论: Campbell, Grossman & Wang (1993) JFE
  - 高成交量+正收益 → 动量延续
  - 高成交量+负收益 → 短期反转
  
优化设计:
  - 方向信号 = sign(ret) (1/-1)
  - 成交量加权 = sign(ret) * volume / MA20(|volume|)
  - 20日均值平滑 = ts_mean(方向信号, 20d)
  - 最终 = neutralize(ts_mean(...), log_amount_20d)
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data"
OUTPUT_PATH = DATA_PATH / "factor_vwsr_v1.csv"

df = pd.read_csv(DATA_PATH / "csi1000_kline_raw.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

# returns
df["ret"] = df.groupby("stock_code")["close"].pct_change()

# sign(ret): 1 if up, -1 if down, 0 if flat
df["dir"] = np.sign(df["ret"])

# amplitude-weighted volume: dir * volume / MA20(|volume|)
# This normalizes volume by recent average to detect "abnormal" signed volume
df["abs_vol_ma20"] = df.groupby("stock_code")["volume"].transform(
    lambda x: x.abs().rolling(20, min_periods=10).mean()
)
df["signed_vol_ratio"] = np.where(
    df["abs_vol_ma20"] > 0,
    df["dir"] * df["volume"] / df["abs_vol_ma20"],
    np.nan
)

# 20d rolling mean of signed volume ratio
df["factor_raw"] = df.groupby("stock_code")["signed_vol_ratio"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

# 市值代理：成交额20日均值
df["amount_ma20"] = df.groupby("stock_code")["amount"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

# 求输出：date, stock_code, factor_raw, amount_ma20
out = df[["date", "stock_code", "factor_raw", "amount_ma20"]].copy()
out["date"] = out["date"].dt.strftime("%Y-%m-%d")
out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"[OK] Factor {OUTPUT_PATH}")
print(f"  shape: {out.shape}")
print(f"  factor_raw stats: mean={out['factor_raw'].mean():.4f}, std={out['factor_raw'].std():.4f}")
print(f"  non-null: {out['factor_raw'].notna().sum()}")
