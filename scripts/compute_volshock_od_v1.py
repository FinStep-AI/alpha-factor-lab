#!/usr/bin/env python3
"""
因子: volshock_od_v1 — Volume-Shock × Overnight-Return-Direction

论文来源:
  Hameed / Tseng 2025 "Volume Shocks and Overnight Returns"
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5156605

核心思想 (本土化, Daily OHLCV only):
  日内放量 + 隔夜收益方向同向时, 次日收益预测力更强.
  论文本用 Intraday vs Overnight 拆解; 中证1000没有盘口拆解,
  这里用纯日频近似:
    ① vol_shock(t) = vol(t) / MA20(vol) — 1           (20日波动率波动)
    ② od_ret(t)     = close(t) / close(t-1) — 1        (隔夜收益, 用收盘价近似)
    ③ close_pos(t)  = (close - low) / (high - low + EPS)  (收盘位置)

  信号:
    raw = vol_shock(t-1) * sign(od_ret(t-1)) * (2 × close_pos(t-1) - 1)

  中性化: 用成交额(amount) 做 OLS 回归取残差, 再做 MAD 缩尾 + z-score.
  截面频率: 逐日滚动

输出: data/factor_volshock_od_v1.csv  [date | stock_code | factor_value]
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

EPS = 1e-9
DATA  = Path(__file__).parent.parent / "data"
LV_W  = 20   # 滚动窗口
WIN   = 3.0  # MAD 缩尾倍数
OUT   = DATA / "factor_volshock_od_v1.csv"

# ── 1. 加载数据 ──────────────────────────────────────────────────────────────
print("[1/4] 加载数据 …")
kline = pd.read_csv(DATA / "csi1000_kline_raw.csv", parse_dates=["date"])
kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)

# 日收益 = close / close_lag1 — 1
kline["close_lag1"] = kline.groupby("stock_code")["close"].shift(1)
kline["od_ret"] = kline["close"] / kline["close_lag1"] - 1.0

# ── 2. 滚动计算 vol_shock / close_pos / sign_od_ret ──────────────────────────
print("[2/4] 滚动计算信号 …")

def roll_group(g):
    g = g.sort_values("date").copy()
    # vol_shock = vol / MA20(vol) — 1
    ma = g["volume"].rolling(LV_W, min_periods=LV_W).mean()
    g["vol_shock"] = g["volume"] / (ma + EPS) - 1.0
    # 收盘位置
    rng = g["high"] - g["low"] + EPS
    g["close_pos"] = (g["close"] - g["low"]) / rng
    # sign(od_ret), 0 视为 0
    g["sign_od"] = np.sign(g["od_ret"])
    return g

kline = kline.groupby("stock_code", group_keys=False).apply(roll_group)
kline = kline.dropna(subset=["vol_shock", "od_ret", "close_pos", "sign_od", "amount"]).copy()

# 隔夜方向强度 (把 sign 量化, 避免极端值过强)
kline["od_dir"] = kline["sign_od"] * np.abs(kline["od_ret"]).clip(upper=0.10)

# close_pos_dev  =(close_pos - 0.5) * 2  => [-1, 1]
kline["cp_dev"] = (kline["close_pos"] - 0.5) * 2.0

# lag1
for col in ["vol_shock", "od_dir", "cp_dev"]:
    kline[col + "_lag1"] = kline.groupby("stock_code")[col].shift(1)

kline = kline.dropna(subset=["vol_shock_lag1", "od_dir_lag1", "cp_dev_lag1"]).copy()

# 原始因子
kline["factor_raw"] = (
    kline["vol_shock_lag1"] * kline["od_dir_lag1"] * kline["cp_dev_lag1"]
)

# ── 3. 成交额中性化 + MAD + z-score ─────────────────────────────────────────
print("[3/4] 中性化 & 截面标准化 …")

def cross_section_neutral(g):
    ym = g[["factor_raw", "amount"]].dropna()
    ym = ym[ym["amount"] > 0]
    if len(ym) < 30 or ym["amount"].nunique() < 2:
        # amount 无效 → 用 rank 近似截面中性化
        ranks = ym["factor_raw"].rank(method="average")
        n = len(ym) + 1e-9
        z = (ranks - (n + 1) / 2) / ((n**2 - 1) / 12 + EPS)
        z = np.clip(z, -WIN, WIN)
        std = z.std()
        if std < EPS:
            g.loc[ym.index, "factor_value"] = 0.0
        else:
            g.loc[ym.index, "factor_value"] = (z - z.mean()) / std
        g.loc[~g.index.isin(ym.index), "factor_value"] = np.nan
        return g
    y = ym["factor_raw"].values
    x = ym["amount"].values
    try:
        slope, intercept, _, _, _ = sp_stats.linregress(x, y)
    except Exception:
        g["factor_value"] = np.nan
        return g
    resid = y - (intercept + slope * x)

    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) + EPS
    z = (resid - med) / (1.4826 * mad)              # MAD z-score
    z = np.clip(z, -WIN, WIN)
    std = z.std()
    if std < EPS:
        g.loc[ym.index, "factor_value"] = 0.0
    else:
        g.loc[ym.index, "factor_value"] = (z - z.mean()) / std
    g.loc[~g.index.isin(ym.index), "factor_value"] = np.nan
    return g

kline = kline.groupby("date", group_keys=False).apply(cross_section_neutral)
kline = kline.dropna(subset=["factor_value"])

# ── 4. 输出 ──────────────────────────────────────────────────────────────────
print("[4/4] 写入 CSV …")
out = kline[["date", "stock_code", "factor_value"]].copy()
out["date"] = out["date"].dt.strftime("%Y-%m-%d")
out = out.sort_values(["date", "stock_code"]).reset_index(drop=True)
out.to_csv(OUT, index=False)

print(f"  行数={len(out)}  日期跨度={out['date'].min()} ~ {out['date'].max()}")
print(f"  平均截面股票数={out.groupby('date')['stock_code'].count().mean():.0f}")
print(f"  输出: {OUT}")
