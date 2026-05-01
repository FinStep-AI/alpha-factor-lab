#!/usr/bin/env python3
"""
close_position_v1: close_t / MA20 - 1 的截面因
高价: close > MA20 -> close大于均线 = 趋势延续动量
low: close < MA20
田: log(close / MA20)
成交额 OLS neutralization + MAD + z-score
"""

import pandas as pd, numpy as np
from numpy.linalg import lstsq
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

DATA_DIR   = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab/data")
OUTPUT_DIR = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab")
WINDOW_MA  = 20

print("=== close_position_v1: close%MA20 ===")

kline = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv")
kline["date"] = pd.to_datetime(kline["date"])
kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)

# MA20
kline["ma20"] = kline.groupby("stock_code")["close"].transform(
    lambda x: x.rolling(WINDOW_MA, min_periods=int(WINDOW_MA*0.7)).mean()
)

# close_position = close/ma20 - 1
kline["close_pos_raw"] = kline["close"] / kline["ma20"] - 1.0

# 20日平均成交额 (对数)
kline["log_amount_20d"] = kline.groupby("stock_code")["amount"].transform(
    lambda x: np.log1p(x.rolling(WINDOW_MA, min_periods=int(WINDOW_MA*0.7)).mean())
)

valid = kline.dropna(subset=["close_pos_raw", "log_amount_20d"]).copy()
print(f"有效截面样本: {len(valid)} 行, 日期={valid.date.min().date()}~{valid.date.max().date()}")

# 截面OLS neutralization
def cs_ols(df, y_col, x_col):
    res = []
    for dt, g in df.groupby("date", sort=True):
        y, x = g[y_col].values.astype(float), g[x_col].values.astype(float)
        ok = np.isfinite(y) & np.isfinite(x)
        if ok.sum() < 30:
            continue
        X = np.column_stack([np.ones(ok.sum()), x[ok]])
        bet, _, _, _ = lstsq(X, y[ok], rcond=None)
        resid = np.full(len(g), np.nan)
        resid[ok] = y[ok] - X @ bet
        for idx, sc, rv in zip(g.index, g["stock_code"].values, resid):
            res.append({"date": dt, "stock_code": sc, "neutralized": float(rv) if np.isfinite(rv) else np.nan})
    return pd.DataFrame(res)

print("[2] 截面OLS neutralization...")
neut = cs_ols(valid, "close_pos_raw", "log_amount_20d").dropna(subset=["neutralized"])
print(f"  中性化完成: {len(neut)} 行")

# MAD winsorize + z-score
def mad_winsorize(s, n=5.0):
    med = s.median(); mad = (s - med).abs().median() * 1.4826
    if mad < 1e-10: return s
    return s.clip(med - n*mad, med + n*mad)

print("[3] MAD winsorize + z-score...")
rows = []
for dt, g in neut.groupby("date", sort=True):
    n = len(g)
    if n < 30: continue
    win = mad_winsorize(g["neutralized"], 5.0)
    mu, sd = win.mean(), win.std()
    if sd < 1e-10: continue
    z = (win - mu) / sd
    for sc, fv in zip(g["stock_code"].values, z.values):
        rows.append({"date": dt.date().isoformat(), "stock_code": int(sc),
                     "factor_value": round(float(fv), 6)})

final = pd.DataFrame(rows).sort_values(["date", "stock_code"]).reset_index(drop=True)
out = OUTPUT_DIR / "data/factor_close_ma20_v1.csv"
final.to_csv(out, index=False)
print(f"\n✅ 因子写入: {out}")
print(f"  行={len(final)}, 股票={final.stock_code.nunique()}, 天={final.date.nunique()}")
print(f"  均值={final.factor_value.mean():.4f}, 标准差={final.factor_value.std():.4f}")
