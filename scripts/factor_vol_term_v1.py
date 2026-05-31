"""
factor_vol_term_v1.py
---------------------
波动率期限结构因子 (Volatility Term-Structure)

构造逻辑
~~~~~~~~
vol_term_struct = std(ret, 5d) / std(ret, 20d)

低比值  →  短期波动率压缩于中期之下 → 低波收敛 → 蓄势/平静期
高比值  →  短期波动率飙升，突破中期均值 → 高波扩张 → 信息密集/恐慌

在中证1000上取正向（高短期/中期波动率比 → 后续有正alpha），对应：
高短期波动 = 知情事件密集释放，信息流动性增强 → 动量延续。

截面处理
~~~~~~~~
1. 对因子值做 MAD winsorize（5.2σ）
2. 对 log(amount)_20d 做 OLS 中性化
3. 残差 z-score
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

KLINE = "data/csi1000_kline_raw.csv"
OUT   = "data/factor_vol_term_v1.csv"

print("Loading kline …")
df = pd.read_csv(KLINE, parse_dates=["date"])
df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
df["ret"] = df.groupby("stock_code")["close"].pct_change()
df = df.dropna(subset=["ret"])

# 滚动波动率
df["std5"]  = df.groupby("stock_code")["ret"].transform(lambda s: s.rolling(5,  min_periods=4).std())
df["std20"] = df.groupby("stock_code")["ret"].transform(lambda s: s.rolling(20, min_periods=15).std())
df["log_amount_20d"] = (
    df.groupby("stock_code")["amount"]
    .transform(lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1))
)

df["vol_term"] = df["std5"] / (df["std20"] + 1e-8)
df = df.dropna(subset=["vol_term", "log_amount_20d"])

# 截面处理
results = []
for dt, panel in df.groupby("date"):
    x = panel[["stock_code", "vol_term", "log_amount_20d"]].copy()
    if len(x) < 30:
        continue
    med = x["vol_term"].median()
    mad = (x["vol_term"] - med).abs().median() * 1.4826
    if mad < 1e-8:
        continue
    lo, hi = med - 5.2 * mad, med + 5.2 * mad
    x["vol_term"] = x["vol_term"].clip(lo, hi)

    y = x["vol_term"].values
    X = np.column_stack([np.ones(len(x)), x["log_amount_20d"].values])
    try:
        beta  = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
    except Exception:
        continue

    s = resid.std()
    if s < 1e-8:
        continue
    z = (resid - resid.mean()) / s

    results.append(pd.DataFrame({
        "date": dt,
        "stock_code": x["stock_code"].values,
        "factor_vol_term_v1": z,
    }))

out = pd.concat(results, ignore_index=True)
out = out.sort_values(["date", "stock_code"]).reset_index(drop=True)
out.to_csv(OUT, index=False)
print(f"Done. {len(out)} rows → {OUT}")
print(f"Range: {out['date'].min()} ~ {out['date'].max()}")
print(f"Median stocks/day: {out.groupby('date')['stock_code'].count().median():.0f}")
print(out["factor_vol_term_v1"].describe())
