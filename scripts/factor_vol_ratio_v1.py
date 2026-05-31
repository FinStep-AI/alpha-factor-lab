"""
factor_vol_ratio_v1.py
----------------------
短-长波动率比值因子 (Short-Long Volatility Ratio)

构造
~~~~
    vol_ratio = std(ret, 5d) / std(ret, 40d)

高值 = 近期波动率飙升 / 长期波动率相对稳定 → 短期信息密集
低值 = 近期波动率压制 / 长期波动率偏高 → 平静 / 蓄势

截面操作
~~~~~~~
对 vol_ratio 原始值：
  1. MAD winsorize（5.2σ）
  2. OLS 中性化 on log(amount_20d) → 残差
  3. 残差 z-score

此构造与 vol_log60d_v4（长期绝对水平）和 vol_term_v1（5d/20d）
在量纲和窗口上都不同，预计相关性较低。
"""

import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")

KLINE = "data/csi1000_kline_raw.csv"
OUT   = "data/factor_vol_ratio_v1.csv"

df = pd.read_csv(KLINE, parse_dates=["date"])
df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
df = df.sort_values(["stock_code","date"]).reset_index(drop=True)
df["ret"] = df.groupby("stock_code")["close"].pct_change()
df = df.dropna(subset=["ret"])

df["std5"]  = df.groupby("stock_code")["ret"].transform(lambda s: s.rolling(5,  min_periods=4).std())
df["std40"] = df.groupby("stock_code")["ret"].transform(lambda s: s.rolling(40, min_periods=30).std())
df["log_amount_20d"] = df.groupby("stock_code")["amount"].transform(
    lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1))

df["vol_ratio"] = df["std5"] / (df["std40"] + 1e-8)
df = df.dropna(subset=["vol_ratio","log_amount_20d"])

results = []
for dt, p in df.groupby("date"):
    x = p[["stock_code","vol_ratio","log_amount_20d"]].copy()
    if len(x) < 30: continue
    med = x["vol_ratio"].median()
    mad = (x["vol_ratio"] - med).abs().median() * 1.4826
    if mad < 1e-8: continue
    x["vol_ratio"] = x["vol_ratio"].clip(med - 5.2*mad, med + 5.2*mad)
    y = x["vol_ratio"].values
    X = np.column_stack([np.ones(len(x)), x["log_amount_20d"].values])
    try:
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X @ b
    except Exception:
        continue
    s = r.std()
    if s < 1e-8: continue
    z = (r - r.mean()) / s
    results.append(pd.DataFrame({"date":dt,"stock_code":x["stock_code"].values,
                                 "factor_vol_ratio_v1":z}))

out = pd.concat(results, ignore_index=True).sort_values(["date","stock_code"]).reset_index(drop=True)
out.to_csv(OUT, index=False)
print(f"Done. {len(out)} rows → {OUT}")
print(f"Range: {out['date'].min()} ~ {out['date'].max()}")
print(f"Stocks/day (median): {out.groupby('date')['stock_code'].count().median():.0f}")
print(out["factor_vol_ratio_v1"].describe())
