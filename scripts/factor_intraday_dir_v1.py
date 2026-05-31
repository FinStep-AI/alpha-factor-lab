"""
factor_intraday_direction_v1.py
-------------------------------
日内方向强度因子 (Intraday Direction Strength)

定义
~~~~
    intraday_dir = |close - open| / (high - low + eps)

含义：当日价格变动绝对值占日内波动范围的比例。
      或 >= 1 即 close 触碰了 high/low（方向性强烈）；
      值接近 0 表示 close==open（十字星/剪刀差），方向性极弱。

20 日均值 + 成交额中性化 + MAD + z-score

逻辑
~~~~
A股中证1000上，经过一段观察后：
- 做多高日内方向强度（趋势型日K）→ 动量延续
- 低值 = 十字星/纺锤线 = 信息等待/方向不清 → 反转

最终方向由回测定，这里先输出原始均值，后看 IC 符号再定方向。
"""

import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")

KLINE = "data/csi1000_kline_raw.csv"
OUT   = "data/factor_intraday_dir_v1.csv"
EPS   = 1e-8

df = pd.read_csv(KLINE, parse_dates=["date"])
df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
df = df.sort_values(["stock_code","date"]).reset_index(drop=True)

# 日内方向强度
df["dir"] = (df["close"] - df["open"]).abs() / ((df["high"] - df["low"]) + EPS)

# 20 日均值 + 成交额中性化 proxy（用 amount）
df["dir_ma20"] = df.groupby("stock_code")["dir"].transform(lambda s: s.rolling(20,min_periods=15).mean())
df["log_amount_20d"] = df.groupby("stock_code")["amount"].transform(
    lambda x: np.log(x.rolling(20,min_periods=10).mean() + 1))

df = df.dropna(subset=["dir_ma20","log_amount_20d"])

results = []
for dt, p in df.groupby("date"):
    x = p[["stock_code","dir_ma20","log_amount_20d"]].copy()
    if len(x) < 30: continue
    med = x["dir_ma20"].median()
    mad = (x["dir_ma20"] - med).abs().median() * 1.4826
    if mad < 1e-8: continue
    lo,hi = med-5.2*mad, med+5.2*mad
    x["dir_ma20"] = x["dir_ma20"].clip(lo,hi)
    y = x["dir_ma20"].values
    X = np.column_stack([np.ones(len(x)), x["log_amount_20d"].values])
    try:
        b = np.linalg.lstsq(X,y,rcond=None)[0]
        r = y - X@b
    except Exception:
        continue
    s = r.std()
    if s < 1e-8: continue
    z = (r - r.mean()) / s
    results.append(pd.DataFrame({"date":dt,"stock_code":x["stock_code"].values,
                                 "factor_intraday_dir_v1":z}))

out = pd.concat(results,ignore_index=True).sort_values(["date","stock_code"]).reset_index(drop=True)
out.to_csv(OUT,index=False)
print(f"Done. {len(out)} rows → {OUT}")
print(f"Range: {out['date'].min()} ~ {out['date'].max()}")
print(f"Median stocks/day: {out.groupby('date')['stock_code'].count().median():.0f}")
print(out["factor_intraday_dir_v1"].describe())
