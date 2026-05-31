"""
factor_pras_v1.py
-----------------
日内收盘位置偏差因子 (Price Relative Asymmetry Signal)

定义: PRAS = ((high - close) - (close - low)) / (high - low + eps)
     也就是 (high + low - 2*close) / (high - low + eps)
     高值 = close 靠近日内底部（卖方主导）→ 反转做多
     低值 = close 靠近日内顶部（买方主导）→ 反转做空

构造: 20 日 PRAS 均值 → 量额 OLS 中性化 → MAD 缩尾 → z-score → 取反（做多 close 偏弱组）
"""

import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")

KLINE = "data/csi1000_kline_raw.csv"
OUT   = "data/factor_pras_v1.csv"
EPS   = 1e-8

print("Loading …")
df = pd.read_csv(KLINE, parse_dates=["date"])
df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

# 去重：取每个 (date, stock_code) 的第一条
df = df.drop_duplicates(subset=["date", "stock_code"], keep="first")

# PRAS := (high+low-2*close) / (high-low)
df["pras"] = (df["high"] + df["low"] - 2 * df["close"]) / ((df["high"] - df["low"]) + EPS)

# 20 日均量额中性化用标量
df["log_amount_20d"] = (
    df.groupby("stock_code")["amount"]
    .transform(lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1))
)

df["pras_m20"] = df.groupby("stock_code")["pras"].transform(
    lambda s: s.rolling(20, min_periods=15).mean())
df = df.dropna(subset=["pras_m20", "log_amount_20d"])

results = []
for dt, panel in df.groupby("date"):
    x = panel[["stock_code", "pras_m20", "log_amount_20d"]].copy()
    if len(x) < 50:
        continue
    y = x["pras_m20"].values
    X = np.column_stack([np.ones(len(x)), x["log_amount_20d"].values])
    try:
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X @ b          # 中性化残差
    except Exception:
        continue
    med = np.median(r)
    mad = np.median(np.abs(r - med)) * 1.4826
    if mad < 1e-8:
        continue
    r = np.clip(r, med - 5.2 * mad, med + 5.2 * mad)
    s = r.std()
    if s < 1e-8:
        continue
    z = (r - r.mean()) / s
    z = -z                    # 翻转：做多 close 偏弱（PRAS 高）的股票
    results.append(pd.DataFrame({
        "date": dt, "stock_code": x["stock_code"].values, "factor_pras_v1": z}))

out = pd.concat(results, ignore_index=True).drop_duplicates(["date", "stock_code"])
out = out.sort_values(["date", "stock_code"]).reset_index(drop=True)
out.to_csv(OUT, index=False)
print(f"Done. {len(out)} rows → {OUT}")
print(f"Date range : {out['date'].min()} ~ {out['date'].max()}")
print(f"Stocks/day : {out.groupby('date')['stock_code'].count().median():.0f}")
print(out["factor_pras_v1"].describe())
