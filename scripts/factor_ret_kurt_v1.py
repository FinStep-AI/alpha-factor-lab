"""
factor_ret_kurt_v1.py
---------------------
收益率峰度因子 (Return Kurtosis)

构造逻辑
~~~~~~~~
对每只个股过去 20 个交易日的日收益率序列计算 Fisher 峰度（ excess kurtosis，
即已减去正态分布基准值3，使正态=0）。

   kurt_raw = E[(r - E[r])^4] / E[(r - E[r])^2]^2
   kurt_ex  = kurt_raw - 3          # Fisher excess kurtosis

截面处理
~~~~~~~~
1. 按截面做 MAD winsorize（阈值 5.2×MAD）
2. 对 log(amount)_20d 做 OLS 中性化，取残差
3. 残差再做一次 z-score → 最终因子值

直觉
~~~~
高正峰度 → 收益分布尖峰厚尾 → 极端收益(涨/跌)频繁出现 → 尾部信息密度高；
   在中证1000小盘股里通常伴随知情交易驱动的信息事件，其截面溢价可测。

Barra Style  : MICRO / 量价微观结构（独立于 Volatility 类振幅CV因子）
数据依赖   : 日线 OHLCV（已有 csi1000_kline_raw.csv）
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ─── 加载数据 ────────────────────────────────────────────────────────────────
KLINE  = "data/csi1000_kline_raw.csv"
AMOUNT = "data/csi1000_fundamental_cache.csv"  # not used here, kept for compat
OUT    = "data/factor_ret_kurt_v1.csv"

print("Loading kline …")
df = pd.read_csv(KLINE, parse_dates=["date"])
df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)

# 计算日收益率
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
df["ret"] = df.groupby("stock_code")["close"].pct_change()

# 保留有收益率的记录
df = df.dropna(subset=["ret"])

# ─── 滚动峰度（20日窗口，最小观测=15）───────────────────────────────────
WINDOW = 20
MIN_OBS = 15

def roll_kurt(g: pd.Series) -> pd.Series:
    return g.rolling(WINDOW, min_periods=MIN_OBS).kurt()

print("Computing rolling kurtosis …")
df["kurt"] = df.groupby("stock_code")["ret"].transform(roll_kurt)

# 20日成交额均值（用于中性化）
df["log_amount_20d"] = (
    df.groupby("stock_code")["amount"]
    .transform(lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1))
)

# 只保留峰度非 NaN 的行
df = df.dropna(subset=["kurt", "log_amount_20d"])

# ─── 截面处理：MAD winsorize → OLS中性化 → z-score ──────────────────────────
results = []
for dt, panel in df.groupby("date"):
    x = panel[["stock_code", "kurt", "log_amount_20d"]].copy()
    if len(x) < 30:
        continue

    # MAD winsorize
    med = x["kurt"].median()
    mad = (x["kurt"] - med).abs().median() * 1.4826
    if mad < 1e-8:
        continue
    lo, hi = med - 5.2 * mad, med + 5.2 * mad
    x["kurt"] = x["kurt"].clip(lo, hi)

    # OLS neutralize on log_amount_20d
    y = x["kurt"].values
    X = np.column_stack([np.ones(len(x)), x["log_amount_20d"].values])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
    except Exception:
        continue

    # z-score
    std = resid.std()
    if std < 1e-8:
        continue
    z = (resid - resid.mean()) / std

    tmp = pd.DataFrame({
        "date": dt,
        "stock_code": x["stock_code"].values,
        "factor_ret_kurt_v1": z,
    })
    results.append(tmp)

factor_df = pd.concat(results, ignore_index=True)
factor_df = factor_df.sort_values(["date", "stock_code"]).reset_index(drop=True)
factor_df.to_csv(OUT, index=False)

print(f"Done. {len(factor_df)} rows → {OUT}")
print(f"Date range : {factor_df['date'].min()} ~ {factor_df['date'].max()}")
print(f"Stocks/date : {factor_df.groupby('date')['stock_code'].count().median():.0f} (median)")
print(f"Factor stats:\n{factor_df['factor_ret_kurt_v1'].describe()}")
