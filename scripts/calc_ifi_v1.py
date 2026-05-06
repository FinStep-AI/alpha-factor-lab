
"""
Intraday Flow Imbalance (IFI) — 日内资金流不均衡因子
======================================================
日内资金流 = sign(close-open) × amount  (近似开盘→收盘的资金方向)
因子定义：20日滚动日内资金流偏度 + 集中度 — OLS成交额中性化

信号逻辑：
  1. Close>Open → 日内净买入 (正资金流)
  2. Close<Open → 日内净卖出 (负资金流)
  3. 连续净买入 + 资金量集中 → 自然人持续 + 法人持续买入 → 知情交易信号

与现有因子关系：
  - 本质上是对独立 in range_eff ≈ 日内价格效率 但用资金流代替价格
  - 与价格动量不对称性互补（一个看绝对价格幅度，一个看资金流方向）
  - 与成交量稀有度互补（一个看频率，一个看资金流幅度）

数据来源：csi1000_kline_raw.csv
输出：data/factor_ifi_v1.csv
"""

import pandas as pd
import numpy as np

# ----------------------------
# 0. 加载数据
# ----------------------------
print("加载 K 线数据 ...")
kline = pd.read_csv("data/csi1000_kline_raw.csv")
kline.columns = kline.columns.str.lower()
kline["date"] = pd.to_datetime(kline["date"])
kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)

print(f"K线数据: {len(kline)} rows, 股票: {kline.stock_code.nunique()}")

# ----------------------------
# 1. 基础字段校验
# ----------------------------
# 计算方向信号
kline["signed_flow"] = kline["pct_change"] / 100.0
kline["intraday_direction"] = np.sign(kline["signed_flow"])

# 资金流量（资金流加权方向）
kline["flow_amount"] = kline["intraday_direction"] * kline["amount"]

# ----------------------------
# 2. 构造因子
#    IFI = sign(close-open) × amount 的20日滚动均值的截面变换
# ----------------------------
WINDOW = 20

print(f"计算 {WINDOW}日滚动 IFI ...")
kline["ifi_raw"] = (
    kline.groupby("stock_code")["flow_amount"]
    .transform(lambda x: x.rolling(WINDOW, min_periods=10).mean())
)

# 筛掉 nan & 极端值
mask = kline["ifi_raw"].notna()
ifi_valid = kline[mask].copy()

# ----------------------------
# 3. 截面变换：OLS成交额中性化 + MAD缩尾 + z-score
# ----------------------------
from numpy.linalg import lstsq

def neutralize_ols(group):
    """OLS neutralization against log(amount_20d)"""
    y = group["ifi_raw"].values
    x = group["log_amount_20d"].values
    valid = np.isfinite(y) & np.isfinite(x)
    if valid.sum() < 30:
        group["ifi_neutral"] = np.nan
        return group
    X = np.column_stack([np.ones(valid.sum()), x[valid]])
    coef, _, _, _ = lstsq(X, y[valid], rcond=None)
    resid = np.full(len(y), np.nan)
    resid[valid] = y[valid] - X @ coef
    group["ifi_neutral"] = resid
    return group

print("计算 20日成交额均值 ...")
ifi_valid["log_amount_20d"] = (
    ifi_valid.groupby("stock_code")["amount"]
    .transform(lambda x: np.log1p(x.rolling(20, min_periods=10).mean()))
)

print("OLS中性化 ...")
ifi_valid = ifi_valid.groupby("date", group_keys=False).apply(neutralize_ols)

# MAD 缩尾 5.5σ
def mad_winsorize(series, n=5.5):
    med = series.median()
    mad = (series - med).abs().median() * 1.4826
    if mad == 0:
        return series.clip(series.quantile(0.01), series.quantile(0.99))
    return series.clip(med - n * mad, med + n * mad)

print("MAD缩尾 + z-score ...")
ifi_valid["ifi_factor"] = (
    ifi_valid.groupby("date")["ifi_neutral"]
    .transform(lambda x: mad_winsorize(x))
)
ifi_valid["ifi_factor"] = ifi_valid.groupby("date")["ifi_factor"].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x
)

# ----------------------------
# 4. 输出因子CSV
# ----------------------------
out = ifi_valid[["date", "stock_code", "ifi_factor"]].dropna(subset=["ifi_factor"])
out = out.rename(columns={"ifi_factor": "factor_value"})
out = out.sort_values(["date", "stock_code"]).reset_index(drop=True)

print(f"输出因子值: {len(out)} rows, 日期范围: {out.date.min()} ~ {out.date.max()}")
out.to_csv("data/factor_ifi_v1.csv", index=False)
print("文件已写入 data/factor_ifi_v1.csv")

# 保存因子统计用于迅速决策
print("\n因子分布:")
print(out["factor_value"].describe())
