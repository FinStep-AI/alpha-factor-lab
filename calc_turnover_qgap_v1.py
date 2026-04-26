#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turnover Concentration Asymmetry (TCA) v1

构造：20日换手率分布70%分位 - 30%分位比较，换手率在高端或低端不对称分布

逻辑：成交量在某个方向过度集中的股票更有可能有特定风格的知情交易。

构造：
- 70%高位换手率均值 - 30%低位换手率均值，衡量高端比低端的集中程度。
- 求 Z-score 后做市场价中性化；
- 正值 可能对应：高活跃日集中的信息驱动势头；
- 负值 对应高活跃日但在清淡日段，可能表明方向已知但换手不均匀。

公式: f_raw = mean(turn[turn>q70]) - mean(turn[turn<q30]), 20 d window
中性: log_amount / 成交额 OLS → 残差 → MAD win → z-score
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 参数
# ============================================================
DATA_DIR = Path(__file__).parent.parent / "data"
KLINE_FILE = DATA_DIR / "csi1000_kline_raw.csv"
OUTPUT_FILE = DATA_DIR / "factor_turnover_qgap_v1.csv"
WINDOW = 20
Q_HIGH = 0.70   # 高位截点
Q_LOW  = 0.30   # 低位截点

# ============================================================
# 加载数据
# ============================================================
print("=== 换手率Q-分位差值因子(TA - TB) v1 ===")
print(f"加载数据: {KLINE_FILE}")

df = pd.read_csv(KLINE_FILE, parse_dates=["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
# 过滤极值
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
df = df.dropna(subset=["date", "stock_code", "turnover", "amount"])

# ============================================================
# 因子计算
# ============================================================
print(f"计算 20日-Q分位差值 ...")

results = []

for code, grp in df.groupby("stock_code"):
    grp = grp.sort_values("date").copy()

    # 滚动窗口内计算70%和30%分位
    def _qgap(x):
        qts = x.quantile([Q_LOW, Q_HIGH])
        ql = qts[Q_LOW]
        qh = qts[Q_HIGH]
        hi = x[x >= qh]
        lo = x[x <= ql]
        if len(hi) == 0 or len(lo) == 0:
            return np.nan
        return hi.mean() - lo.mean()

    grp["tca_raw"] = grp["turnover"].rolling(window=WINDOW, min_periods=15).apply(lambda x: _qgap(x), raw=False)

    # 成交额 对数
    grp["log_amount"] = np.log1p(grp["amount"].clip(lower=1))

    # 市值中性化: OLS 对 (tca_raw ~ log_amount)
    from numpy.polynomial import polynomial as P

    for i, row in grp.iterrows():
        if pd.isna(row["tca_raw"]) or pd.isna(row["log_amount"]):
            grp.at[i, "tca_neutral"] = np.nan
            continue
        # 截面回归: log_amount残差
        mask = grp["tca_raw"].notna() & grp["log_amount"].notna()
        if mask.sum() < 30:
            grp.at[i, "tca_neutral"] = np.nan
            continue
        xs = grp.loc[mask, "log_amount"].values
        ys = grp.loc[mask, "tca_raw"].values
        try:
            # OLS (带截距)
            X = np.column_stack([np.ones(len(xs)), xs])
            beta = np.linalg.lstsq(X, ys, rcond=None)[0]
            resid = ys - X @ beta
            # 对应当前行残差
            idx_in_mask = list(mask[mask].index).index(i)
            grp.at[i, "tca_neutral"] = resid[idx_in_mask]
        except Exception:
            grp.at[i, "tca_neutral"] = np.nan

    results.append(grp[["date", "stock_code", "tca_raw", "tca_neutral"]])

print("合并截面值...")
out = pd.concat(results, ignore_index=True).dropna(subset=["tca_neutral"])

print("截面 MAD winsorize + z-score...")
def mad_zscore(s):
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if mad < 1e-8:
        return s * 0.0
    return ((s - med) / mad).clip(-5, 5)

out["factor_value"] = out.groupby("date")["tca_neutral"].transform(mad_zscore)

print(f"写入: {OUTPUT_FILE}")
out[["date", "stock_code", "factor_value"]].rename(
    columns={"date": "date", "stock_code": "stock_code", "factor_value": "factor_value"}
).to_csv(OUTPUT_FILE, index=False)

# 督导
n_dates = out["date"].nunique()
n_stocks = out["stock_code"].nunique()
print(f"\n✅ 因子计算完成")
print(f"  截面数: {n_dates}")
print(f"  平均股票数: {int(out.groupby('date')['stock_code'].count().mean())}")
print(f"  输出: {OUTPUT_FILE}")
