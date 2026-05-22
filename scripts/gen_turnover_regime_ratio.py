#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子：波动状态换手率比值 v1 (Turnover Regime Ratio)
ID: turnover_regime_ratio_v1

逻辑
----
20日内，每日 Rank(amplitude)，取振幅 Top-10 日的均值换手率 / Bottom-10 日的均值换手率。
log 变换后成交额OLS中性化 + MAD缩尾 + z-score。

高比值 = 换手增量集中在高信息密度日（知情交易偏多）→ 正 alpha。
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

KLINE = "data/csi1000_kline_raw.csv"
OUT   = "data/factor_turnover_regime_ratio_v1.csv"
WIN   = 20
HALF  = 10          # 取10高10低

# ── 1. 读取并预处理 ───────────────────────────────────────────────
print("[1] 读取 K-line …")
df = pd.read_csv(KLINE, usecols=["date","stock_code","open","close","high","low",
                                   "volume","amount","amplitude","turnover"])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code","date"]).reset_index(drop=True)
df["prev_close"] = df.groupby("stock_code")["close"].shift(1)

# 振幅：(high-low)/prev_close（amplitude 列若有缺失用此补）
miss = df["amplitude"].isna() | (df["amplitude"] == 0)
df.loc[miss, "amplitude"] = (
    (df.loc[miss,"high"] - df.loc[miss,"low"]) /
    df.loc[miss,"prev_close"]
)

# 20日滚动成交额均值（用于中性化）
df["log_amount_20d"] = (
    df.groupby("stock_code")["amount"]
      .transform(lambda x: np.log(x.rolling(WIN, min_periods=WIN).mean() + 1))
)

# ── 2. 向量化：对每个 stock×date 取过去20天的振幅 rank ────────────
print("[2] 向量化滚动振幅分组 …")

def rank_rolling(series: pd.Series, window: int) -> pd.Series:
    """滚动排名 1..window（升序）"""
    return series.rolling(window, min_periods=window).rank(method="first")

df = df.sort_values(["stock_code","date"])

# 振幅升序 rank（1=最低振幅，WIN=最高振幅）
df["ampl_rank"] = (
    df.groupby("stock_code")["amplitude"]
      .transform(lambda x: x.rolling(WIN, min_periods=WIN).rank())
)

# 高波日 / 低波日 mask（rank > HALF → 高，rank <= HALF → 低）
df["hi_flag"] = (df["ampl_rank"] > HALF).astype(int)
df["lo_flag"] = (df["ampl_rank"] <= HALF).astype(int)

# 加权换手率（仅标记日的换手进入均值计算）
df["turn_hi"] = df["turnover"] * df["hi_flag"]
df["turn_lo"] = df["turnover"] * df["lo_flag"]

hi_sum = df.groupby(["stock_code"])["turn_hi"].transform(
    lambda x: x.rolling(WIN, min_periods=WIN).sum())
lo_sum = df.groupby(["stock_code"])["turn_lo"].transform(
    lambda x: x.rolling(WIN, min_periods=WIN).sum())

# 因为 rank <=10 的那个窗口恰好10个1，和直接 / HALF 等价
hi_mean = hi_sum / HALF
lo_mean = lo_sum / HALF

df["factor_raw"] = np.log(
    np.where(lo_mean > 1e-8, hi_mean / lo_mean, np.nan)
)

# 截断日期范围（需同时有足够历史）
mask_valid = df["factor_raw"].notna() & df["log_amount_20d"].notna()
df = df.loc[mask_valid].copy()
print(f"    有效截面: {df['date'].nunique()} 天，{df['stock_code'].nunique()} 只")

# ── 3. 截面中性化（OLS + MAD + z-score）─────────────────────────
print("[3] 截面中性化 …")

def neutralize_group(g: pd.DataFrame) -> pd.Series:
    y = g["factor_raw"].values.astype(float)
    x = g["log_amount_20d"].values.astype(float)
    # OLS
    X = np.column_stack([np.ones(len(x)), x])
    try:
        b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ b
    except Exception:
        resid = y
    # MAD winsorize 5.2σ（MAD→σ 系数 1.4826）
    med  = np.nanmedian(resid)
    mad  = np.nanmedian(np.abs(resid - med)) + 1e-8
    k    = 5.2 * 1.4826
    resid = np.clip(resid, med - k * mad, med + k * mad)
    # z-score
    m = np.nanmean(resid)
    s = np.nanstd(resid) + 1e-8
    return pd.Series((resid - m) / s, index=g.index)

df["factor_value"] = df.groupby("date", group_keys=False).apply(neutralize_group)

out = df[["date","stock_code","factor_value"]].dropna(subset=["factor_value"])
out["date"] = out["date"].dt.strftime("%Y-%m-%d")
out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
out.to_csv(OUT, index=False)

print(f"[完成] 写入 {OUT}")
print(f"      行数 {len(out)}，日期 {out['date'].min()} ~ {out['date'].max()}")
print(f"      factor_raw 均值={df['factor_raw'].mean():.4f} std={df['factor_raw'].std():.4f}")
print(f"      factor_value 均值={df['factor_value'].mean():.4f} std={df['factor_value'].std():.4f}")
