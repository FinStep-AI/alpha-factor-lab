#!/usr/bin/env python3
"""
成交量极值事件收益确认因子 (vol_ext_confirm_v1)
===============================================
思路：
  20日滚动窗口内，将每个截面日的成交量做截面排名（相对位置），
  标记出「放量日」（截面排名 ≥ 70% 分位）和「缩量日」（≤ 30% 分位）。

  对个股而言，看它在这些极值日之后、未来 5 日的平均累计收益：
    high_ret = mean( 放量日后 forward_5d_return )
    low_ret  = mean( 缩量日后 forward_5d_return )
    raw = high_ret - low_ret

  高 raw → 放量涨、缩量跌 → 放量是信息确认/买入信号
  低 raw → 放量跌、缩量涨 → 放量是恐慌/卖出信号

中性化：成交额20日均值 OLS 残差 + MAD 去极值 + z-score
输出：data/factor_vol_ext_confirm_v1.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ── 路径 ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
kline_path   = BASE / "data" / "csi1000_kline_raw.csv"
returns_path = BASE / "data" / "csi1000_returns.csv"
out_path     = BASE / "data" / "factor_vol_ext_confirm_v1.csv"

print("[1/5] 读取数据 …")
kline = pd.read_csv(kline_path, parse_dates=["date"])
returns = pd.read_csv(returns_path, parse_dates=["date"])

# ── 2. 截面成交量排名（逐日） ────────────────────────────────────────────
print("[2/5] 计算截面 volume 排名 …")

# pivot: (date, stock_code) → volume
vol_pv = kline.pivot_table(index="date", columns="stock_code", values="volume")
vol_pv = vol_pv.sort_index()

# 截面分位数 rank（axis=1 = 每行/每日期截面排名，method=average 处理并列）
# rank 范围 [1, n_stocks]，归一化到 [0,1]
vol_rank = vol_pv.rank(axis=1, pct=True, method="average")   # (date × stock)

print(f"  vol_rank shape: {vol_rank.shape}")

# ── 3. 前瞻收益矩阵 ──────────────────────────────────────────────────────
print("[3/5] 计算前瞻 5 日收益 …")
ret_pv = returns.pivot_table(index="date", columns="stock_code", values="return")
ret_pv = ret_pv.sort_index()

# forward_return(t) = prod(r_{t+1 … t+5}) − 1
log_ret = np.log1p(ret_pv.clip(lower=-0.999))
cum_log = log_ret.cumsum()
fwd5 = np.expm1(cum_log.shift(-5) - cum_log)   # (date × stock)

# ── 4. 逐日滚动：标记放量日 / 缩量日 → 平均后续收益差 ────────────────────
print("[4/5] 逐日滚动计算因子 …")

window = 20          # 回看窗口
hi_q, lo_q = 0.70, 0.30   # 分位阈值
fwd_collect = []     # 收集每日截面结果

dates = vol_rank.index
stocks = vol_rank.columns

# 整体对齐
common_dates = vol_rank.index.intersection(fwd5.index)
common_stocks = vol_rank.columns.intersection(fwd5.columns)
vol_r  = vol_rank.loc[common_dates, common_stocks]
fwd5_r = fwd5.loc[common_dates, common_stocks]

# 直接用 numpy 3D 构造滚动窗口（避免挤 DataFrame 循环）
vr_arr  = vol_r.values          # (T, N)  float64
fw_arr  = fwd5_r.values         # (T, N)

T, N = vr_arr.shape
print(f"  共 {T} 个截面日 × {N} 只股票")

# 用 pandas rolling on axis=0
vr_df  = vol_r.copy()
fw_df  = fwd5_r.copy()

roll_results = []
for i in range(window, T):
    win_vr = vr_df.iloc[i - window : i]          # 20 rows
    win_fw = fw_df.iloc[i - window : i]          # 20 rows
    today  = vr_df.index[i]

    # 截面放量/缩量掩码
    hi_mask = win_vr.ge(hi_q)     # True = 放量日
    lo_mask = win_vr.le(lo_q)     # True = 缩量日

    # 放量日的平均 forward_5d 收益（每只股票取该股票所有放量日的均值）
    hi_mean = win_fw.where(hi_mask).mean(axis=0)   # NaN if never high
    lo_mean = win_fw.where(lo_mask).mean(axis=0)   # NaN if never low

    # 内外均值差 → 逐股原始因子
    raw = hi_mean - lo_mean
    raw.name = today
    roll_results.append(raw)

factor_raw_df = pd.DataFrame(roll_results)
print(f"  factor_raw shape: {factor_raw_df.shape}")

# ── 5. 成交额中性化 ──────────────────────────────────────────────────────
print("[5/5] 成交额中性化 …")

amt_pv = kline.pivot_table(index="date", columns="stock_code", values="amount")
amt_pv = amt_pv.sort_index()
log_amt_20d = np.log(amt_pv.rolling(20).mean() + 1)

# 对齐
fac = factor_raw_df
common_d = fac.index.intersection(log_amt_20d.index)
common_s = fac.columns.intersection(log_amt_20d.columns)
fac  = fac.loc[common_d, common_s]
la   = log_amt_20d.loc[common_d, common_s]

# OLS 残差：factor_raw ~ log_amount_20d   (截面逐日回归)


def neutralize_cross_section(y_row, x_row):
    """单日截面 OLS 中性化，返回残差（ndarray or NaN）。"""
    mask = np.isfinite(y_row) & np.isfinite(x_row)
    if mask.sum() < 30:
        return np.full_like(y_row, np.nan)
    y, x = y_row[mask], x_row[mask]
    x_dm = x - x.mean()
    beta = (x_dm * y).sum() / ((x_dm ** 2).sum() + 1e-12)
    alpha = y.mean() - beta * x.mean()
    resid = np.full_like(y_row, np.nan)
    resid[mask] = y - (alpha + beta * x)
    return resid


resid_rows = []
for i in range(len(fac)):
    r = neutralize_cross_section(fac.values[i], la.values[i])
    resid_rows.append(r)

factor_resid = pd.DataFrame(resid_rows, index=fac.index, columns=fac.columns)

# MAD 去极值 + z-score
def mad_zscore(s: pd.Series, k: float = 3.5) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if mad < 1e-10:
        return pd.Series(0.0, index=s.index)
    clipped = s.clip(med - k * mad, med + k * mad)
    return (clipped - clipped.mean()) / (clipped.std() + 1e-10)


factor_final = factor_resid.apply(mad_zscore, axis=1)

# ── 写出 ─────────────────────────────────────────────────────────────────
print(f"\n写出因子 → {out_path}")

out = (
    factor_final
    .stack()                                   # (date, stock_code)  MultiIndex
    .reset_index()
    .rename(columns={0: "factor_value"})
    .dropna(subset=["factor_value"])
)

out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
out["date"] = out["date"].astype(str)
out.to_csv(out_path, index=False)

stats = out["factor_value"].describe()
print(f"\n---- 因子描述统计 ----")
print(stats.to_string())
print(f"\n写出 {len(out):,} 行 → factor_vol_ext_confirm_v1.csv ✓")
