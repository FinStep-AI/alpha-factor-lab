#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子：ROE 变化率 (ROE Acceleration v1)
Barra: Growth (首次入库填补)
参考: Hou、Xue & Zhang (2015) "Digesting Anomalies"
      因子构造逻辑: ROE 同比增长率作为 Quality/Growth 代理信号
*****************************************************
因子定义：
  1. 取每期 ROE（季度，前瞻-forward to daily）
  2. 计算 ROE 季环比变化 ΔROE_Q = ROE_t - ROE_{t-1}
  3. 计算 ROE 同比变化 ΔROE_Y = ROE_t - ROE_{t-4}
  4. 综合变化率 = 0.6 * YoY + 0.4 * QoQ (同比权重更高，剔除行业季节性)
  5. 20日滚动均值（平滑单期波动）
  6. 成交额对数中性化+ MAD 缩尾+z-score

逻辑：
  - ROE 加速增长 = 盈利质量改善
  - 做多 ROE 持续改善的股票（正向因子）
  - 中性化后排除单纯大盘溢价
"""

import pandas as pd
import numpy as np

print("[ROE Acceleration v1] 开始计算因子...")

# ── 1. 加载数据 ──────────────────────────────────────────
kline = pd.read_csv("data/csi1000_kline_raw.csv")
fund  = pd.read_csv("data/csi1000_fundamental_cache.csv")
ret   = pd.read_csv("data/csi1000_returns.csv")
codes = pd.read_csv("data/a_share_codes.json")

kline["date"] = pd.to_datetime(kline["date"])
ret["date"]   = pd.to_datetime(ret["date"])
fund["report_date"] = pd.to_datetime(fund["report_date"])

# ── 2. 预处理：只保留 CSI1000 股票 + 有效 ROE ──────────
fund = fund[fund["stock_code"].isin(kline["stock_code"].unique())].copy()
fund = fund.dropna(subset=["roe"]).copy()
fund = fund.sort_values(["stock_code", "report_date"]).reset_index(drop=True)

print(f"[ROE Acceleration] 基本面数据: {fund.stock_code.nunique()} 只, {fund.report_date.nunique()} 期")

# ── 3. 计算同环比 ROE 变化率 ─────────────────────────────
def compute_roe_changes(g):
    g = g.sort_values("report_date").copy()
    g["roe_prev_q"] = g["roe"].shift(1)   # 上季度
    g["roe_prev_y"] = g["roe"].shift(4)   # 上年同期

    g["dq_roe"] = g["roe"] - g["roe_prev_q"]   # QoQ (pp)
    g["dy_roe"] = g["roe"] - g["roe_prev_y"]   # YoY (pp)
    return g

fund = fund.groupby("stock_code", group_keys=False).apply(compute_roe_changes)
fund = fund.dropna(subset=["dy_roe"]).reset_index(drop=True)  # 至少需要2期以上

# 综合变化率: 0.6*Yoy + 0.4*Qoq (当 QoQ 可用时填充 yr)
has_qoq = fund["dq_roe"].notna()
fund.loc[has_qoq, "roe_accel_raw"] = 0.6 * fund.loc[has_qoq, "dy_roe"] + 0.4 * fund.loc[has_qoq, "dq_roe"]
fund.loc[~has_qoq, "roe_accel_raw"] = fund.loc[~has_qoq, "dy_roe"]

print(f"[ROE Acceleration] ROE 变化率(有效): {fund.roe_accel_raw.notna().sum():,} 条")

# ── 4. 将季度因子值 Forward-fill 到每日截面 ─────────────
fund_merged = fund[["stock_code", "report_date", "roe_accel_raw"]].copy()
fund_merged = fund_merged.rename(columns={"report_date": "date_ff"})

# 每日截面
daily_dates = kline[["date", "stock_code"]].drop_duplicates().copy()
daily_dates = daily_dates.merge(
    fund_merged.rename(columns={"stock_code": "stock_code_ff"}),
    left_on="stock_code", right_on="stock_code_ff", how="left"
)
daily_dates["match"] = (daily_dates["date"] >= daily_dates["date_ff"])
daily_dates = daily_dates[daily_dates["match"]].copy()

# 每只每天取最新报告期之前的因子值
daily_dates = daily_dates.sort_values(["stock_code", "date", "date_ff"])
daily_dates["latest_ff"] = daily_dates.groupby(["stock_code", "date"])["date_ff"].transform("max")
daily_factor = (
    daily_dates[daily_dates["date_ff"] == daily_dates["latest_ff"]]
    .groupby(["date", "stock_code"])["roe_accel_raw"]
    .last()
    .reset_index()
)

print(f"[ROE Acceleration] 每日截面因子有效值: {daily_factor.roe_accel_raw.notna().sum():,} / {len(daily_factor):,} ({daily_factor.roe_accel_raw.notna().mean()*100:.1f}%)")

# ── 5. 成交额中性化+截面上 winsorize+z-score ───────────
# 合并成交额
daily_factor = daily_factor.merge(
    kline[["date", "stock_code", "amount"]].drop_duplicates(),
    on=["date", "stock_code"], how="left"
)
daily_factor["log_amount_20d"] = daily_factor.groupby("stock_code")["amount"].transform(
    lambda x: x.rolling(20, min_periods=10).mean().apply(lambda v: np.log(v+1) if pd.notna(v) else v)
)
daily_factor["log_amount_20d"] = daily_factor.groupby("stock_code")["log_amount_20d"].ffill()

def cross_section_winsorize(series, n_mad=3.0):
    med = series.median()
    mad = (series - med).abs().median()
    scaled = 1.4826 * mad if mad > 0 else series.std()
    if scaled == 0 or np.isnan(scaled):
        return series
    return series.clip(med - n_mad * scaled, med + n_mad * scaled)

results = []
for dt, grp in daily_factor.groupby("date"):
    g = grp.copy()
    valid = g["roe_accel_raw"].notna() & g["log_amount_20d"].notna()
    if valid.sum() < 20:
        continue
    g = g[valid].copy()
    # Winsorize
    g["factor_w"] = cross_section_winsorize(g["roe_accel_raw"], n_mad=3.0)
    # log-transform for right-skew
    g["factor_wl"] = np.sign(g["factor_w"]) * np.log1p(g["factor_w"].abs())
    # Neutralize log_amount
    from numpy.linalg import lstsq
    X = np.column_stack([np.ones(len(g)), g["log_amount_20d"].values])
    y = g["factor_wl"].values
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        g["factor_n"] = y - X @ beta
    except np.linalg.LinAlgError:
        g["factor_n"] = g["factor_wl"].values
    # z-score
    mu, sigma = g["factor_n"].mean(), g["factor_n"].std()
    g["factor_z"] = np.where(sigma > 0, (g["factor_n"] - mu) / sigma, 0.0)
    g["date"] = dt
    results.append(g[["date", "stock_code", "factor_z"]])

result_df = pd.concat(results, ignore_index=True)
result_df = result_df.sort_values(["date", "stock_code"]).reset_index(drop=True)
result_df = result_df.rename(columns={"factor_z": "factor_value"})

print(f"[ROE Acceleration] 最终因子: {len(result_df):,} 行, {result_df.stock_code.nunique()} 只")
print(f"[ROE Acceleration] 因子分布: mean={result_df.factor_value.mean():.4f}, std={result_df.factor_value.std():.4f}")
print(f"[ROE Acceleration] 有效覆盖: {result_df.factor_value.notna().sum():,} / {len(result_df):,}")

result_df.to_csv("data/factor_roe_accel_v1.csv", index=False)
print("[ROE Acceleration] 保存 → data/factor_roe_accel_v1.csv")
