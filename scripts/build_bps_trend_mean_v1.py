#!/usr/bin/env python3
"""
因子：BPS Trend Mean Q v1（Barra Growth 方向）

逻辑：
- 取个股近 4 期可用季报 BPS
- 计算逐期 BPS 连续复利增长率: log(BPS_t / BPS_{t-1})
- 取最近 4 期（若不足 4 期则用全部可用期数）增长率的均值
- 取对数后年化: (mean_quarterly_growth) * 250 / 60
- 45 天延迟（报告期后45天数据才可用）
- 成交额中性化 + MAD + z-score
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

def build_bps_trend_mean_factor():
    # --- 1. 加载数据 ---
    here = Path(__file__).parent.parent
    kline = pd.read_csv(here / "data" / "csi1000_kline_raw.csv")
    fund  = pd.read_csv(here / "data" / "csi1000_fundamental_cache.csv")
    
    kline["date"]     = pd.to_datetime(kline["date"])
    fund["report_date"] = pd.to_datetime(fund["report_date"])
    
    # 只保留 bps 有效行
    fund = fund.dropna(subset=["bps"])
    fund = fund[fund["bps"] > 0].copy()      # BPS 必须 > 0 才能取对数
    
    # 计算 log(BPS) 差（逐季环比）
    fund = fund.sort_values(["stock_code", "report_date"])
    fund["bps_lag1"]  = fund.groupby("stock_code")["bps"].shift(1)
    fund["bps_lag2"]  = fund.groupby("stock_code")["bps"].shift(2)
    fund["bps_lag3"]  = fund.groupby("stock_code")["bps"].shift(3)
    fund["bps_lag4"]  = fund.groupby("stock_code")["bps"].shift(4)
    
    # 连续复利连续增长率: log(bps_t / bps_{t-1})
    fund["q_chg_1"] = np.log(fund["bps"] / fund["bps_lag1"])
    fund["q_chg_2"] = np.log(fund["bps"] / fund["bps_lag2"])
    fund["q_chg_3"] = np.log(fund["bps"] / fund["bps_lag3"])
    fund["q_chg_4"] = np.log(fund["bps"] / fund["bps_lag4"])
    
    # mean of up to 4 quarterly growth rates
    fund["bps_growth_4q"] = fund[["q_chg_1","q_chg_2","q_chg_3","q_chg_4"]].mean(axis=1)
    
    # --- 2. Point-in-time 映射 + 45 天延迟 ---
    # BPS 通常在报告期后 45 天才"已知"（经审计且在数据库可用）
    REPORT_LAG_DAYS = 45
    fund["available_date"] = fund["report_date"] + pd.Timedelta(days=REPORT_LAG_DAYS)
    fund["lag_repr_date"] = fund["report_date"]       # 用报告日期代表季度所属
    
    # --- 3. 展开到日频 ---
    # 每个交易日使用"该日之前最近可用 BPS (available_date ≤ date)"的 bps_growth_4q
    # 用循环 merge (left join with forward fill) 实现
    
    # 先对每只股票做前向填充
    fund_sorted = fund.sort_values(["stock_code", "available_date"])
    
    # 只取有效行
    fund_valid = fund_sorted[["stock_code","available_date","bps_growth_4q",
                               "report_date","bps","q_chg_1","q_chg_2","q_chg_3"]].dropna(subset=["bps_growth_4q"])
    
    print(f"[info] BPS quarters: {fund_valid.shape[0]}, stocks: {fund_valid['stock_code'].nunique()}")
    print(f"[info] BPS date range: {fund_valid['report_date'].min()} ~ {fund_valid['report_date'].max()}")
    
    # 自写 merge_asof (等价于 pandas merge_asof)
    kline = kline.sort_values(["stock_code","date"]).reset_index(drop=True)
    fund_valid = fund_valid.sort_values(["stock_code","available_date"]).reset_index(drop=True)
    
    result_rows = []
    for code, k_g in kline.groupby("stock_code"):
        f_g = fund_valid[fund_valid["stock_code"] == code].sort_values("available_date")
        if f_g.empty:
            continue
        # merge_asof: last available_date <= date
        merged = pd.merge_asof(
            k_g[["date"]].reset_index().rename(columns={"level_0":"_k_idx"}),
            f_g[["available_date","bps_growth_4q","report_date"]].reset_index().rename(columns={"index":"_f_idx"}),
            left_on="date", right_on="available_date",
            direction="backward"
        )
        merged["stock_code"] = code
        result_rows.append(merged[["_k_idx","stock_code","bps_growth_4q","report_date"]])
    
    merged_all = pd.concat(result_rows, ignore_index=True)
    
    # 相 merge 回 kline
    factor_intermediate = kline.reset_index().merge(
        merged_all[["_k_idx","bps_growth_4q","report_date"]],
        left_on="index", right_on="_k_idx", how="left"
    )
    
    # --- 4. 补充成交额中性化变量 ---
    factor_intermediate["log_amount_mean"] = (
        factor_intermediate.groupby("stock_code")["amount"]
        .transform(lambda x: np.log(x.rolling(20, min_periods=10).mean().replace(0, np.nan)))
    )
    
    # --- 5. 因子值 & 中性化 ---
    # 年化: mean_quarterly_growth * (250 trading days / 60 days per quarter)
    factor_intermediate["bps_trend_raw"] = factor_intermediate["bps_growth_4q"] * (250/60)
    
    print(f"\n[info] Raw factor stats before neutralization:")
    print(f"  valid count: {factor_intermediate['bps_trend_raw'].notna().sum()}")
    print(f"  non-NaN stock-dates: {factor_intermediate['bps_trend_raw'].notna().sum()}/{len(factor_intermediate)}")
    
    # 成交额OLS中性化
    factor_intermediate = _ols_neutralize(factor_intermediate, "bps_trend_raw", ["log_amount_mean"])
    
    # MAD winsor
    factor_intermediate["bps_trend"] = factor_intermediate.groupby("date")["bps_trend_neutral"].transform(
        lambda x: _mad_winsorize(x, 3.0)
    )
    
    # z-score
    gmean = factor_intermediate.groupby("date")["bps_trend"].transform("mean")
    gstd  = factor_intermediate.groupby("date")["bps_trend"].transform("std")
    factor_intermediate["bps_trend"] = (factor_intermediate["bps_trend"] - gmean) / (gstd + 1e-10)
    
    # --- 6. 输出 ---
    out = factor_intermediate[["date","stock_code","bps_trend"]].copy()
    out = out.dropna(subset=["bps_trend"])
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out_path = here / "data" / "factor_bps_trend_mean_v1.csv"
    out.to_csv(out_path, index=False)
    print(f"\n[OK] Factor saved to: {out_path}")
    print(f"  Rows: {len(out)}, Stocks: {out['stock_code'].nunique()}, Dates: {out['date'].nunique()}")
    print(f"  Date range: {out['date'].min()} ~ {out['date'].max()}")
    print(f"\nStats:")
    print(out["bps_trend"].describe())

def _ols_neutralize(df, value_col, neutralizer_cols):
    """截面OLS中性化：对每组截面向 neutralizer_cols 回归，取残差"""
    from numpy.linalg import lstsq
    
    residuals = np.full(len(df), np.nan)
    for date, g in df.groupby("date"):
        idx = g.index
        y = pd.to_numeric(g[value_col], errors="coerce").values
        mask = np.isfinite(y)
        if mask.sum() < 5:
            continue
        # build X: intercept + neutralizer columns
        X_cols = []
        for c in neutralizer_cols:
            if c in g.columns:
                X_cols.append(pd.to_numeric(g[c], errors="coerce").values.reshape(-1,1))
        
        X_full = np.column_stack([np.ones(len(g))] + X_cols)
        y_clean = y[mask]
        X_clean = X_full[mask]
        
        try:
            beta, _, _, _ = lstsq(X_clean, y_clean, rcond=None)
            res = y - X_full @ beta
            residuals[idx] = res
        except Exception:
            residuals[idx] = y
    
    df["_neutralized"] = residuals
    return df

def _mad_winsorize(x, n_mad=3.0):
    x = x.copy()
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return x
    sigma = 1.4826 * mad
    x = x.clip(med - n_mad*sigma, med + n_mad*sigma)
    return x

if __name__ == "__main__":
    build_bps_trend_mean_factor()
