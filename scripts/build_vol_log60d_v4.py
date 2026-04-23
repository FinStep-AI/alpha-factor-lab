#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: 波动率水平 v4 — 60日窗口 (Volatility Level v4, 60d)
================================================================
用60日波动率窗口替代20日, 信号更稳定, IC自相关更高, 标准误更低。

Ang, Hodrick, Xing & Zhang (2006) JFE
方向: 正向 (高波动 = 高收益 — 信息驱动/动量延续, A股小盘股特征)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import lstsq

def main():
    base = Path(__file__).resolve().parent.parent
    raw = pd.read_csv(base / "data" / "csi1000_kline_raw.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    if "pct_change" not in raw.columns or raw["pct_change"].isna().mean() > 0.3:
        raw["pct_change"] = raw.groupby("stock_code")["close"].pct_change() * 100
    
    print(f"[INFO] 数据: {raw['stock_code'].nunique()} 只, {raw['date'].nunique()} 天")
    
    WINDOW = 60
    MIN_PERIODS = 45
    
    def compute_features(group):
        group = group.sort_values("date")
        ret = group["pct_change"] / 100.0
        group["factor_raw"] = np.log1p(ret.rolling(WINDOW, min_periods=MIN_PERIODS).std())
        group["log_amount_20d"] = np.log1p(group["amount"].rolling(20, min_periods=15).mean())
        return group
    
    raw = raw.groupby("stock_code", group_keys=False).apply(compute_features)
    
    def winsorize_mad(series, n_mad=5.0):
        median = series.median()
        mad = (series - median).abs().median() * 1.4826
        if pd.isna(mad) or mad == 0:
            return series
        return series.clip(median - n_mad * mad, median + n_mad * mad)
    
    raw["factor_raw"] = raw.groupby("date")["factor_raw"].transform(winsorize_mad)
    
    results = []
    for dt, grp in raw.groupby("date"):
        mask = grp["factor_raw"].notna() & grp["log_amount_20d"].notna()
        sub = grp[mask].copy()
        if len(sub) < 50:
            continue
        y = sub["factor_raw"].values.astype(float)
        X = np.column_stack([np.ones(len(sub)), sub["log_amount_20d"].values.astype(float)])
        try:
            beta, _, _, _ = lstsq(X, y, rcond=None)
            resid = y - X @ beta
        except:
            resid = y - y.mean()
        sub = sub.copy()
        sub["factor_value"] = resid
        results.append(sub[["date", "stock_code", "factor_value"]])
    
    if not results:
        print("[ERROR] 无有效结果!")
        return
    
    df_factor = pd.concat(results, ignore_index=True)
    df_factor["factor_value"] = df_factor.groupby("date")["factor_value"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0.0
    )
    
    out_path = base / "data" / "factor_vol_log60d_v4.csv"
    df_factor.to_csv(out_path, index=False)
    print(f"[INFO] 输出: {out_path} | {len(df_factor)} rows, {df_factor['date'].nunique()} days")

if __name__ == "__main__":
    main()
