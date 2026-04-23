#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: 波动率水平 v3 (Volatility Level v3)
================================================================
v2已入库: 60日均振幅, IC=0.022, t=1.98
v3升级: 20日raw波动率, 直接衡量近期实际波动

理论依据: Ang, Hodrick, Xing & Zhang (2006)
"The Cross-Section of Volatility and Expected Returns", Journal of Finance

原始发现在美股市: 高波动股票未来收益更低 (low vol anomaly)
但A股小盘股情况不同: 高波动=信息驱动=动量延续

用20日收益率标准差(非振幅), 对数变换, 成交额中性化
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
    
    WINDOW = 20
    MIN_PERIODS = 15
    
    def compute_features(group):
        group = group.sort_values("date")
        ret = group["pct_change"] / 100.0
        
        # 20日波动率
        vol_20d = ret.rolling(WINDOW, min_periods=MIN_PERIODS).std()
        
        # 对数变换 (压缩极端值)
        group["factor_raw"] = np.log1p(vol_20d)  # log(1+std)
        
        group["log_amount_20d"] = np.log1p(group["amount"].rolling(WINDOW, min_periods=MIN_PERIODS).mean())
        return group
    
    raw = raw.groupby("stock_code", group_keys=False).apply(compute_features)
    
    def winsorize_mad(series, n_mad=5.0):
        median = series.median()
        mad = (series - median).abs().median() * 1.4826
        if pd.isna(mad) or mad == 0:
            return series
        lower = median - n_mad * mad
        upper = median + n_mad * mad
        return series.clip(lower, upper)
    
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
    
    out_path = base / "data" / "factor_vol_log20d_v3.csv"
    df_factor.to_csv(out_path, index=False)
    
    print(f"[INFO] 因子输出: {out_path}")
    print(f"[INFO] 样本: {len(df_factor)} 行, {df_factor['date'].nunique()} 天, {df_factor['stock_code'].nunique()} 股")

if __name__ == "__main__":
    main()
