#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: 振幅成交量背离 (Amplitude-Volume Divergence v1) - 快速版
使用Pearson correlation代替Spearman以加速
"""

import numpy as np
import pandas as pd
from pathlib import Path

def rolling_corr_manual(x, y, window=20, min_periods=15):
    """手动计算滚动Pearson相关系数，比scipy快得多"""
    n = len(x)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        start = i - window + 1
        xw = x[start:i+1]
        yw = y[start:i+1]
        mask = (~np.isnan(xw)) & (~np.isnan(yw))
        if mask.sum() < min_periods:
            continue
        xv = xw[mask]
        yv = yw[mask]
        
        xm = xv.mean()
        ym = yv.mean()
        xd = xv - xm
        yd = yv - ym
        
        denom = np.sqrt((xd**2).sum() * (yd**2).sum())
        if denom > 1e-12:
            result[i] = (xd * yd).sum() / denom
    
    return result

def main():
    base = Path(__file__).resolve().parent.parent
    raw = pd.read_csv(base / "data" / "csi1000_kline_raw.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    print(f"[INFO] 数据: {raw['stock_code'].nunique()} 只, {raw['date'].nunique()} 天")
    
    WINDOW = 20
    
    def compute_factor(group):
        group = group.sort_values("date")
        amp = group["amplitude"].values.astype(float)
        to = group["turnover"].values.astype(float)
        amount = group["amount"].values.astype(float)
        
        # Rolling Pearson corr(amplitude, turnover)
        corr_vals = rolling_corr_manual(amp, to, WINDOW, 15)
        
        # Factor = negative correlation (high divergence → positive)
        group["factor_raw"] = -corr_vals
        
        # log amount 20d
        group["log_amount_20d"] = np.log1p(pd.Series(amount)).rolling(WINDOW, min_periods=15).mean().values
        
        return group
    
    raw = raw.groupby("stock_code", group_keys=False).apply(compute_factor)
    
    print("[INFO] 因子计算完成, 开始中性化...")
    
    # Winsorize
    def winsorize_group(series, pct=0.05):
        lower = series.quantile(pct)
        upper = series.quantile(1 - pct)
        return series.clip(lower, upper)
    
    raw["factor_raw"] = raw.groupby("date")["factor_raw"].transform(winsorize_group)
    
    # OLS neutralization
    from numpy.linalg import lstsq
    
    results = []
    for dt, grp in raw.groupby("date"):
        mask = grp["factor_raw"].notna() & grp["log_amount_20d"].notna()
        sub = grp[mask].copy()
        if len(sub) < 50:
            continue
        
        y = sub["factor_raw"].values
        X = np.column_stack([np.ones(len(sub)), sub["log_amount_20d"].values])
        
        try:
            beta, _, _, _ = lstsq(X, y, rcond=None)
            resid = y - X @ beta
        except:
            resid = y - y.mean()
        
        sub["factor_value"] = resid
        results.append(sub[["date", "stock_code", "factor_value"]])
    
    df_factor = pd.concat(results, ignore_index=True)
    
    out_path = base / "data" / "factor_amp_vol_diverge_v1.csv"
    df_factor.to_csv(out_path, index=False)
    print(f"[INFO] 因子输出: {out_path}")
    print(f"[INFO] 因子样本: {len(df_factor)} 行, {df_factor['date'].nunique()} 天, {df_factor['stock_code'].nunique()} 股")
    print(f"[INFO] 因子均值: {df_factor['factor_value'].mean():.6f}, 标准差: {df_factor['factor_value'].std():.6f}")

if __name__ == "__main__":
    main()
