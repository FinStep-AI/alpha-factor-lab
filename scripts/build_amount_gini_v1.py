#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: 日内反转强度 (Intraday Reversal Intensity v1)
====================================================
思路:
  衡量日内"开盘跳空"与"盘中回落"的反转程度。
  
  具体公式:
    gap_ret = (open_t - close_{t-1}) / close_{t-1}  # 隔夜跳空
    intra_ret = (close_t - open_t) / open_t           # 日内收益
    reversal_signal = -gap_ret * intra_ret             # 乘积为正=反转
    
  当隔夜跳空为正但日内回落(或反之), reversal_signal > 0
  取20日均值, 高值=日内反转频繁且强烈=市场分歧大
  
  假设:
    日内反转强烈的股票, 短期交易噪音大, 价格偏离基本面,
    反而在20日视角均值回复, 后续有超额收益。
    (类似shadow_pressure的逻辑, 但从不同角度)

  另一种构造 (simpler, 更直接):
    abs_reversal = |gap_ret - intra_ret| 的20日均值
    这衡量隔夜方向与日内方向的绝对分歧程度
    
  最终选择更简洁的: signed reversal score
    daily_score = sign(gap_ret) != sign(intra_ret) → 1, else 0
    weighted by |gap_ret - intra_ret|
    20日均值后市值中性化

实际上, 让我换一个更有经济学基础的因子:

因子: 成交额集中度 (Turnover Concentration / Gini)
====================================================
衡量过去20日的日成交额分布是否集中在少数几天(Gini系数高)。

高Gini = 成交额集中在少数天 = 可能有大事件/主力进出
低Gini = 成交额均匀分布 = 常规交易

假设:
  在中证1000小盘股中, 成交额集中(Gini高)通常意味着
  有知情交易者集中建仓/出货。
  方向需要回测确认。

公式:
  1. 取过去20日每日成交额序列 [a1, a2, ..., a20]
  2. 计算Gini系数 = (2 * sum(i*a_sorted[i]) / (n*sum(a))) - (n+1)/n
  3. 5% winsorize + 市值中性化(OLS on log_amount_20d)
"""

import numpy as np
import pandas as pd
from pathlib import Path

def gini_coefficient(x):
    """计算一个序列的Gini系数 (0=完全均匀, 1=完全集中)"""
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or x.sum() == 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) / (n * np.sum(x))) - (n + 1) / n

def main():
    base = Path(__file__).resolve().parent.parent
    raw = pd.read_csv(base / "data" / "csi1000_kline_raw.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    print(f"[INFO] 数据: {raw['stock_code'].nunique()} 只, {raw['date'].nunique()} 天")
    
    WINDOW = 20
    
    # 计算滚动Gini系数
    def compute_gini_rolling(group):
        group = group.sort_values("date")
        amounts = group["amount"].values
        ginis = []
        for i in range(len(amounts)):
            if i < WINDOW - 1:
                ginis.append(np.nan)
            else:
                window_data = amounts[max(0, i - WINDOW + 1):i + 1]
                # 至少15个有效值
                valid = window_data[~np.isnan(window_data)]
                if len(valid) >= 15:
                    ginis.append(gini_coefficient(valid))
                else:
                    ginis.append(np.nan)
        group["amount_gini"] = ginis
        return group
    
    raw = raw.groupby("stock_code", group_keys=False).apply(compute_gini_rolling)
    
    # 计算log_amount_20d用于市值中性化
    raw["log_amount"] = np.log1p(raw["amount"])
    
    def rolling_log_amount(group):
        group = group.sort_values("date")
        group["log_amount_20d"] = group["log_amount"].rolling(WINDOW, min_periods=15).mean()
        return group
    
    raw = raw.groupby("stock_code", group_keys=False).apply(rolling_log_amount)
    
    # 5% winsorize
    def winsorize_group(series, pct=0.05):
        lower = series.quantile(pct)
        upper = series.quantile(1 - pct)
        return series.clip(lower, upper)
    
    raw["amount_gini"] = raw.groupby("date")["amount_gini"].transform(winsorize_group)
    
    # OLS市值中性化
    from numpy.linalg import lstsq
    
    results = []
    for dt, grp in raw.groupby("date"):
        mask = grp["amount_gini"].notna() & grp["log_amount_20d"].notna()
        sub = grp[mask].copy()
        if len(sub) < 50:
            continue
        
        y = sub["amount_gini"].values
        X = np.column_stack([np.ones(len(sub)), sub["log_amount_20d"].values])
        
        try:
            beta, _, _, _ = lstsq(X, y, rcond=None)
            resid = y - X @ beta
        except:
            resid = y - y.mean()
        
        sub["factor_value"] = resid
        results.append(sub[["date", "stock_code", "factor_value"]])
    
    df_factor = pd.concat(results, ignore_index=True)
    
    # 输出
    out_path = base / "data" / "factor_amount_gini_v1.csv"
    df_factor.to_csv(out_path, index=False)
    print(f"[INFO] 因子输出: {out_path}")
    print(f"[INFO] 因子样本: {len(df_factor)} 行, {df_factor['date'].nunique()} 天, {df_factor['stock_code'].nunique()} 股")
    print(f"[INFO] 因子均值: {df_factor['factor_value'].mean():.6f}, 标准差: {df_factor['factor_value'].std():.6f}")

if __name__ == "__main__":
    main()
