#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: 收益稳定性 v1 (Return Stability v1)
================================================================
思路来源: Bali, Brown, Murray & Tang (2017) "A Lottery-Demand-Based Explanation of the Beta Anomaly"
变体: 用收益率的"稳定性"作为Quality代理。

核心逻辑:
  稳定的收益模式 = 较少极端波动 = 更可预测的未来现金流 = 更高的Quality评分
  
改进变体 (与ret_skewness区分):
  不是看偏度(高阶矩), 而是看收益率的相对离散程度:
    Ret_Stability = 1 / CV(returns)
    CV = std(ret) / mean(|ret|)  [用绝对均值避免正负抵消]
  
  高值 = 相对离散度低 = 收益稳定
  低值 = 波动大 = 信息混乱/风险高
  
  注意: 这里mean(|ret|)衡量的是平均绝对日收益, std衡量离散。
        CV高 = 波动相对平均幅度大 = 不稳定。

市值中性化 + z-score
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
        ret = group["pct_change"] / 100.0  # 小数收益率
        
        # 滚动 std 和 mean(|ret|)
        roll_std = ret.rolling(WINDOW, min_periods=MIN_PERIODS).std()
        roll_mean_abs = ret.abs().rolling(WINDOW, min_periods=MIN_PERIODS).mean()
        
        # CV = std / mean(|ret|), 防止除零
        cv = roll_std / roll_mean_abs.replace(0, np.nan)
        
        # 稳定性 = 1 / CV (越高 = 越稳定)
        # 直接取负号: 低稳定性(高CV)→正高分 → 后续回测判断方向
        group["factor_raw"] = -cv  # 负CV: 波动大→负值, 波动小→正值
        
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
    
    out_path = base / "data" / "factor_ret_stability_v1.csv"
    df_factor.to_csv(out_path, index=False)
    
    print(f"[INFO] 因子输出: {out_path}")
    print(f"[INFO] 样本: {len(df_factor)} 行, {df_factor['date'].nunique()} 天, {df_factor['stock_code'].nunique()} 股")
    print(f"[INFO] 因子统计: mean={df_factor['factor_value'].mean():.4f}, std={df_factor['factor_value'].std():.4f}")

if __name__ == "__main__":
    main()
