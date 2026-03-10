#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: 放量下跌反转 (Volume-Confirmed Reversal v1)
====================================================
思路:
  经典短期反转的改进版。
  
  传统短期反转: 过去5-20日跌幅越大, 后续反弹越强
  问题: 有些下跌是信息驱动的(坏消息), 不会反弹
  
  改进: 加入成交量信号区分"恐慌性下跌"(会反弹) vs "信息性下跌"(不反弹)
  
  Avramov, Chordia & Goyal (2006): 
    短期反转效应在高换手率股票中更强(流动性驱动的价格偏离)
  
  构造:
    1. 5日收益率: ret5 = close(t)/close(t-5) - 1
    2. 5日平均换手率相对20日的z-score: to_z = (MA5_to - MA20_to) / std20_to
    3. 交互项: reversal_signal = -ret5 * max(to_z, 0)
       只在放量时(to_z>0)激活反转信号
       下跌+放量 → 正值(做多), 上涨+放量 → 负值(做空)
    4. 5% winsorize + 市值中性化

  也测试简化版: 纯短期反转 -ret5, 市值中性化
"""

import numpy as np
import pandas as pd
from pathlib import Path

def main():
    base = Path(__file__).resolve().parent.parent
    raw = pd.read_csv(base / "data" / "csi1000_kline_raw.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    print(f"[INFO] 数据: {raw['stock_code'].nunique()} 只, {raw['date'].nunique()} 天")
    
    def compute_factor(group):
        group = group.sort_values("date")
        close = group["close"].values
        to = group["turnover"].values
        
        n = len(close)
        factor = np.full(n, np.nan)
        
        for i in range(19, n):  # need 20 days history
            # 5-day return
            if close[i-5] > 0:
                ret5 = close[i] / close[i-5] - 1
            else:
                continue
            
            # turnover z-score (5d avg vs 20d)
            to_20 = to[i-19:i+1]
            valid_to = to_20[~np.isnan(to_20)]
            if len(valid_to) < 15:
                continue
            
            to_5 = to[i-4:i+1]
            valid_to5 = to_5[~np.isnan(to_5)]
            if len(valid_to5) < 3:
                continue
            
            ma20 = valid_to.mean()
            std20 = valid_to.std()
            ma5 = valid_to5.mean()
            
            if std20 > 1e-8:
                to_z = (ma5 - ma20) / std20
            else:
                to_z = 0
            
            # 放量下跌反转信号
            # 只在放量时激活: max(to_z, 0)
            # -ret5 * to_z_clipped: 下跌+放量 → 正值
            to_z_clipped = max(to_z, 0)
            factor[i] = -ret5 * (1 + to_z_clipped)  # 基础反转 + 放量加成
        
        group["factor_raw"] = factor
        group["log_amount_20d"] = np.log1p(group["amount"]).rolling(20, min_periods=15).mean()
        return group
    
    raw = raw.groupby("stock_code", group_keys=False).apply(compute_factor)
    
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
    
    out_path = base / "data" / "factor_vol_reversal_v1.csv"
    df_factor.to_csv(out_path, index=False)
    print(f"[INFO] 因子输出: {out_path}")
    print(f"[INFO] 因子样本: {len(df_factor)} 行, {df_factor['date'].nunique()} 天, {df_factor['stock_code'].nunique()} 股")
    print(f"[INFO] 因子均值: {df_factor['factor_value'].mean():.6f}, 标准差: {df_factor['factor_value'].std():.6f}")

if __name__ == "__main__":
    main()
