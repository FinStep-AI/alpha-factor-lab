#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: 日内动量一致性 (Intraday Momentum Consistency v1)
========================================================
思路:
  衡量个股近20日的日内走势一致性。具体:
  
  signed_body = (close - open) / open  # 日内方向
  body_consistency = mean(signed_body, 20d)  
  
  如果近20日持续"收阳"(close > open), body_consistency > 0
  如果近20日持续"收阴", body_consistency < 0
  
  动量假设: 日内持续收阳=多方持续占优, 后续动量延续
  
  改进: 加入振幅加权
  weighted_body = signed_body / amplitude  (规范化by当日振幅)
  这样大振幅日权重降低, 避免单日异常值主导

  实际构造:
  1. body_ratio = (close - open) / (high - low + 1e-8)
     范围 [-1, 1], 衡量收盘在日内范围中的方向和强度
  2. 20日均值
  3. 5% winsorize + 市值中性化

Wait — 这其实和 close_location_v1 很接近 ((close-low)/(high-low))。
让我换个角度:

因子: 跳空回补率 (Gap Fill Rate v1)
========================================
衡量隔夜跳空后日内回补的比率。

gap = open_t / close_{t-1} - 1
fill_rate = 1 - (close_t - close_{t-1}) / (open_t - close_{t-1})  when gap != 0

如果跳空完全回补(close=prev_close), fill_rate = 1
如果跳空不回补(close=open), fill_rate = 0
如果反向突破, fill_rate > 1

20日均值后市值中性化。

高回补率 = 每次跳空都被日内交易回补 = 反转力量强
低回补率 = 跳空后继续向跳空方向走 = 动量力量强

先跑两个方向看哪个有效。

实际上让我换一个更有扎实理论基础的:

因子: 已实现偏度符号一致性 (Signed Realized Skewness Persistence)
================================================================
不对,已经试过realized_skew了。

OK 让我直接试一个纯新的: 
收益率离散度因子 (Return Dispersion / Cross-sectional Momentum)
— 不,这是截面层面的,需要不同数据。

最终选择: 成交量趋势 (Volume Trend v1)
========================================
思路:
  用成交量的线性趋势斜率(20日OLS)衡量量能变化趋势。
  正斜率=放量趋势, 负斜率=缩量趋势。
  
  与turnover_decay不同:
  - turnover_decay是5日/20日比值(离散点)
  - volume_trend是连续20日线性拟合斜率(更平滑)
  
  假设:
  - 在中证1000, 缩量趋势(负斜率)=卖压逐渐耗尽→反弹
  - 这与shadow_pressure/close_vwap_dev(反转系因子)一脉相承
  
  或者反过来: 放量趋势=资金持续流入→动量延续
  需要回测确认方向。

公式:
  1. 每日标准化成交量: vol_norm = volume / mean(volume, 60d)
  2. 20日滚动OLS: vol_norm ~ t, 取斜率
  3. 5% winsorize + 市值中性化
"""

import numpy as np
import pandas as pd
from pathlib import Path

def rolling_slope(values, window=20, min_periods=15):
    """计算滚动线性回归斜率"""
    n = len(values)
    slopes = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        start = i - window + 1
        y = values[start:i+1]
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < min_periods:
            continue
        y_valid = y[valid_mask]
        x = np.arange(len(y_valid), dtype=float)
        # OLS slope = cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = y_valid.mean()
        cov_xy = ((x - x_mean) * (y_valid - y_mean)).mean()
        var_x = ((x - x_mean) ** 2).mean()
        if var_x > 1e-12:
            slopes[i] = cov_xy / var_x
    
    return slopes

def main():
    base = Path(__file__).resolve().parent.parent
    raw = pd.read_csv(base / "data" / "csi1000_kline_raw.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    print(f"[INFO] 数据: {raw['stock_code'].nunique()} 只, {raw['date'].nunique()} 天")
    
    # 1. 标准化成交量 (除以60日均值)
    def compute_norm_vol(group):
        group = group.sort_values("date")
        ma60 = group["volume"].rolling(60, min_periods=30).mean()
        group["vol_norm"] = group["volume"] / ma60.replace(0, np.nan)
        return group
    
    raw = raw.groupby("stock_code", group_keys=False).apply(compute_norm_vol)
    
    # 2. 20日滚动OLS斜率
    def compute_slope(group):
        group = group.sort_values("date")
        group["vol_slope"] = rolling_slope(group["vol_norm"].values, window=20, min_periods=15)
        return group
    
    raw = raw.groupby("stock_code", group_keys=False).apply(compute_slope)
    
    # 3. log_amount_20d for neutralization
    raw["log_amount"] = np.log1p(raw["amount"])
    def rolling_log_amount(group):
        group = group.sort_values("date")
        group["log_amount_20d"] = group["log_amount"].rolling(20, min_periods=15).mean()
        return group
    
    raw = raw.groupby("stock_code", group_keys=False).apply(rolling_log_amount)
    
    # 4. Winsorize
    def winsorize_group(series, pct=0.05):
        lower = series.quantile(pct)
        upper = series.quantile(1 - pct)
        return series.clip(lower, upper)
    
    raw["vol_slope"] = raw.groupby("date")["vol_slope"].transform(winsorize_group)
    
    # 5. OLS neutralization
    from numpy.linalg import lstsq
    
    results = []
    for dt, grp in raw.groupby("date"):
        mask = grp["vol_slope"].notna() & grp["log_amount_20d"].notna()
        sub = grp[mask].copy()
        if len(sub) < 50:
            continue
        
        y = sub["vol_slope"].values
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
    out_path = base / "data" / "factor_vol_trend_v1.csv"
    df_factor.to_csv(out_path, index=False)
    print(f"[INFO] 因子输出: {out_path}")
    print(f"[INFO] 因子样本: {len(df_factor)} 行, {df_factor['date'].nunique()} 天, {df_factor['stock_code'].nunique()} 股")
    print(f"[INFO] 因子均值: {df_factor['factor_value'].mean():.6f}, 标准差: {df_factor['factor_value'].std():.6f}")

if __name__ == "__main__":
    main()
