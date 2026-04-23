#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: 日内回复强度 v1 (Intraday Reversal Strength v1)
================================================================
思路:
  衡量日内价格从开盘到收盘的回复/反转程度。
  核心指标: (open - close) / amplitude
  其中 amplitude = (high - low) / prev_close
  
  正值: 日内从开盘下跌 (开盘价高于收盘价, 且下跌幅度占日内波幅的比例)
  负值: 日内从开盘上涨 (开盘价低于收盘价, 反弹占日内波幅的比例)
  
  取20日均值: 平滑单日噪音, 捕捉持续性的日内回复特征。
  
  假设:
    日内回复越强(即 open→close 方向与 open→midpoint 方向相反的程度越大)
    说明:
    - 高开深回落: 追涨资金被套, 弱势信号 → 可能需要反向逻辑
    - 低开大反弹: 卖压耗尽/抄底盘强劲, 强势信号
    
  本因子取原始值(不定方向), 让回测确认方向。预期:
    - 做多低开大反弹(负值) 的股票
    - 或 做空高开深回落(正值) 的股票
  
  市值中性化 (OLS on log_amount_20d) + MAD winsorize + z-score
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
    
    # amplitude 可能不存在, 计算
    if "amplitude" not in raw.columns or raw["amplitude"].isna().mean() > 0.3:
        raw["prev_close"] = raw.groupby("stock_code")["close"].shift(1)
        raw["amplitude"] = (raw["high"] - raw["low"]) / raw["prev_close"] * 100
    
    print(f"[INFO] 数据: {raw['stock_code'].nunique()} 只, {raw['date'].nunique()} 天")
    
    WINDOW = 20
    MIN_PERIODS = 15
    
    def compute_features(group):
        group = group.sort_values("date")
        
        # 日内回复强度: (open - close) / amplitude
        # amplitude 已经是百分比, 但如果为0或很小则跳过
        amp = group["amplitude"].replace(0, np.nan)
        group["intraday_rev"] = (group["open"] - group["close"]) / amp
        
        # 20日均值
        group["factor_raw"] = group["intraday_rev"].rolling(WINDOW, min_periods=MIN_PERIODS).mean()
        
        # 20日均成交额 (用作中性化变量)
        group["log_amount_20d"] = np.log1p(group["amount"].rolling(WINDOW, min_periods=MIN_PERIODS).mean())
        
        return group
    
    raw = raw.groupby("stock_code", group_keys=False).apply(compute_features)
    
    # MAD Winsorize
    def winsorize_mad(series, n_mad=5.0):
        median = series.median()
        mad = (series - median).abs().median() * 1.4826  # 等价于正态分布的1.4826倍
        if mad == 0:
            return series
        lower = median - n_mad * mad
        upper = median + n_mad * mad
        return series.clip(lower, upper)
    
    raw["factor_raw"] = raw.groupby("date")["factor_raw"].transform(winsorize_mad)
    
    # OLS neutralization (截面上对 log_amount_20d 做回归取残差)
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
    
    # 截面z-score
    df_factor["factor_value"] = df_factor.groupby("date")["factor_value"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0.0
    )
    
    out_path = base / "data" / "factor_intraday_rev_str_v1.csv"
    df_factor.to_csv(out_path, index=False)
    
    print(f"[INFO] 因子输出: {out_path}")
    print(f"[INFO] 样本: {len(df_factor)} 行, {df_factor['date'].nunique()} 天, {df_factor['stock_code'].nunique()} 股")
    print(f"[INFO] 因子统计: mean={df_factor['factor_value'].mean():.4f}, std={df_factor['factor_value'].std():.4f}")

if __name__ == "__main__":
    main()
