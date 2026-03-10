#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: 异常换手收益交互 (Abnormal Turnover-Return Interaction v1)
================================================================
思路:
  衡量"放量下跌"或"缩量上涨"的程度。
  
  Gervais, Kaniel & Mingelgrin (2001): 异常高成交量后股票倾向反转
  Llorente, Michaely, Saar & Wang (2002): 
    成交量-收益率交互项 = corr(volume_change, ret * lagged_ret)
    如果交互为正 = 信息驱动(动量), 为负 = 噪音驱动(反转)

  简化版本 (适合日频K线):
  
  abnormal_turnover = (turnover - MA20_turnover) / std20_turnover  # z-score
  ret_signed = pct_change / 100
  
  interaction = abnormal_turnover * ret_signed
  factor = -mean(interaction, 20d)  # 取负: 放量下跌(负interaction)→做多

  高因子值 = 近20日频繁出现"放量下跌"或"缩量上涨"
  = 噪音交易主导 → 后续均值回复 → 超额收益

  市值中性化 (OLS on log_amount_20d)
"""

import numpy as np
import pandas as pd
from pathlib import Path

def main():
    base = Path(__file__).resolve().parent.parent
    raw = pd.read_csv(base / "data" / "csi1000_kline_raw.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    if "pct_change" not in raw.columns or raw["pct_change"].isna().mean() > 0.3:
        raw["pct_change"] = raw.groupby("stock_code")["close"].pct_change() * 100
    
    print(f"[INFO] 数据: {raw['stock_code'].nunique()} 只, {raw['date'].nunique()} 天")
    
    WINDOW = 20
    
    # 1. 异常换手率 z-score
    def compute_features(group):
        group = group.sort_values("date")
        to = group["turnover"]
        ma = to.rolling(WINDOW, min_periods=15).mean()
        sd = to.rolling(WINDOW, min_periods=15).std()
        group["abn_turnover"] = (to - ma) / sd.replace(0, np.nan)
        
        # 日收益率 (小数)
        group["ret"] = group["pct_change"] / 100.0
        
        # 交互项: abnormal_turnover * return
        group["interaction"] = group["abn_turnover"] * group["ret"]
        
        # 20日均值 (取负: 放量下跌→做多)
        group["factor_raw"] = -group["interaction"].rolling(WINDOW, min_periods=15).mean()
        
        # log_amount for neutralization
        group["log_amount_20d"] = np.log1p(group["amount"]).rolling(WINDOW, min_periods=15).mean()
        
        return group
    
    raw = raw.groupby("stock_code", group_keys=False).apply(compute_features)
    
    # 2. Winsorize
    def winsorize_group(series, pct=0.05):
        lower = series.quantile(pct)
        upper = series.quantile(1 - pct)
        return series.clip(lower, upper)
    
    raw["factor_raw"] = raw.groupby("date")["factor_raw"].transform(winsorize_group)
    
    # 3. OLS neutralization
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
    
    out_path = base / "data" / "factor_abn_to_ret_v1.csv"
    df_factor.to_csv(out_path, index=False)
    print(f"[INFO] 因子输出: {out_path}")
    print(f"[INFO] 因子样本: {len(df_factor)} 行, {df_factor['date'].nunique()} 天, {df_factor['stock_code'].nunique()} 股")
    print(f"[INFO] 因子均值: {df_factor['factor_value'].mean():.6f}, 标准差: {df_factor['factor_value'].std():.6f}")

if __name__ == "__main__":
    main()
