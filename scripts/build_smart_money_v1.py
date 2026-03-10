#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: 大单净买入代理 (Smart Money Proxy v1)
============================================
思路:
  在没有分笔数据的情况下，用日频数据构造大单净买入的代理变量。
  
  核心观察: 知情交易者(大单)倾向于在高成交量时段交易。
  Chordia & Subrahmanyam (2004) 表明 order imbalance 可预测收益。
  
  代理构造: 
    成交量加权收益 (VWAP收益) vs 等权收益的差异
    
    vwap_ret = sum(ret_i * volume_i) / sum(volume_i)  # 高量日权重大
    equal_ret = mean(ret_i)                             # 等权
    
    smart_money = vwap_ret - equal_ret
    
  如果高成交量日倾向上涨(大单买入), smart_money > 0
  如果高成交量日倾向下跌(大单卖出), smart_money < 0
  
  20日窗口 + 市值中性化

  方向假设: 正向 (smart_money > 0 → 大单买入 → 后续有动量)
  
  文献参考:
  - Chordia & Subrahmanyam (2004) "Order Imbalance and Individual Stock Returns"
  - Easley, Lopez de Prado & O'Hara (2012) "Flow Toxicity and Liquidity"
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
    
    def compute_smart_money(group):
        group = group.sort_values("date")
        ret = group["pct_change"].values / 100.0  # 小数
        vol = group["volume"].values.astype(float)
        
        n = len(ret)
        smart = np.full(n, np.nan)
        
        for i in range(WINDOW - 1, n):
            start = i - WINDOW + 1
            r = ret[start:i+1]
            v = vol[start:i+1]
            mask = (~np.isnan(r)) & (~np.isnan(v)) & (v > 0)
            if mask.sum() < 15:
                continue
            r_valid = r[mask]
            v_valid = v[mask]
            
            vwap_ret = np.sum(r_valid * v_valid) / np.sum(v_valid)
            equal_ret = np.mean(r_valid)
            smart[i] = vwap_ret - equal_ret
        
        group["smart_money"] = smart
        
        # log_amount for neutralization
        group["log_amount_20d"] = np.log1p(group["amount"]).rolling(WINDOW, min_periods=15).mean()
        
        return group
    
    raw = raw.groupby("stock_code", group_keys=False).apply(compute_smart_money)
    
    # Winsorize
    def winsorize_group(series, pct=0.05):
        lower = series.quantile(pct)
        upper = series.quantile(1 - pct)
        return series.clip(lower, upper)
    
    raw["smart_money"] = raw.groupby("date")["smart_money"].transform(winsorize_group)
    
    # OLS neutralization
    from numpy.linalg import lstsq
    
    results = []
    for dt, grp in raw.groupby("date"):
        mask = grp["smart_money"].notna() & grp["log_amount_20d"].notna()
        sub = grp[mask].copy()
        if len(sub) < 50:
            continue
        
        y = sub["smart_money"].values
        X = np.column_stack([np.ones(len(sub)), sub["log_amount_20d"].values])
        
        try:
            beta, _, _, _ = lstsq(X, y, rcond=None)
            resid = y - X @ beta
        except:
            resid = y - y.mean()
        
        sub["factor_value"] = resid
        results.append(sub[["date", "stock_code", "factor_value"]])
    
    df_factor = pd.concat(results, ignore_index=True)
    
    out_path = base / "data" / "factor_smart_money_v1.csv"
    df_factor.to_csv(out_path, index=False)
    print(f"[INFO] 因子输出: {out_path}")
    print(f"[INFO] 因子样本: {len(df_factor)} 行, {df_factor['date'].nunique()} 天, {df_factor['stock_code'].nunique()} 股")
    print(f"[INFO] 因子均值: {df_factor['factor_value'].mean():.6f}, 标准差: {df_factor['factor_value'].std():.6f}")

if __name__ == "__main__":
    main()
