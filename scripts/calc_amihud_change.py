#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amihud Illiquidity Change (流动性改善因子)

思路: 
  Amihud非流动性指标的短期/长期比率 → 流动性是否在改善
  ratio = Amihud_5d / Amihud_20d
  低ratio = 近期流动性比长期好 = 流动性正在改善 → 做多(反向因子)
  
学术根据:
  - Amihud (2002) 非流动性因子的有效性已验证(我们的amihud_illiq_v2)
  - Brennan, Huh & Subrahmanyam (2013) "An Analysis of the Amihud Illiquidity 
    Premium", Review of Asset Pricing Studies
  - 流动性改善意味着更多关注/资金流入，价格趋势延续

计算:
  1. 每日Amihud = |ret| / amount
  2. Amihud_5d = MA5(Amihud), Amihud_20d = MA20(Amihud)
  3. ratio = -log(Amihud_5d / Amihud_20d)  (取反: 高=流动性改善)
  4. 成交额OLS中性化 + MAD + z-score
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression

def main():
    data_dir = Path(__file__).parent.parent / "data"
    kline_path = data_dir / "csi1000_kline_raw.csv"
    
    print("读取数据...")
    df = pd.read_csv(kline_path, dtype={"stock_code": str})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    # 日收益率
    df["ret"] = df.groupby("stock_code")["close"].pct_change()
    
    # Amihud = |ret| / amount (百万)
    df["amihud"] = df["ret"].abs() / (df["amount"] / 1e6)
    # 排除amount=0
    df.loc[df["amount"] <= 0, "amihud"] = np.nan
    
    # pivot
    amihud_pivot = df.pivot(index="date", columns="stock_code", values="amihud")
    amt_pivot = df.pivot(index="date", columns="stock_code", values="amount")
    
    # 短期/长期Amihud
    amihud_5 = amihud_pivot.rolling(5, min_periods=3).mean()
    amihud_20 = amihud_pivot.rolling(20, min_periods=14).mean()
    
    # 20d平均成交额(用于中性化)
    amt_ma20 = amt_pivot.rolling(20, min_periods=14).mean()
    log_amt = np.log(amt_ma20.clip(lower=1))
    
    # ratio: -log(short/long) → 高=流动性改善
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = -np.log(amihud_5 / amihud_20)
    
    print(f"ratio均值: {np.nanmean(ratio.values):.4f}, 中位数: {np.nanmedian(ratio.values):.4f}")
    
    # 截面OLS中性化 + MAD + z-score
    print("截面中性化...")
    factor_records = []
    for dt in ratio.index:
        if dt not in log_amt.index:
            continue
        
        r = ratio.loc[dt]
        a = log_amt.loc[dt]
        
        valid = r.notna() & a.notna() & np.isfinite(r)
        if valid.sum() < 50:
            continue
        
        x = a[valid].values.reshape(-1, 1)
        y = r[valid].values
        codes = r[valid].index.tolist()
        
        # MAD winsorize
        med = np.median(y)
        mad = np.median(np.abs(y - med))
        if mad > 0:
            y = np.clip(y, med - 5 * 1.4826 * mad, med + 5 * 1.4826 * mad)
        
        lr = LinearRegression()
        lr.fit(x, y)
        resid = y - lr.predict(x)
        
        std = resid.std()
        if std > 0:
            resid = (resid - resid.mean()) / std
        
        for code, val in zip(codes, resid):
            factor_records.append({"date": dt, "stock_code": code, "factor_value": val})
    
    result = pd.DataFrame(factor_records)
    result = result.sort_values(["date", "stock_code"]).reset_index(drop=True)
    
    print(f"最终因子: {result.shape[0]} 行, {result['stock_code'].nunique()} 只股票")
    
    output_path = data_dir / "factor_amihud_change_v1.csv"
    result.to_csv(output_path, index=False)
    print(f"保存: {output_path}")

if __name__ == "__main__":
    main()
