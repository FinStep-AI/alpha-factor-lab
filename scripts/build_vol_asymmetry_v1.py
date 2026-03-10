#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: 量能不对称 (Volume Asymmetry v1)
============================================
思路:
  过去20日中，上涨日成交量占比 vs 下跌日成交量占比的不对称程度。
  VA = sum(volume * I(ret>0)) / sum(volume) - 0.5
  正值=买方力量主导, 负值=卖方力量主导。

  在中证1000小盘股中，信息传播慢，买方量能集中的股票
  后续动量延续概率更高（Campbell, Grossman & Wang 1993）。

构造:
  1. 日收益率 ret = pct_change（已有字段）
  2. 每日标记 up_day = (ret > 0)
  3. 滚动20日: up_vol_ratio = sum(volume * up_day, 20d) / sum(volume, 20d)
  4. VA = up_vol_ratio - 0.5 (居中化)
  5. 5% winsorize + 市值中性化(log_amount as proxy)

方向: 正向 (高VA = 买方主导 → 高收益)
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

def main():
    base = Path(__file__).resolve().parent.parent
    raw = pd.read_csv(base / "data" / "csi1000_kline_raw.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    # 确保有 pct_change 列，如果缺失或NaN太多就自行计算
    if "pct_change" not in raw.columns or raw["pct_change"].isna().mean() > 0.3:
        raw["pct_change"] = raw.groupby("stock_code")["close"].pct_change() * 100
    
    print(f"[INFO] 数据: {raw['stock_code'].nunique()} 只, {raw['date'].nunique()} 天")
    
    WINDOW = 20
    
    # 计算: up_day标记
    raw["up_day"] = (raw["pct_change"] > 0).astype(float)
    raw["vol_up"] = raw["volume"] * raw["up_day"]
    
    # 滚动20日求和
    def rolling_sum(group):
        group = group.sort_values("date")
        group["sum_vol_up"] = group["vol_up"].rolling(WINDOW, min_periods=15).sum()
        group["sum_vol"] = group["volume"].rolling(WINDOW, min_periods=15).sum()
        return group
    
    raw = raw.groupby("stock_code", group_keys=False).apply(rolling_sum)
    
    # VA = up_vol_ratio - 0.5
    raw["up_vol_ratio"] = raw["sum_vol_up"] / raw["sum_vol"].replace(0, np.nan)
    raw["vol_asymmetry"] = raw["up_vol_ratio"] - 0.5
    
    # 5% winsorize
    def winsorize(series, pct=0.05):
        lower = series.quantile(pct)
        upper = series.quantile(1 - pct)
        return series.clip(lower, upper)
    
    raw["vol_asymmetry"] = raw.groupby("date")["vol_asymmetry"].transform(winsorize)
    
    # 市值中性化: 用 log(20日平均成交额) 作为市值代理
    raw["log_amount"] = np.log1p(raw["amount"])
    
    def rolling_log_amount(group):
        group = group.sort_values("date")
        group["log_amount_20d"] = group["log_amount"].rolling(WINDOW, min_periods=15).mean()
        return group
    
    raw = raw.groupby("stock_code", group_keys=False).apply(rolling_log_amount)
    
    # OLS中性化: factor_resid = factor - beta * log_amount_20d - alpha
    from numpy.linalg import lstsq
    
    results = []
    for dt, grp in raw.groupby("date"):
        mask = grp["vol_asymmetry"].notna() & grp["log_amount_20d"].notna()
        sub = grp[mask].copy()
        if len(sub) < 50:
            continue
        
        y = sub["vol_asymmetry"].values
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
    out_path = base / "data" / "factor_vol_asymmetry_v1.csv"
    df_factor.to_csv(out_path, index=False)
    print(f"[INFO] 因子输出: {out_path}")
    print(f"[INFO] 因子样本: {len(df_factor)} 行, {df_factor['date'].nunique()} 天, {df_factor['stock_code'].nunique()} 股")
    print(f"[INFO] 因子均值: {df_factor['factor_value'].mean():.6f}, 标准差: {df_factor['factor_value'].std():.6f}")
    
    # 也输出收益率矩阵
    ret_path = base / "data" / "returns_for_backtest.csv"
    if not ret_path.exists():
        ret_df = raw[["date", "stock_code", "pct_change"]].copy()
        ret_df["pct_change"] = ret_df["pct_change"] / 100.0  # 转为小数
        ret_df.to_csv(ret_path, index=False)
        print(f"[INFO] 收益率输出: {ret_path}")
    
    return df_factor

if __name__ == "__main__":
    main()
