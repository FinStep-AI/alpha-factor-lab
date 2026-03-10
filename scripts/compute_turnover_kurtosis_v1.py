#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
换手率峰度因子 (Turnover Kurtosis v1)
=======================================

核心思路:
  计算过去20天日换手率的峰度(excess kurtosis)。
  
  高峰度 = 换手率分布呈尖峰厚尾 = 偶尔爆量交易 = 事件驱动型
  低峰度 = 换手率均匀分布 = 稳定型
  
  在中证1000中:
  - 高峰度(间歇性爆量) → 可能反映游资/庄家行为 → 短期波动大
  - 低峰度(稳定换手) → 更可能是基本面驱动的缓慢趋势
  
  两个方向测试:
  方案A: 做多低峰度(稳定换手 = 趋势确认) 
  方案B: 做多高峰度(事件驱动 = 信息溢价)

处理:
  1. 20日滚动excess kurtosis(turnover)
  2. 5%缩尾 → OLS log_amount_20d 中性化 → z-score
"""

import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


def compute_turnover_kurtosis(kline_path: str, output_path: str, 
                               window: int = 20, direction: str = "negative"):
    """
    direction: "negative" = 做多低峰度, "positive" = 做多高峰度
    """
    dir_str = direction
    print(f"{'='*60}")
    print(f"Turnover Kurtosis Factor (window={window}, dir={dir_str})")
    print(f"{'='*60}")
    
    # 1. Load data
    print("\n[1] 加载数据...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    n_stocks = df['stock_code'].nunique()
    n_dates = df['date'].nunique()
    print(f"    股票数: {n_stocks}, 日期数: {n_dates}")
    
    # 2. Rolling kurtosis of turnover
    print(f"[2] 计算{window}日换手率峰度...")
    def calc_kurt(group):
        group = group.sort_values('date').copy()
        to = group['turnover'].values.astype(float)
        n = len(to)
        kurt_vals = np.full(n, np.nan)
        
        for i in range(window - 1, n):
            t = to[i - window + 1:i + 1]
            valid = ~np.isnan(t) & (t > 0)
            if valid.sum() < int(window * 0.75):
                continue
            t_valid = t[valid]
            if np.std(t_valid) < 1e-10:
                kurt_vals[i] = 0.0
                continue
            # Use Fisher's definition (excess kurtosis)
            kurt_vals[i] = stats.kurtosis(t_valid, fisher=True)
        
        group['to_kurt'] = kurt_vals
        return group
    
    df = df.groupby('stock_code', group_keys=False).apply(calc_kurt)
    
    # 3. log_amount for neutralization
    print("[3] 计算市值代理...")
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(window, min_periods=int(window*0.75)).mean() + 1)
    )
    
    # 4. Winsorize 5%
    print("[4] 5%缩尾...")
    def winsorize_cs(group):
        v = group['to_kurt']
        m = v.notna()
        if m.sum() < 10:
            return v
        lower = v[m].quantile(0.05)
        upper = v[m].quantile(0.95)
        return v.clip(lower, upper)
    
    df['to_kurt'] = df.groupby('date', group_keys=False).apply(winsorize_cs)
    
    # 5. OLS neutralization
    print("[5] OLS市值中性化...")
    def neutralize(group):
        y = group['to_kurt']
        x = group['log_amount_20d']
        mask = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            return pd.Series(np.nan, index=group.index)
        y_c = y[mask].values
        x_c = x[mask].values
        X = np.column_stack([np.ones(len(x_c)), x_c])
        try:
            beta = np.linalg.lstsq(X, y_c, rcond=None)[0]
            residuals = y_c - X @ beta
            result = pd.Series(np.nan, index=group.index)
            result[mask] = residuals
            return result
        except Exception:
            return pd.Series(np.nan, index=group.index)
    
    df['factor'] = df.groupby('date', group_keys=False).apply(neutralize)
    
    # 6. Cross-sectional z-score
    print("[6] 截面z-score标准化...")
    def zscore_cs(group):
        v = group['factor']
        m = v.notna()
        if m.sum() < 10:
            return v
        mu = v[m].mean()
        s = v[m].std()
        if s < 1e-10:
            return v * 0
        return (v - mu) / s
    
    df['factor'] = df.groupby('date', group_keys=False).apply(zscore_cs)
    
    # 7. Direction
    if direction == "negative":
        df['factor'] = -df['factor']
    
    # 8. Output
    print("[7] 输出因子值...")
    output = df[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    output = output.rename(columns={'factor': 'factor_value'})
    output.to_csv(output_path, index=False)
    
    n_valid = output.shape[0]
    n_dates_valid = output['date'].nunique()
    avg_stocks = output.groupby('date')['stock_code'].nunique().median()
    
    print(f"\n✅ 因子计算完成!")
    print(f"   输出: {output_path}")
    print(f"   有效记录: {n_valid:,}")
    print(f"   有效日期: {n_dates_valid}")
    print(f"   平均每日股票数: {avg_stocks:.0f}")


if __name__ == "__main__":
    kline_path = sys.argv[1] if len(sys.argv) > 1 else "data/csi1000_kline_raw.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/factor_turnover_kurt_v1.csv"
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    direction = sys.argv[4] if len(sys.argv) > 4 else "negative"
    
    compute_turnover_kurtosis(kline_path, output_path, window, direction)
