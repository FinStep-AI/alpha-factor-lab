#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
波动率的波动率因子 (Volatility of Volatility v1)
==================================================

学术来源:
  Baltussen, Van Bekkum & Groen-Xu (2019) "Forecasting Volatility-of-Volatility"
  Huang, Schlag, Shaliastovich & Thimme (2019) "Volatility-of-Volatility Risk" JFE

核心思路:
  - 先计算每日的"已实现波动率": 用(high-low)/close作为日内波动率代理(Parkinson estimator)
  - 再计算过去20天已实现波动率的标准差 = Vol-of-Vol
  - 高VoV = 波动率不稳定，风险更高
  
经济学直觉(两种假设，分别测试):
  假设A (正向): 高VoV承担更高不确定性风险 → 要求更高补偿 → 正向因子
    (类似中证1000的"高波动溢价"逻辑)
  假设B (反向): 高VoV = 不确定性太大 → 机构回避 → 流动性下降 → 负向
    
  在中证1000小盘股中，根据已有经验(高波动/高Amihud = 正向溢价)，
  预计假设A更可能成立: 高VoV → 高收益

处理:
  1. Parkinson波动率: σ_p = (high-low)/close
  2. 20日滚动std(σ_p) = VoV
  3. 对数变换(平滑右偏分布)
  4. 5%缩尾
  5. OLS log_amount_20d 中性化
  6. 截面z-score标准化
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")


def compute_vol_of_vol(kline_path: str, output_path: str, 
                       window: int = 20, direction: str = "positive"):
    """
    计算波动率的波动率因子。
    
    Parameters
    ----------
    direction : str
        "positive" = 高VoV→高收益 (做多高VoV)
        "negative" = 低VoV→高收益 (做多低VoV)
    """
    print(f"{'='*60}")
    print(f"Volatility-of-Volatility Factor (window={window}, dir={direction})")
    print(f"{'='*60}")
    
    # 1. Load data
    print("\n[1] 加载数据...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    n_stocks = df['stock_code'].nunique()
    n_dates = df['date'].nunique()
    print(f"    股票数: {n_stocks}, 日期数: {n_dates}")
    
    # 2. Parkinson volatility proxy: (high-low)/close
    print("[2] 计算Parkinson日内波动率代理...")
    df['parkinson_vol'] = (df['high'] - df['low']) / df['close']
    
    # Handle edge cases (limit-up/limit-down where high=low)
    df.loc[df['parkinson_vol'] < 1e-6, 'parkinson_vol'] = np.nan
    
    # 3. Rolling std of daily volatility = Vol-of-Vol
    print(f"[3] 计算{window}日滚动VoV...")
    def calc_vov(group):
        group = group.sort_values('date').copy()
        group['vov'] = group['parkinson_vol'].rolling(
            window, min_periods=int(window * 0.75)
        ).std()
        return group
    
    df = df.groupby('stock_code', group_keys=False).apply(calc_vov)
    
    # 4. Log transform (smooth right-skewed distribution)
    print("[4] 对数变换...")
    df['log_vov'] = np.log(df['vov'] + 1e-8)
    
    # 5. log_amount for neutralization
    print("[5] 计算市值代理...")
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(window, min_periods=int(window*0.75)).mean() + 1)
    )
    
    # 6. Winsorize 5%
    print("[6] 5%缩尾...")
    def winsorize_cs(group):
        v = group['log_vov']
        m = v.notna()
        if m.sum() < 10:
            return v
        lower = v[m].quantile(0.05)
        upper = v[m].quantile(0.95)
        return v.clip(lower, upper)
    
    df['log_vov'] = df.groupby('date', group_keys=False).apply(winsorize_cs)
    
    # 7. OLS neutralization
    print("[7] OLS市值中性化...")
    def neutralize(group):
        y = group['log_vov']
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
    
    # 8. Cross-sectional z-score
    print("[8] 截面z-score标准化...")
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
    
    # 9. Direction
    if direction == "negative":
        df['factor'] = -df['factor']
    
    # 10. Output
    print("[9] 输出因子值...")
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
    
    desc = output['factor_value'].describe()
    print(f"\n   均值: {desc['mean']:.4f}")
    print(f"   标准差: {desc['std']:.4f}")
    print(f"   范围: [{desc['min']:.4f}, {desc['max']:.4f}]")
    
    return output


if __name__ == "__main__":
    kline_path = sys.argv[1] if len(sys.argv) > 1 else "data/csi1000_kline_raw.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/factor_vol_of_vol_v1.csv"
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    direction = sys.argv[4] if len(sys.argv) > 4 else "positive"
    
    compute_vol_of_vol(kline_path, output_path, window, direction)
