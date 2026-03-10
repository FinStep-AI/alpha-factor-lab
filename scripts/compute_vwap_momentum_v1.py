#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VWAP动量因子 (VWAP Momentum v1)
==================================

核心思路:
  用VWAP(成交量加权平均价)的20日动量替代传统close-to-close动量。
  
  vwap_mom_20d = (vwap_t / vwap_{t-20}) - 1
  
  为什么VWAP动量更好:
  1. VWAP反映了每天交易的"真实平均成本"
  2. 不受尾盘操纵/异动影响(close容易被大单操纵)
  3. 隐含了成交量信息(高量区间权重大)
  
  在中证1000中:
  - VWAP动量与close动量的差异 = 微观结构噪音
  - 使用VWAP可以过滤掉收盘竞价噪音
  
  方向:
  先试反向(中证1000的短期反转效应): 做空VWAP动量高的
  如果不行再试正向

处理:
  1. 从kline数据计算VWAP = amount / volume
  2. 计算20日VWAP动量
  3. 反向使用(做多VWAP动量低的)
  4. 5%缩尾 → OLS市值中性化 → z-score
"""

import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def compute_vwap_momentum(kline_path: str, output_path: str, 
                           window: int = 20, reverse: bool = True):
    dir_str = "reverse" if reverse else "positive"
    print(f"{'='*60}")
    print(f"VWAP Momentum Factor (window={window}, dir={dir_str})")
    print(f"{'='*60}")
    
    # 1. Load data
    print("\n[1] 加载数据...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    n_stocks = df['stock_code'].nunique()
    n_dates = df['date'].nunique()
    print(f"    股票数: {n_stocks}, 日期数: {n_dates}")
    
    # 2. VWAP
    print("[2] 计算VWAP...")
    # VWAP = amount / (volume * 100)
    # amount is in RMB, volume is in 手(lots of 100 shares)
    df['vwap'] = df['amount'] / (df['volume'] * 100 + 1e-8)
    
    # Sanity check: vwap should be close to close price
    ratio = df['vwap'] / (df['close'] + 1e-8)
    df.loc[(ratio < 0.8) | (ratio > 1.2) | (df['amount'] < 1), 'vwap'] = np.nan
    
    # 3. VWAP momentum
    print(f"[3] 计算{window}日VWAP动量...")
    df['vwap_lag'] = df.groupby('stock_code')['vwap'].shift(window)
    df['vwap_mom'] = (df['vwap'] / df['vwap_lag']) - 1
    
    # Clip extreme values (beyond ±100% in 20 days is unusual but possible)
    df['vwap_mom'] = df['vwap_mom'].clip(-1.0, 2.0)
    
    # 4. Direction
    if reverse:
        df['vwap_mom'] = -df['vwap_mom']
    
    # 5. log_amount for neutralization
    print("[4] 计算市值代理...")
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(window, min_periods=int(window*0.75)).mean() + 1)
    )
    
    # 6. Winsorize 5%
    print("[5] 5%缩尾...")
    def winsorize_cs(group):
        v = group['vwap_mom']
        m = v.notna()
        if m.sum() < 10:
            return v
        lower = v[m].quantile(0.05)
        upper = v[m].quantile(0.95)
        return v.clip(lower, upper)
    
    df['vwap_mom'] = df.groupby('date', group_keys=False).apply(winsorize_cs)
    
    # 7. OLS neutralization
    print("[6] OLS市值中性化...")
    def neutralize(group):
        y = group['vwap_mom']
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
    print("[7] 截面z-score标准化...")
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
    
    # 9. Output
    print("[8] 输出因子值...")
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


if __name__ == "__main__":
    kline_path = sys.argv[1] if len(sys.argv) > 1 else "data/csi1000_kline_raw.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/factor_vwap_momentum_v1.csv"
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    reverse = (sys.argv[4] if len(sys.argv) > 4 else "reverse") == "reverse"
    
    compute_vwap_momentum(kline_path, output_path, window, reverse)
