#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开盘跳空偏离度因子 (Opening Gap Bias v1)
==========================================

核心思路:
  衡量过去20天开盘跳空(开盘价 vs 前收盘价)的系统性偏差。
  
  gap_t = (open_t - close_{t-1}) / close_{t-1}
  
  factor = mean(gap, 20d)
  
  经济学直觉:
  - 持续负跳空 = 夜间持续有利空消息/情绪释放 → 日内反转机会 → 正向(买入低开股)
  - 持续正跳空 = 夜间追涨情绪 → 日内回落 → 反向
  
  在A股中证1000中，开盘集合竞价受散户情绪影响大:
  - 散户倾向在开盘时恐慌卖出(低开) → 过度反应 → 后续反弹
  - 散户倾向在开盘时追高买入(高开) → 过度反应 → 后续回落
  
  因此我们 **反向** 使用: factor = -mean(gap_20d)
  
文献参考:
  - Gao, Han, Li & Zhou (2018) "Market Intraday Momentum" JFE
  - Bogousslavsky (2016) "Infrequent Rebalancing, Return Autocorrelation, and Seasonality"
  
处理:
  1. 计算逐日跳空 gap
  2. 20日滚动均值
  3. 5%缩尾
  4. OLS log_amount_20d 中性化
  5. 截面z-score标准化
  6. 取反(做多负跳空=低开股)
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")


def compute_opening_gap_bias(kline_path: str, output_path: str, window: int = 20):
    print(f"{'='*60}")
    print(f"Opening Gap Bias Factor (window={window})")
    print(f"{'='*60}")
    
    # 1. Load data
    print("\n[1] 加载数据...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    n_stocks = df['stock_code'].nunique()
    n_dates = df['date'].nunique()
    print(f"    股票数: {n_stocks}, 日期数: {n_dates}")
    
    # 2. Overnight gap
    print("[2] 计算隔夜跳空...")
    df['prev_close'] = df.groupby('stock_code')['close'].shift(1)
    df['gap'] = (df['open'] - df['prev_close']) / df['prev_close']
    
    # Clip extreme gaps (limit up/down: ±20% for CSI1000 stocks)
    df['gap'] = df['gap'].clip(-0.21, 0.21)
    
    # 3. Rolling mean of gap
    print(f"[3] 计算{window}日滚动均值...")
    def calc_gap_mean(group):
        group = group.sort_values('date').copy()
        group['gap_mean'] = group['gap'].rolling(
            window, min_periods=int(window * 0.75)
        ).mean()
        return group
    
    df = df.groupby('stock_code', group_keys=False).apply(calc_gap_mean)
    
    # 4. log_amount for neutralization
    print("[4] 计算市值代理...")
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(window, min_periods=int(window*0.75)).mean() + 1)
    )
    
    # 5. Winsorize 5%
    print("[5] 5%缩尾...")
    def winsorize_cs(group):
        v = group['gap_mean']
        m = v.notna()
        if m.sum() < 10:
            return v
        lower = v[m].quantile(0.05)
        upper = v[m].quantile(0.95)
        return v.clip(lower, upper)
    
    df['gap_mean'] = df.groupby('date', group_keys=False).apply(winsorize_cs)
    
    # 6. OLS neutralization
    print("[6] OLS市值中性化...")
    def neutralize(group):
        y = group['gap_mean']
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
    
    # 7. Cross-sectional z-score
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
    
    # 8. Negate: 做多低开股(反向)
    df['factor'] = -df['factor']
    
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
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/factor_opening_gap_bias_v1.csv"
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    compute_opening_gap_bias(kline_path, output_path, window)
