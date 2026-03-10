#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日内方向强度因子 (Intraday Directional Strength v1)
=====================================================

核心思路:
  衡量日内价格走势的方向性强度:
  
  IDS = (close - open) / (high - low)
  
  当 close > open (阳线): IDS > 0, 值越大越强
  当 close < open (阴线): IDS < 0, 值越小越弱
  
  特殊情况: high = low (一字涨/跌停): 设为0
  
  20日滚动均值: factor = mean(IDS, 20d)
  
  经济学直觉:
  持续的强日内方向性(高IDS) = 买入力量持续强劲 → 趋势延续
  持续的弱日内方向性(低IDS) = 卖出力量持续强劲 → 趋势延续(跌势)
  
  正向使用: 高IDS → 高收益 (动量)
  
  这类似于"收盘位置"因子但包含了方向信息(阳线vs阴线)。
  
文献参考:
  - Brock, Lakonishok & LeBaron (1992) 技术分析有效性
  - A股实证: 日K线形态分析系列研报 (中金/国信)
"""

import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def compute_intraday_direction(kline_path: str, output_path: str, window: int = 20):
    print(f"{'='*60}")
    print(f"Intraday Directional Strength (window={window})")
    print(f"{'='*60}")
    
    # 1. Load data
    print("\n[1] 加载数据...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    n_stocks = df['stock_code'].nunique()
    n_dates = df['date'].nunique()
    print(f"    股票数: {n_stocks}, 日期数: {n_dates}")
    
    # 2. Intraday Directional Strength
    print("[2] 计算日内方向强度...")
    range_hl = df['high'] - df['low']
    df['ids'] = np.where(range_hl > 1e-4, 
                         (df['close'] - df['open']) / range_hl, 
                         0.0)
    
    # 3. Rolling mean
    print(f"[3] 计算{window}日滚动均值...")
    df['ids_mean'] = df.groupby('stock_code')['ids'].transform(
        lambda x: x.rolling(window, min_periods=int(window*0.75)).mean()
    )
    
    # 4. log_amount for neutralization
    print("[4] 计算市值代理...")
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(window, min_periods=int(window*0.75)).mean() + 1)
    )
    
    # 5. Winsorize 5%
    print("[5] 5%缩尾...")
    def winsorize_cs(group):
        v = group['ids_mean']
        m = v.notna()
        if m.sum() < 10:
            return v
        lower = v[m].quantile(0.05)
        upper = v[m].quantile(0.95)
        return v.clip(lower, upper)
    
    df['ids_mean'] = df.groupby('date', group_keys=False).apply(winsorize_cs)
    
    # 6. OLS neutralization
    print("[6] OLS市值中性化...")
    def neutralize(group):
        y = group['ids_mean']
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
    
    # 8. Direction: 先测正向(高IDS→高收益)
    # 如果不行再反转
    
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


if __name__ == "__main__":
    kline_path = sys.argv[1] if len(sys.argv) > 1 else "data/csi1000_kline_raw.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/factor_intraday_direction_v1.csv"
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    compute_intraday_direction(kline_path, output_path, window)
