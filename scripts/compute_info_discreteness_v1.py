#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信息离散度因子 (Information Discreteness v1)
=============================================

学术来源:
  Da, Gurun & Warachka (2014) "Frog in the Pan: Continuous Information 
  and Momentum" Review of Financial Studies 27(7):2171-2218
  https://doi.org/10.1093/rfs/hhu003

核心思路:
  - 过去20日的累计收益可以由"少数大涨/跌日"驱动(离散)或"持续小涨/跌日"驱动(连续)
  - 连续型上涨(每天都涨一点) → 投资者过度关注 → 后续容易反转
  - 离散型上涨(偶尔大涨) → 投资者注意力不足 → 动量延续
  
公式:
  ID = sign(R_20d) × (% positive days - % negative days)
  
  - R_20d > 0 且大部分天都涨 → ID 高 (连续上涨) → 反转信号
  - R_20d > 0 但仅少数天涨 → ID 低 (离散上涨) → 动量延续
  - R_20d < 0 且大部分天都跌 → ID 高 (连续下跌) → 反转信号  
  - R_20d < 0 但仅少数天跌 → ID 低 (离散下跌) → 动量延续

方向:
  反向使用 → 做多低ID(离散)，做空高ID(连续)
  
处理:
  1. 20日滚动计算 ID
  2. 5% 缩尾
  3. OLS 市值(log_amount_20d)中性化
  4. 截面 z-score 标准化
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")


def compute_info_discreteness(kline_path: str, output_path: str, window: int = 20):
    """
    计算信息离散度因子。
    """
    print(f"{'='*60}")
    print(f"Information Discreteness Factor (window={window})")
    print(f"{'='*60}")
    
    # 1. Load data
    print("\n[1] 加载数据...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    n_stocks = df['stock_code'].nunique()
    n_dates = df['date'].nunique()
    print(f"    股票数: {n_stocks}, 日期数: {n_dates}")
    
    # 2. Daily returns
    print("[2] 计算日收益率...")
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    
    # 3. Rolling Information Discreteness
    print(f"[3] 计算{window}日滚动信息离散度...")
    
    def calc_id(group):
        group = group.sort_values('date').copy()
        ret = group['ret'].values
        n = len(ret)
        id_vals = np.full(n, np.nan)
        
        for i in range(window, n):  # need window days of returns
            r = ret[i - window + 1:i + 1]
            valid = ~np.isnan(r)
            if valid.sum() < window * 0.75:  # need at least 75% valid
                continue
            
            r_valid = r[valid]
            cum_ret = np.sum(r_valid)  # approximate cumulative return
            
            n_pos = np.sum(r_valid > 0)
            n_neg = np.sum(r_valid < 0)
            n_total = len(r_valid)
            
            pct_pos = n_pos / n_total
            pct_neg = n_neg / n_total
            
            # ID = sign(cum_ret) * (pct_pos - pct_neg)
            if cum_ret > 0:
                id_val = pct_pos - pct_neg
            elif cum_ret < 0:
                id_val = -(pct_pos - pct_neg)  # sign(-1) * (pct_pos - pct_neg)
            else:
                id_val = 0.0
            
            id_vals[i] = id_val
        
        group['raw_id'] = id_vals
        return group
    
    df = df.groupby('stock_code', group_keys=False).apply(calc_id)
    
    # 4. log_amount for neutralization
    print("[4] 计算市值代理...")
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(window, min_periods=int(window*0.75)).mean() + 1)
    )
    
    # 5. Winsorize 5%
    print("[5] 5%缩尾...")
    def winsorize_cs(group):
        v = group['raw_id']
        m = v.notna()
        if m.sum() < 10:
            return v
        lower = v[m].quantile(0.05)
        upper = v[m].quantile(0.95)
        return v.clip(lower, upper)
    
    df['raw_id'] = df.groupby('date', group_keys=False).apply(winsorize_cs)
    
    # 6. OLS neutralization
    print("[6] OLS市值中性化...")
    def neutralize(group):
        y = group['raw_id']
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
    
    # 8. Negate for direction: low ID (discrete) = high factor value
    # We want to short high-ID (continuous) and long low-ID (discrete)
    # So negate: factor = -ID
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
    
    return output


if __name__ == "__main__":
    kline_path = sys.argv[1] if len(sys.argv) > 1 else "data/csi1000_kline_raw.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/factor_info_discreteness_v1.csv"
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    compute_info_discreteness(kline_path, output_path, window)
