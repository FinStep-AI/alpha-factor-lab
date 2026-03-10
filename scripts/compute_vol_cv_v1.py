#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成交量离散度因子 (Volume Dispersion / CV of Volume v1)
========================================================

核心思路:
  用日成交量的变异系数(CV = std/mean)衡量成交量分布的集中度。
  
  高CV = 成交量波动大，某几天爆量、其余日冷清 → 机构/大单间歇性交易
  低CV = 成交量平稳 → 散户持续交易
  
  假设: 高CV(间歇性放量)的股票，机构参与度高，信息含量大，
  后续20天有动量延续(正向因子)。

  另一种解释(反向):
  高CV = 交易不稳定 = 流动性风险大 → 被回避 → 反向
  
  两个方向都测试。

构造:
  1. 20日滚动 CV(volume) = std(volume_20d) / mean(volume_20d)
  2. 5%缩尾
  3. OLS log_amount_20d 中性化
  4. 截面z-score标准化
  
相关文献:
  - Chordia, Roll & Subrahmanyam (2001) "Market Liquidity and Trading Activity"
  - Llorente et al. (2002) "Dynamic Volume-Return Relation"
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")


def compute_volume_cv(kline_path: str, output_path: str, window: int = 20):
    """计算成交量变异系数因子。"""
    print(f"{'='*60}")
    print(f"Volume CV Factor (window={window})")
    print(f"{'='*60}")
    
    # 1. Load data
    print("\n[1] 加载数据...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    n_stocks = df['stock_code'].nunique()
    n_dates = df['date'].nunique()
    print(f"    股票数: {n_stocks}, 日期数: {n_dates}")
    
    # 2. Rolling CV of volume
    print(f"[2] 计算{window}日成交量CV...")
    def calc_vol_cv(group):
        group = group.sort_values('date').copy()
        vol = group['volume'].astype(float)
        roll_mean = vol.rolling(window, min_periods=int(window*0.75)).mean()
        roll_std = vol.rolling(window, min_periods=int(window*0.75)).std()
        group['vol_cv'] = roll_std / (roll_mean + 1e-8)
        return group
    
    df = df.groupby('stock_code', group_keys=False).apply(calc_vol_cv)
    
    # 3. log_amount for neutralization
    print("[3] 计算市值代理...")
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(window, min_periods=int(window*0.75)).mean() + 1)
    )
    
    # 4. Winsorize 5%
    print("[4] 5%缩尾...")
    def winsorize_cs(group):
        v = group['vol_cv']
        m = v.notna()
        if m.sum() < 10:
            return v
        lower = v[m].quantile(0.05)
        upper = v[m].quantile(0.95)
        return v.clip(lower, upper)
    
    df['vol_cv'] = df.groupby('date', group_keys=False).apply(winsorize_cs)
    
    # 5. OLS neutralization
    print("[5] OLS市值中性化...")
    def neutralize(group):
        y = group['vol_cv']
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
    
    # 7. Output (positive direction: high CV = high factor)
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
    
    desc = output['factor_value'].describe()
    print(f"\n   均值: {desc['mean']:.4f}")
    print(f"   标准差: {desc['std']:.4f}")
    print(f"   范围: [{desc['min']:.4f}, {desc['max']:.4f}]")


if __name__ == "__main__":
    kline_path = sys.argv[1] if len(sys.argv) > 1 else "data/csi1000_kline_raw.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/factor_vol_cv_v1.csv"
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    compute_volume_cv(kline_path, output_path, window)
