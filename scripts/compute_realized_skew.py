#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realized Skewness Factor (realized_skew_v1)

复现论文: Amaya, Christoffersen, Jacobs & Vasquez (2015)
"Does Realized Skewness Predict the Cross-Section of Equity Returns?"
Journal of Financial Economics, 116(1), 132-152.

原文发现:
- 低realized skewness的股票后续收益更高（负偏度溢价）
- 投资者偏好正偏度（彩票型）资产，导致正偏度股票被高估
- 使用5分钟高频数据计算周度realized skewness

本土化适配:
- 使用日频收益率(无高频数据)，计算20日滚动偏度
- 反向使用: 低skewness → 做多（投资者厌恶负偏度，要求风险补偿）
- 市值中性化
- 股票池: 中证1000
"""

import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


def compute_realized_skewness(kline_path: str, output_path: str, window: int = 20):
    """
    计算realized skewness因子。
    
    公式: skewness(daily_returns, window=20), 然后反向 + 市值中性化
    
    Amaya et al. 原文用5min高频数据算realized skewness:
      RSkew = sqrt(N) * sum(r_i^3) / (sum(r_i^2))^(3/2)
    
    我们用日频简化版: 
      rolling skewness of daily returns (20-day window)
    然后取负值（低偏度 → 高因子值 → 做多）
    """
    print("=" * 60)
    print("Realized Skewness Factor (Amaya et al. 2015 JFE)")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1] 加载数据...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    n_stocks = df['stock_code'].nunique()
    n_dates = df['date'].nunique()
    print(f"    股票数: {n_stocks}, 日期数: {n_dates}")
    print(f"    日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    
    # 2. Compute daily returns
    print("\n[2] 计算日收益率...")
    df = df.sort_values(['stock_code', 'date'])
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    
    # 3. Compute rolling realized skewness (20-day)
    print(f"\n[3] 计算{window}日滚动偏度...")
    
    def rolling_skew(group, win):
        """计算滚动偏度"""
        return group['ret'].rolling(window=win, min_periods=max(win // 2, 10)).skew()
    
    df['raw_skew'] = df.groupby('stock_code', group_keys=False).apply(
        lambda g: rolling_skew(g, window)
    )
    
    # 4. 反向: 低偏度 → 高因子值（做多低偏度股票）
    # Amaya et al.: buying low skewness, selling high skewness → positive return
    print("\n[4] 反向处理（低偏度 → 高因子值）...")
    df['neg_skew'] = -df['raw_skew']
    
    # 5. 市值中性化 (使用 log(amount) 作为市值代理)
    print("\n[5] 市值中性化...")
    # 计算20日平均成交额作为市值代理
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    
    def neutralize_cross_section(group):
        """截面上对市值做OLS回归，取残差"""
        y = group['neg_skew']
        x = group['log_amount_20d']
        
        mask = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            return pd.Series(np.nan, index=group.index)
        
        y_clean = y[mask]
        x_clean = x[mask]
        
        # OLS: y = a + b*x + epsilon
        X = np.column_stack([np.ones(len(x_clean)), x_clean.values])
        try:
            beta = np.linalg.lstsq(X, y_clean.values, rcond=None)[0]
            residuals = y_clean.values - X @ beta
            result = pd.Series(np.nan, index=group.index)
            result[mask] = residuals
            return result
        except:
            return pd.Series(np.nan, index=group.index)
    
    df['factor'] = df.groupby('date', group_keys=False).apply(neutralize_cross_section)
    
    # 6. Winsorize (MAD 3倍)
    print("\n[6] Winsorize (MAD 3x)...")
    def winsorize_mad(group):
        vals = group['factor']
        mask = vals.notna()
        if mask.sum() < 10:
            return vals
        clean = vals[mask]
        median = clean.median()
        mad = np.median(np.abs(clean - median))
        if mad < 1e-10:
            return vals
        lower = median - 3 * 1.4826 * mad
        upper = median + 3 * 1.4826 * mad
        return vals.clip(lower, upper)
    
    df['factor'] = df.groupby('date', group_keys=False).apply(winsorize_mad)
    
    # 7. 截面z-score标准化
    print("\n[7] 截面z-score标准化...")
    def zscore_cs(group):
        vals = group['factor']
        mask = vals.notna()
        if mask.sum() < 10:
            return vals
        mean = vals[mask].mean()
        std = vals[mask].std()
        if std < 1e-10:
            return vals * 0
        return (vals - mean) / std
    
    df['factor'] = df.groupby('date', group_keys=False).apply(zscore_cs)
    
    # 8. 输出
    print(f"\n[8] 输出因子值...")
    output = df[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    output = output.rename(columns={'factor': 'factor_value'})
    output.to_csv(output_path, index=False)
    
    n_valid = output.shape[0]
    n_dates_valid = output['date'].nunique()
    n_stocks_valid = output.groupby('date')['stock_code'].nunique().median()
    
    print(f"    有效记录: {n_valid:,}")
    print(f"    有效日期: {n_dates_valid}")
    print(f"    平均每日股票数: {n_stocks_valid:.0f}")
    
    # 9. 基本统计
    print("\n[9] 因子基本统计:")
    desc = output['factor_value'].describe()
    print(f"    均值: {desc['mean']:.4f}")
    print(f"    标准差: {desc['std']:.4f}")
    print(f"    最小值: {desc['min']:.4f}")
    print(f"    最大值: {desc['max']:.4f}")
    
    # 分布检查
    print("\n[10] 分布检查（每日截面偏度均值）:")
    daily_skew = output.groupby('date')['factor_value'].skew().mean()
    daily_kurt = output.groupby('date')['factor_value'].apply(lambda x: x.kurtosis()).mean()
    print(f"    截面偏度均值: {daily_skew:.4f}")
    print(f"    截面超额峰度均值: {daily_kurt:.4f}")
    
    print(f"\n✅ 因子计算完成! 输出: {output_path}")
    return output


if __name__ == "__main__":
    kline_path = sys.argv[1] if len(sys.argv) > 1 else "data/csi1000_kline_raw.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/factor_realized_skew_v1.csv"
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    compute_realized_skewness(kline_path, output_path, window)
