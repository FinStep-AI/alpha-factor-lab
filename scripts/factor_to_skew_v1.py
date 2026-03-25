#!/usr/bin/env python3
"""
因子：换手率偏度 (Turnover Skewness)
公式：skew(turnover, 20d)
       → 成交额OLS中性化 + MAD winsorize + z-score
方向：测试两个方向

以及备选：换手率峰度 (Turnover Kurtosis) = kurtosis(turnover, 20d)
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

def neutralize_and_zscore(df, raw_col, x_col):
    """OLS中性化 + MAD winsorize + z-score"""
    sub = df[[raw_col, x_col]].dropna()
    if len(sub) < 50:
        return None
    raw = sub[raw_col].values
    x = sub[x_col].values
    X = np.column_stack([np.ones(len(x)), x])
    try:
        beta = np.linalg.lstsq(X, raw, rcond=None)[0]
        residuals = raw - X @ beta
    except:
        return None
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    if mad < 1e-12:
        return None
    clipped = np.clip(residuals, med - 5*1.4826*mad, med + 5*1.4826*mad)
    std = clipped.std()
    if std < 1e-12:
        return None
    z = (clipped - clipped.mean()) / std
    return pd.Series(z, index=sub.index)


def main():
    print("Loading data...")
    kline = pd.read_csv(DATA_DIR / 'csi1000_kline_raw.csv')
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    print(f"Data: {kline.shape[0]} rows, {kline['stock_code'].nunique()} stocks")
    
    # 换手率clip
    kline['to'] = kline['turnover'].clip(lower=0.01)
    
    # 20日换手率偏度
    print("Computing 20-day turnover skewness...")
    kline['to_skew_20'] = kline.groupby('stock_code')['to'].transform(
        lambda s: s.rolling(20, min_periods=15).skew()
    )
    
    # 20日换手率峰度
    print("Computing 20-day turnover kurtosis...")
    kline['to_kurt_20'] = kline.groupby('stock_code')['to'].transform(
        lambda s: s.rolling(20, min_periods=15).apply(lambda x: sp_stats.kurtosis(x, fisher=True), raw=True)
    )
    
    # 20日log成交额均值
    kline['log_amount'] = np.log(kline['amount'].clip(lower=1))
    kline['log_amount_20d'] = kline.groupby('stock_code')['log_amount'].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    
    # 处理换手率偏度因子
    print("\nProcessing turnover skewness factor...")
    results_skew = []
    for date, group in kline.groupby('date'):
        df = group[['stock_code', 'to_skew_20', 'log_amount_20d']].copy()
        z_series = neutralize_and_zscore(df, 'to_skew_20', 'log_amount_20d')
        if z_series is None:
            continue
        for idx in z_series.index:
            results_skew.append({
                'date': date,
                'stock_code': group.loc[idx, 'stock_code'],
                'factor_value': z_series[idx]
            })
    
    df_skew = pd.DataFrame(results_skew)
    print(f"Skew factor: {df_skew.shape[0]} rows, {df_skew['date'].nunique()} dates")
    df_skew.to_csv(DATA_DIR / 'factor_to_skew_v1.csv', index=False)
    
    # 处理换手率峰度因子
    print("\nProcessing turnover kurtosis factor...")
    results_kurt = []
    for date, group in kline.groupby('date'):
        df = group[['stock_code', 'to_kurt_20', 'log_amount_20d']].copy()
        z_series = neutralize_and_zscore(df, 'to_kurt_20', 'log_amount_20d')
        if z_series is None:
            continue
        for idx in z_series.index:
            results_kurt.append({
                'date': date,
                'stock_code': group.loc[idx, 'stock_code'],
                'factor_value': z_series[idx]
            })
    
    df_kurt = pd.DataFrame(results_kurt)
    print(f"Kurt factor: {df_kurt.shape[0]} rows, {df_kurt['date'].nunique()} dates")
    df_kurt.to_csv(DATA_DIR / 'factor_to_kurt_v1.csv', index=False)
    
    # 构造复合因子: 负偏度 + 低峰度 = 换手率平稳偏高
    # 也试试：log(turnover) * (1 - |skewness|/3) 抑制极端偏度
    print("\nProcessing negative skewness factor...")
    results_neg_skew = []
    for date, group in kline.groupby('date'):
        df = group[['stock_code', 'to_skew_20', 'log_amount_20d']].copy()
        df = df.dropna()
        if len(df) < 50:
            continue
        df['neg_skew'] = -df['to_skew_20']
        z_series = neutralize_and_zscore(df, 'neg_skew', 'log_amount_20d')
        if z_series is None:
            continue
        for idx in z_series.index:
            results_neg_skew.append({
                'date': date,
                'stock_code': group.loc[idx, 'stock_code'],
                'factor_value': z_series[idx]
            })
    
    df_neg_skew = pd.DataFrame(results_neg_skew)
    print(f"Neg-Skew factor: {df_neg_skew.shape[0]} rows, {df_neg_skew['date'].nunique()} dates")
    df_neg_skew.to_csv(DATA_DIR / 'factor_to_neg_skew_v1.csv', index=False)
    
    print("\nAll saved!")

if __name__ == '__main__':
    main()
