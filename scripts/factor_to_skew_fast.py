#!/usr/bin/env python3
"""
因子：换手率偏度 (Turnover Skewness) - 快速版
正向和负向两个版本都输出
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

def main():
    print("Loading data...")
    kline = pd.read_csv(DATA_DIR / 'csi1000_kline_raw.csv')
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    print(f"Data: {kline.shape[0]} rows, {kline['stock_code'].nunique()} stocks")
    
    kline['to'] = kline['turnover'].clip(lower=0.01)
    
    # 20日换手率偏度 - pandas内置skew快很多
    print("Computing 20-day turnover skewness...")
    kline['to_skew_20'] = kline.groupby('stock_code')['to'].transform(
        lambda s: s.rolling(20, min_periods=15).skew()
    )
    
    # 20日log成交额均值
    kline['log_amount'] = np.log(kline['amount'].clip(lower=1))
    kline['log_amount_20d'] = kline.groupby('stock_code')['log_amount'].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    
    # 截面处理 - 正向(高偏度=高因子值) 和 负向(低偏度=高因子值)
    print("Cross-sectional processing...")
    results_pos = []
    results_neg = []
    
    for date, group in kline.groupby('date'):
        df = group[['stock_code', 'to_skew_20', 'log_amount_20d']].dropna()
        if len(df) < 50:
            continue
        
        raw = df['to_skew_20'].values
        x = df['log_amount_20d'].values
        
        # OLS中性化
        X = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(X, raw, rcond=None)[0]
            residuals = raw - X @ beta
        except:
            continue
        
        # MAD winsorize + z-score
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))
        if mad < 1e-12:
            continue
        clipped = np.clip(residuals, med - 5*1.4826*mad, med + 5*1.4826*mad)
        std = clipped.std()
        if std < 1e-12:
            continue
        z = (clipped - clipped.mean()) / std
        
        codes = df['stock_code'].values
        for i in range(len(codes)):
            results_pos.append({'date': date, 'stock_code': codes[i], 'factor_value': z[i]})
            results_neg.append({'date': date, 'stock_code': codes[i], 'factor_value': -z[i]})
    
    df_pos = pd.DataFrame(results_pos)
    df_neg = pd.DataFrame(results_neg)
    
    print(f"Factor: {df_pos.shape[0]} rows, {df_pos['date'].nunique()} dates")
    
    df_pos.to_csv(DATA_DIR / 'factor_to_skew_v1.csv', index=False)
    df_neg.to_csv(DATA_DIR / 'factor_to_neg_skew_v1.csv', index=False)
    print("Saved!")

if __name__ == '__main__':
    main()
