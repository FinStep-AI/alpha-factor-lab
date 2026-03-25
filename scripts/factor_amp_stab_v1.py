#!/usr/bin/env python3
"""
因子：振幅稳定性 (Amplitude Stability, AmpStab)
公式：-CV(amplitude, 20d) = -std(amp)/mean(amp)
       → 成交额OLS中性化 + MAD winsorize + z-score
方向：正向（低CV=稳定振幅=高因子值=高预期收益）
Barra风格：Quality / Low Volatility

逻辑：
- 低振幅变异系数 = 日内波动率平稳 = 无极端事件 = 价格发现有序
- 与TAE不同：TAE看换手率/振幅比(交易活跃度效率)，AmpStab看振幅自身的稳定性
- Quality代理：高质量股票振幅波动更稳定
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
    
    # amplitude字段已有（百分比），用它
    kline['amp'] = kline['amplitude'].clip(lower=0.01)
    
    # 20日振幅均值和标准差
    kline['amp_mean_20'] = kline.groupby('stock_code')['amp'].transform(
        lambda s: s.rolling(20, min_periods=15).mean()
    )
    kline['amp_std_20'] = kline.groupby('stock_code')['amp'].transform(
        lambda s: s.rolling(20, min_periods=15).std()
    )
    
    # CV = std/mean，取负（低CV → 高因子值）
    kline['raw_factor'] = -(kline['amp_std_20'] / (kline['amp_mean_20'] + 0.01))
    
    # 20日log成交额均值用于中性化
    kline['log_amount'] = np.log(kline['amount'].clip(lower=1))
    kline['log_amount_20d'] = kline.groupby('stock_code')['log_amount'].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    
    # 截面处理
    print("Cross-sectional neutralization...")
    results = []
    for date, group in kline.groupby('date'):
        df = group[['stock_code', 'raw_factor', 'log_amount_20d']].dropna()
        if len(df) < 50:
            continue
        
        raw = df['raw_factor'].values
        x = df['log_amount_20d'].values
        
        # OLS中性化
        X = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(X, raw, rcond=None)[0]
            residuals = raw - X @ beta
        except:
            continue
        
        # MAD winsorize
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))
        if mad < 1e-12:
            continue
        clipped = np.clip(residuals, med - 5*1.4826*mad, med + 5*1.4826*mad)
        
        # z-score
        std = clipped.std()
        if std < 1e-12:
            continue
        z = (clipped - clipped.mean()) / std
        
        for i, (_, row) in enumerate(df.iterrows()):
            results.append({
                'date': date,
                'stock_code': row['stock_code'],
                'factor_value': z[i]
            })
    
    result_df = pd.DataFrame(results)
    print(f"Factor computed: {result_df.shape[0]} rows, {result_df['date'].nunique()} dates")
    print(f"Mean: {result_df['factor_value'].mean():.4f}, Std: {result_df['factor_value'].std():.4f}")
    
    output_path = DATA_DIR / 'factor_amp_stab_v1.csv'
    result_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    main()
