#!/usr/bin/env python3
"""
因子: 高低价位收敛度 (High-Low Convergence) v1
公式: -log(MA5(high-low)/close / MA60(high-low)/close)
     = -log(MA5(amplitude) / MA60(amplitude))
     
简化: -log(MA5_amp / MA60_amp), 成交额中性化

逻辑:
- 衡量振幅是否在收窄(蓄势)还是扩张
- 负对数: 高值 = 近期振幅 << 长期振幅 = 波动率压缩 = 蓄势
- amp_compress_v1 之前测过(IC=0.022,t=2.06,mono=0.7) 用的是5/20窗口
- 改进: 用5/60窗口，更长的基准期应该更稳定
- 也测试正方向(高振幅扩张=高收益)

Barra风格: Volatility (短期vs长期波动率关系)
"""

import numpy as np
import pandas as pd
import sys, os

def compute_factor():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'csi1000_kline_raw.csv')
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    print(f"数据: {len(df)} 行, {df['stock_code'].nunique()} 股, {df['date'].nunique()} 日")
    
    g = df.groupby('stock_code')
    
    # Amplitude = (high - low) / prev_close
    df['prev_close'] = g['close'].shift(1)
    df['amplitude'] = (df['high'] - df['low']) / df['prev_close'].replace(0, np.nan)
    
    # MA5 and MA60 of amplitude
    df['amp_ma5'] = g['amplitude'].transform(lambda x: x.rolling(5, min_periods=3).mean())
    df['amp_ma60'] = g['amplitude'].transform(lambda x: x.rolling(60, min_periods=30).mean())
    
    # Amplitude convergence = log(MA5_amp / MA60_amp)
    # Positive = recent amp expanding, Negative = recent amp contracting
    ratio = df['amp_ma5'] / df['amp_ma60']
    ratio = ratio.replace([0, np.inf, -np.inf], np.nan)
    df['raw_factor'] = np.log(ratio)
    
    # log_amount_20d for neutralization
    df['log_amount_20d'] = g['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean().replace(0, np.nan))
    )
    
    # Also compute log_amount_60d
    df['log_amount_60d'] = g['amount'].transform(
        lambda x: np.log(x.rolling(60, min_periods=30).mean().replace(0, np.nan))
    )
    
    print(f"原始因子有效值: {df['raw_factor'].notna().sum()}/{len(df)} ({df['raw_factor'].notna().mean()*100:.1f}%)")
    
    # Cross-sectional processing
    results = []
    for date, group in df.groupby('date'):
        sub = group[['stock_code', 'raw_factor', 'log_amount_60d']].dropna()
        if len(sub) < 30:
            continue
        
        y = sub['raw_factor'].values.copy()
        x = sub['log_amount_60d'].values
        
        # MAD winsorize
        med = np.median(y)
        mad = np.median(np.abs(y - med))
        if mad > 0:
            bound = 3 * 1.4826 * mad
            y = np.clip(y, med - bound, med + bound)
        
        # OLS neutralize
        X = np.column_stack([np.ones(len(x)), x])
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 10:
            continue
        try:
            XtX = X[mask].T @ X[mask]
            Xty = X[mask].T @ y[mask]
            beta = np.linalg.solve(XtX + 1e-10 * np.eye(2), Xty)
            resid = y - X @ beta
        except:
            resid = y
        
        # Second MAD winsorize
        med2 = np.median(resid)
        mad2 = np.median(np.abs(resid - med2))
        if mad2 > 0:
            bound2 = 3 * 1.4826 * mad2
            resid = np.clip(resid, med2 - bound2, med2 + bound2)
        
        # Z-score
        std = np.std(resid)
        if std > 0:
            zscore = (resid - np.mean(resid)) / std
        else:
            zscore = resid * 0
        
        for i, idx in enumerate(sub.index):
            results.append({
                'date': date,
                'stock_code': sub.loc[idx, 'stock_code'],
                'factor': zscore[i]
            })
    
    result_df = pd.DataFrame(results)
    result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')
    
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'factor_amp_conv_v1.csv')
    result_df.to_csv(out_path, index=False)
    
    # Also save flipped version
    result_neg = result_df.copy()
    result_neg['factor'] = -result_neg['factor']
    out_path_neg = os.path.join(os.path.dirname(__file__), '..', 'data', 'factor_amp_conv_neg_v1.csv')
    result_neg.to_csv(out_path_neg, index=False)
    
    print(f"\n因子已保存: {out_path}")
    print(f"反向因子已保存: {out_path_neg}")
    print(f"有效值: {result_df['factor'].notna().sum()}")
    print(f"均值: {result_df['factor'].mean():.4f}")
    print(f"标准差: {result_df['factor'].std():.4f}")
    print(f"日期范围: {result_df['date'].min()} ~ {result_df['date'].max()}")

if __name__ == '__main__':
    compute_factor()
