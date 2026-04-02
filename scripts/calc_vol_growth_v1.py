#!/usr/bin/env python3
"""
因子: 成交量增速 (Volume Growth Rate) v1
公式: log(MA5(volume) / MA60(volume)), 成交额OLS中性化 + MAD winsorize + z-score

逻辑:
- MA5/MA60 衡量近期成交量相对长期均值的偏离程度
- 正值 = 近期放量 = 市场关注度/参与度提升
- 负值 = 近期缩量 = 市场关注度/参与度下降
- 与turnover_level(20日换手水平)不同: 本因子捕捉的是成交量的变化趋势(动态)，而非水平(静态)
- 与vol_cv_neg(成交量稳定性)不同: 本因子看方向(增/减)，vol_cv_neg看波动(稳/不稳)

假说: 成交量持续放大的股票正在获得更多投资者关注，信息在加速传播，
      价格发现更充分，后续有正alpha (注意力效应/信息扩散假说)
      
Barra风格: Growth代理 (成交量增长反映公司被市场"发现"的速度)
"""

import numpy as np
import pandas as pd
import sys
import os

def compute_factor():
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'csi1000_kline_raw.csv')
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    print(f"数据加载: {len(df)} 行, {df['stock_code'].nunique()} 只股票, {df['date'].nunique()} 个交易日")
    
    # Compute MA5 and MA60 of volume per stock
    g = df.groupby('stock_code')
    df['vol_ma5'] = g['volume'].transform(lambda x: x.rolling(5, min_periods=3).mean())
    df['vol_ma60'] = g['volume'].transform(lambda x: x.rolling(60, min_periods=30).mean())
    
    # Volume growth rate = log(MA5 / MA60)
    ratio = df['vol_ma5'] / df['vol_ma60']
    ratio = ratio.replace([0, np.inf, -np.inf], np.nan)
    df['raw_factor'] = np.log(ratio)
    
    # Also compute log_amount_20d for neutralization
    df['log_amount_20d'] = g['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean().replace(0, np.nan))
    )
    
    print(f"原始因子有效值: {df['raw_factor'].notna().sum()}/{len(df)} ({df['raw_factor'].notna().mean()*100:.1f}%)")
    
    # Cross-sectional processing: each date
    results = []
    for date, group in df.groupby('date'):
        sub = group[['stock_code', 'raw_factor', 'log_amount_20d']].dropna()
        if len(sub) < 30:
            continue
        
        y = sub['raw_factor'].values
        x = sub['log_amount_20d'].values
        
        # MAD winsorize
        med = np.median(y)
        mad = np.median(np.abs(y - med))
        if mad > 0:
            lower = med - 3 * 1.4826 * mad
            upper = med + 3 * 1.4826 * mad
            y = np.clip(y, lower, upper)
        
        # OLS neutralize by log_amount_20d
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
        
        # Second MAD winsorize on residuals
        med2 = np.median(resid)
        mad2 = np.median(np.abs(resid - med2))
        if mad2 > 0:
            lower2 = med2 - 3 * 1.4826 * mad2
            upper2 = med2 + 3 * 1.4826 * mad2
            resid = np.clip(resid, lower2, upper2)
        
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
    
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'factor_vol_growth_v1.csv')
    result_df.to_csv(out_path, index=False)
    
    print(f"\n因子已保存: {out_path}")
    print(f"有效值: {result_df['factor'].notna().sum()}")
    print(f"均值: {result_df['factor'].mean():.4f}")
    print(f"标准差: {result_df['factor'].std():.4f}")
    print(f"偏度: {result_df['factor'].skew():.4f}")
    print(f"峰度: {result_df['factor'].kurtosis():.4f}")
    print(f"日期范围: {result_df['date'].min()} ~ {result_df['date'].max()}")
    
    return result_df

if __name__ == '__main__':
    compute_factor()
