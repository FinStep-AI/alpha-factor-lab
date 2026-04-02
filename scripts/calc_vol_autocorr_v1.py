#!/usr/bin/env python3
"""
因子: 成交量自相关 (Volume Autocorrelation) v1
公式: rolling_corr(vol_t, vol_{t-1}, 20d), 成交额OLS中性化

逻辑:
- 高自相关 = 成交量有惯性 = 系统性交易模式(可能是机构程序化交易)
- 低自相关 = 成交量随机 = 散户噪音交易
- 与vol_cv_neg(成交量波动性)不同: CV衡量离散度,自相关衡量时序持续性
- 与turnover_level不同: 换手水平是截面静态信息,自相关是时序动态信息
- ret_autocorr_v1测过(IC=0.0038,t=1.57,mono=0.9) - 收益率自相关IC弱但单调性好
- 成交量自相关是全新角度

假说: 成交量高度自相关的股票有更有序的交易模式,
      信息传播更系统化,价格发现效率更高→正alpha
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
    
    # Volume lag 1
    df['vol_lag1'] = g['volume'].shift(1)
    
    # Rolling 20-day correlation between volume and lagged volume
    def vol_autocorr(group):
        vol = group['volume']
        vol_lag = group['vol_lag1']
        return vol.rolling(20, min_periods=10).corr(vol_lag)
    
    df['raw_factor'] = g.apply(lambda x: vol_autocorr(x)).reset_index(level=0, drop=True)
    
    # log_amount_20d for neutralization
    df['log_amount_20d'] = g['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean().replace(0, np.nan))
    )
    
    print(f"原始因子有效值: {df['raw_factor'].notna().sum()}/{len(df)} ({df['raw_factor'].notna().mean()*100:.1f}%)")
    
    # Cross-sectional processing
    results = []
    for date, group in df.groupby('date'):
        sub = group[['stock_code', 'raw_factor', 'log_amount_20d']].dropna()
        if len(sub) < 30:
            continue
        
        y = sub['raw_factor'].values.copy()
        x = sub['log_amount_20d'].values
        
        # MAD winsorize
        med = np.median(y)
        mad_val = np.median(np.abs(y - med))
        if mad_val > 0:
            bound = 3 * 1.4826 * mad_val
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
        std_val = np.std(resid)
        if std_val > 0:
            zscore = (resid - np.mean(resid)) / std_val
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
    
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'factor_vol_autocorr_v1.csv')
    result_df.to_csv(out_path, index=False)
    
    print(f"\n因子已保存: {out_path}")
    print(f"有效值: {result_df['factor'].notna().sum()}")
    print(f"日期范围: {result_df['date'].min()} ~ {result_df['date'].max()}")

if __name__ == '__main__':
    compute_factor()
