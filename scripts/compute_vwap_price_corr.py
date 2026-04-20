#!/usr/bin/env python3
"""
因子: VWAP与期末价的20日相关性 (VWAP-Price Correlation v1)

逻辑:
  Brennan, Pasquariello & Subrahmanyam (2006) show that order imbalances
  around market close predict subsequent price movements.
  
  VWAP = volume-weighted average price = 全天交易加权均价
  close-VWAP = 收盘价相对全天均价的位置
  
  计算20日滚动 corr(close-VWAP, amount_chg) 或 corr(close-VWAP, pct_change)
  
  这里用 corr((close-VWAP)/range, daily_ret, 20d)
  衡量"尾盘偏离方向"与"次日收益"的持续性。
  
  如果高corr为正：close-VWAP为正(尾盘推升)的日子，次日继续涨(动量延续)
  如果高corr为负：close-VWAP为正的日子，次日回调(反转)
  
  构造: corr(CLV, daily_ret, 20d) 其中CLV=(close-low)/(high-low)近似但用了振幅
  更直接的是直接用 corr((close-VWAP), daily_ret, 20d)

参考文献: 
  -疲软因子365 (东方证券/华泰证券量价系列)
  -Brennan, Chordia & Subrahmanyam (1998, 2004)

Barra风格: MICRO
"""

import numpy as np
import pandas as pd
import sys

def compute_factor(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    grouped = df.groupby('stock_code')
    
    # 日收益率
    df['ret'] = grouped['close'].pct_change()
    
    # VWAP: (close * volume).rolling() / volume.rolling() ≈ 用amount/volume精确
    # 简化: VWAP = sum(amount) / sum(volume) 用于20日均值
    df['vwap_20'] = (grouped['amount'].transform(lambda x: x.rolling(20,min_periods=10).mean()) /
                     grouped['volume'].transform(lambda x: x.rolling(20,min_periods=10).mean()))
    
    # daily close - VWAP 偏离
    df['close_vwap_dev'] = df['close'] - df['vwap_20']
    
    # cntr- VWAP偏离率除以振幅标准化
    df['amp_20d'] = grouped['amplitude'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    eps = 0.001
    df['norm_cvd'] = df['close_vwap_dev'] / (df['amp_20d'] + eps)
    
    # 20日 rolling corr(norm_cvd, ret)
    print("[INFO] 计算20日滚动相关系数...")
    def rolling_corr(group):
        return group['norm_cvd'].rolling(20, min_periods=10).corr(group['ret'])
    
    from functools import partial
    df['corr_raw'] = grouped.apply(rolling_corr).reset_index(level=0, drop=True)
    
    # 换手率中性化
    df['log_amount_20d'] = np.log(df['amount'].rolling(20, min_periods=10).mean() + 1)
    
    results = []
    for date, group in df.groupby('date'):
        g = group.dropna(subset=['corr_raw', 'log_amount_20d'])
        if len(g) < 30:
            continue
        
        x = g['log_amount_20d'].values
        y = g['corr_raw'].values
        
        X = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residual = y - X @ beta
        except:
            continue
        
        med = np.median(residual)
        mad = np.median(np.abs(residual - med)) * 1.4826
        if mad < 1e-10:
            continue
        residual = np.clip(residual, med - 3*mad, med + 3*mad)
        mu, sigma = residual.mean(), residual.std()
        if sigma < 1e-10:
            continue
        z = (residual - mu) / sigma
        
        g = g.copy()
        g['factor_neutral'] = z
        results.append(g[['date', 'stock_code', 'factor_neutral']])
    
    if not results:
        print("ERROR: no data", file=sys.stderr)
        sys.exit(1)
    
    result_df = pd.concat(results, ignore_index=True)
    print(f"[INFO] VWAP-Price Corr 因子: {result_df['date'].min()} ~ {result_df['date'].max()}, "
          f"{result_df['stock_code'].nunique()} stocks")
    print(f"[INFO] stats: mean={result_df['factor_neutral'].mean():.4f}, std={result_df['factor_neutral'].std():.4f}")
    
    out_path = 'data/factor_vwap_price_corr_v1.csv'
    result_df.to_csv(out_path, index=False)
    print(f"[INFO] 保存 {out_path}")
    return out_path

if __name__ == '__main__':
    compute_factor(pd.read_csv('data/csi1000_kline_raw.csv'))
