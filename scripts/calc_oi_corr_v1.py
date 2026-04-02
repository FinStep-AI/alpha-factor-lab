#!/usr/bin/env python3
"""
因子: 隔夜-日内一致性 (Overnight-Intraday Consistency) v1
公式: rolling_corr(overnight_ret, intraday_ret, 20d)
      成交额OLS中性化 + MAD winsorize + z-score

逻辑:
- overnight_ret = open_t / close_{t-1} - 1 (集合竞价信息)
- intraday_ret = close_t / open_t - 1 (日内交易信息)
- 正相关 = 隔夜和日内方向一致 = 信息确认 = 趋势延续
- 负相关 = 隔夜和日内方向相反 = 日内反转 = 分歧

与已有因子的区别:
- overnight_momentum: 看隔夜收益水平(强弱)
- 本因子: 看隔夜和日内的协同关系(一致性)
- 完全不同的维度

Barra风格: Momentum/微观结构
"""

import numpy as np
import pandas as pd
import os

def compute_factor():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'csi1000_kline_raw.csv')
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    print(f"数据: {len(df)} 行, {df['stock_code'].nunique()} 股")
    
    g = df.groupby('stock_code')
    
    # Overnight return = open / prev_close - 1
    df['prev_close'] = g['close'].shift(1)
    df['overnight_ret'] = df['open'] / df['prev_close'] - 1
    
    # Intraday return = close / open - 1
    df['intraday_ret'] = df['close'] / df['open'] - 1
    
    # Rolling correlation between overnight and intraday returns
    for window in [20, 40, 60]:
        col = f'oi_corr_{window}'
        df[col] = g.apply(
            lambda x: x['overnight_ret'].rolling(window, min_periods=window//2).corr(x['intraday_ret'])
        ).reset_index(level=0, drop=True)
    
    # log_amount for neutralization
    df['log_amount_20d'] = g['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean().replace(0, np.nan))
    )
    df['log_amount_60d'] = g['amount'].transform(
        lambda x: np.log(x.rolling(60, min_periods=30).mean().replace(0, np.nan))
    )
    
    for window in [20, 40, 60]:
        col = f'oi_corr_{window}'
        neut_col = 'log_amount_20d' if window <= 20 else 'log_amount_60d'
        
        results = []
        for date, group in df.groupby('date'):
            sub = group[['stock_code', col, neut_col]].dropna()
            if len(sub) < 30:
                continue
            
            y = sub[col].values.copy()
            x = sub[neut_col].values
            
            med = np.median(y)
            mad_val = np.median(np.abs(y - med))
            if mad_val > 0:
                bound = 3 * 1.4826 * mad_val
                y = np.clip(y, med - bound, med + bound)
            
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
            
            med2 = np.median(resid)
            mad2 = np.median(np.abs(resid - med2))
            if mad2 > 0:
                bound2 = 3 * 1.4826 * mad2
                resid = np.clip(resid, med2 - bound2, med2 + bound2)
            
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
        
        out_path = os.path.join(os.path.dirname(__file__), '..', 'data', f'factor_oi_corr_{window}d_v1.csv')
        result_df.to_csv(out_path, index=False)
        print(f"Window {window}: saved {len(result_df)} rows, range {result_df['date'].min()} ~ {result_df['date'].max()}")

if __name__ == '__main__':
    compute_factor()
