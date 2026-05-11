#!/usr/bin/env python3
"""计算 ma_disp_neg_v1 (反转均线离散度)"""
import numpy as np
import pandas as pd
from pathlib import Path

def calc_ma_disp_neg(df, window=20, ma_list=[5,10,20,40,60,120]):
    df = df.copy()
    for ma in ma_list:
        df[f'ma_{ma}'] = df.groupby('stock_code')['close'].transform(
            lambda x: x.rolling(ma, min_periods=5).mean()
        )
    
    ma_cols = [f'ma_{m}' for m in ma_list]
    all_ma = df[ma_cols].values
    
    # pairwise spread = mean |log(MA_i/MA_j)| among all pairs
    log_ma = np.log(all_ma + 1e-9)
    pair_spread = []
    for i in range(len(ma_list)):
        for j in range(i+1, len(ma_list)):
            pair_spread.append(np.abs(log_ma[:, i] - log_ma[:, j]))
    
    pair_spread = np.array(pair_spread).T
    ma_disp = -np.nanmean(pair_spread, axis=1)  # negate for reversal direction
    
    df['ma_disp_neg'] = ma_disp
    
    # neutralize
    df['amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    df['log_amount'] = np.log1p(df['amount_20d'])
    
    from numpy.linalg import lstsq
    result = []
    for dt, g in df.groupby('date'):
        sub = g[['ma_disp_neg', 'log_amount', 'stock_code']].dropna()
        if len(sub) < 30: continue
        y = sub['ma_disp_neg'].values
        x = sub['log_amount'].values
        X = np.column_stack([np.ones(len(x)), x])
        coeffs, _, _, _ = lstsq(X, y, rcond=None)
        resid = y - X @ coeffs
        med, mad = np.median(resid), np.median(np.abs(resid-np.median(resid)))*1.4826
        if mad < 1e-10: continue
        z = (resid - med) / mad
        z = np.clip(z, -5, 5)
        mu, sigma = np.mean(z), np.std(z)
        if sigma < 1e-10: continue
        z = (z - mu) / sigma
        for idx, val in zip(sub.index, z):
            result.append((dt, sub.loc[idx, 'stock_code'], val))
    
    res = pd.DataFrame(result, columns=['date', 'stock_code', 'factor_value'])
    res.to_csv('data/factor_ma_disp_neg_v1.csv', index=False)
    print("saved data/factor_ma_disp_neg_v1.csv")
    print(res['factor_value'].describe())

df = pd.read_csv('data/csi1000_kline_raw.csv', parse_dates=['date'])
df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
calc_ma_disp_neg(df)
