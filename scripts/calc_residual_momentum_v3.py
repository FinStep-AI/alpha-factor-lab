#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
残差动量因子 v3 — 改进版本，使用向量化计算加速

尝试多种参数组合:
1. 估计60d + 评估10d (更短期的特质动量)
2. 估计120d + 评估20d (更稳定的beta估计)
3. 残差动量取 t-统计量而非简单累加 (标准化处理)
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

def calc_residual_momentum_fast(kline_path, output_path, est_window=60, eval_window=20, use_tstat=False):
    """快速计算残差动量因子 - 向量化"""
    
    print(f"📥 读取数据: {kline_path}")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    if 'pct_change' in df.columns:
        df['ret'] = df['pct_change'] / 100.0
    else:
        df['ret'] = df.groupby('stock_code')['close'].pct_change()
    
    mkt_ret = df.groupby('date')['ret'].mean().reset_index()
    mkt_ret.columns = ['date', 'mkt_ret']
    df = df.merge(mkt_ret, on='date', how='left')
    
    print(f"📊 数据: {df['date'].min()} ~ {df['date'].max()}, {df['stock_code'].nunique()} stocks, {df['date'].nunique()} dates")
    
    total_window = est_window + eval_window
    print(f"⚙️  est={est_window}d, eval={eval_window}d, tstat={use_tstat}")
    
    results = []
    n_stocks = df['stock_code'].nunique()
    
    for idx, (code, grp) in enumerate(df.groupby('stock_code')):
        if idx % 200 == 0:
            print(f"  {idx}/{n_stocks}...")
        
        grp = grp.sort_values('date').reset_index(drop=True)
        n = len(grp)
        ret_arr = grp['ret'].values.astype(float)
        mkt_arr = grp['mkt_ret'].values.astype(float)
        dates = grp['date'].values
        
        for i in range(total_window - 1, n):
            est_s = i - total_window + 1
            est_e = i - eval_window + 1
            eval_s = i - eval_window + 1
            eval_e = i + 1
            
            y_est = ret_arr[est_s:est_e]
            x_est = mkt_arr[est_s:est_e]
            valid_est = ~(np.isnan(y_est) | np.isnan(x_est))
            if valid_est.sum() < max(20, est_window // 3):
                continue
            
            y_e, x_e = y_est[valid_est], x_est[valid_est]
            X_e = np.column_stack([np.ones(len(x_e)), x_e])
            
            try:
                params = np.linalg.lstsq(X_e, y_e, rcond=None)[0]
            except:
                continue
            
            alpha_hat, beta_hat = params
            
            y_eval = ret_arr[eval_s:eval_e]
            x_eval = mkt_arr[eval_s:eval_e]
            valid_eval = ~(np.isnan(y_eval) | np.isnan(x_eval))
            if valid_eval.sum() < max(5, eval_window // 3):
                continue
            
            resid = y_eval[valid_eval] - (alpha_hat + beta_hat * x_eval[valid_eval])
            
            if use_tstat:
                # t-stat of residual mean
                n_r = len(resid)
                mean_r = resid.mean()
                std_r = resid.std()
                if std_r < 1e-10 or n_r < 3:
                    continue
                factor_val = mean_r / (std_r / np.sqrt(n_r))
            else:
                factor_val = resid.sum()
            
            results.append({
                'date': dates[i],
                'stock_code': code,
                'factor_raw': factor_val
            })
    
    factor_df = pd.DataFrame(results)
    print(f"✅ 原始: {len(factor_df)} rows, mean={factor_df['factor_raw'].mean():.6f}, std={factor_df['factor_raw'].std():.6f}")
    
    # 成交额中性化
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    amt_df = df[['date', 'stock_code', 'log_amount_20d']].drop_duplicates()
    factor_df = factor_df.merge(amt_df, on=['date', 'stock_code'], how='left')
    
    def neutralize(group):
        y = group['factor_raw'].values
        x = group['log_amount_20d'].values
        valid = ~(np.isnan(y) | np.isnan(x))
        if valid.sum() < 30:
            group['factor'] = np.nan
            return group
        X = np.column_stack([np.ones(valid.sum()), x[valid]])
        try:
            beta = np.linalg.lstsq(X, y[valid], rcond=None)[0]
            resid = np.full(len(y), np.nan)
            resid[valid] = y[valid] - X @ beta
            group['factor'] = resid
        except:
            group['factor'] = np.nan
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(neutralize)
    
    # MAD Winsorize + Z-score
    def winsorize_zscore(group):
        vals = group['factor'].values.copy()
        valid = ~np.isnan(vals)
        if valid.sum() < 10:
            group['factor'] = np.nan
            return group
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals[valid] - med))
        if mad > 1e-10:
            lower = med - 5 * 1.4826 * mad
            upper = med + 5 * 1.4826 * mad
            vals = np.clip(vals, lower, upper)
        mu = np.nanmean(vals)
        std = np.nanstd(vals)
        if std < 1e-10:
            group['factor'] = 0.0
        else:
            group['factor'] = (vals - mu) / std
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_zscore)
    
    output = factor_df[['date', 'stock_code', 'factor']].dropna()
    output = output.sort_values(['date', 'stock_code'])
    output.to_csv(output_path, index=False)
    
    print(f"✅ 保存: {output_path}, {len(output)} rows, {output['date'].nunique()} dates")
    return output

if __name__ == '__main__':
    kline = sys.argv[1] if len(sys.argv) > 1 else 'data/csi1000_kline_raw.csv'
    out = sys.argv[2] if len(sys.argv) > 2 else 'data/factor_residual_momentum_v1.csv'
    est = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    ev = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    tstat = '--tstat' in sys.argv
    
    calc_residual_momentum_fast(kline, out, est, ev, tstat)
