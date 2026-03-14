#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
残差动量因子 (Residual Momentum) v2 — Blitz, Huij & Martens (2011) JEF 复现

修正版：使用分离的估计窗口和评估窗口，避免OLS残差和为零问题

构造方法（按原论文）：
  1. 估计窗口(60日): 对个股收益 vs 市场收益做OLS, 得到 alpha_hat, beta_hat
  2. 评估窗口(20日): 用估计的模型计算残差 = r_i - (alpha_hat + beta_hat * r_mkt)
  3. 累计评估窗口的残差 → 残差动量信号
  
这样评估窗口的残差和不为零，因为模型参数来自不同窗口
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

def calc_residual_momentum(kline_path, output_path, est_window=60, eval_window=20):
    """计算残差动量因子"""
    
    print(f"📥 读取数据: {kline_path}")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # 计算日收益率
    if 'pct_change' in df.columns:
        df['ret'] = df['pct_change'] / 100.0
    else:
        df['ret'] = df.groupby('stock_code')['close'].pct_change()
    
    # 计算等权市场收益率
    mkt_ret = df.groupby('date')['ret'].mean().reset_index()
    mkt_ret.columns = ['date', 'mkt_ret']
    df = df.merge(mkt_ret, on='date', how='left')
    
    print(f"📊 数据范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"📊 股票数量: {df['stock_code'].nunique()}")
    print(f"📊 日期数量: {df['date'].nunique()}")
    
    total_window = est_window + eval_window
    print(f"⚙️  估计窗口={est_window}d, 评估窗口={eval_window}d, 总需{total_window}d")
    
    results = []
    n_stocks = df['stock_code'].nunique()
    
    for idx, (code, grp) in enumerate(df.groupby('stock_code')):
        if idx % 100 == 0:
            print(f"  处理: {idx}/{n_stocks} ({code})...")
        
        grp = grp.sort_values('date').reset_index(drop=True)
        n = len(grp)
        
        ret_arr = grp['ret'].values
        mkt_arr = grp['mkt_ret'].values
        dates = grp['date'].values
        
        for i in range(total_window - 1, n):
            # 估计窗口: [i - total_window + 1, i - eval_window]
            est_start = i - total_window + 1
            est_end = i - eval_window + 1  # exclusive
            
            # 评估窗口: [i - eval_window + 1, i]
            eval_start = i - eval_window + 1
            eval_end = i + 1  # exclusive
            
            # 估计窗口数据
            y_est = ret_arr[est_start:est_end]
            x_est = mkt_arr[est_start:est_end]
            
            valid_est = ~(np.isnan(y_est) | np.isnan(x_est))
            if valid_est.sum() < 20:  # 至少20个有效估计观测
                continue
            
            # OLS估计
            y_e = y_est[valid_est]
            x_e = x_est[valid_est]
            X_e = np.column_stack([np.ones(len(x_e)), x_e])
            
            try:
                params = np.linalg.lstsq(X_e, y_e, rcond=None)[0]
            except:
                continue
            
            alpha_hat, beta_hat = params[0], params[1]
            
            # 评估窗口残差
            y_eval = ret_arr[eval_start:eval_end]
            x_eval = mkt_arr[eval_start:eval_end]
            
            valid_eval = ~(np.isnan(y_eval) | np.isnan(x_eval))
            if valid_eval.sum() < 10:
                continue
            
            # 计算out-of-sample残差
            resid = y_eval[valid_eval] - (alpha_hat + beta_hat * x_eval[valid_eval])
            cum_resid = resid.sum()
            
            results.append({
                'date': dates[i],
                'stock_code': code,
                'factor_raw': cum_resid
            })
    
    factor_df = pd.DataFrame(results)
    print(f"\n✅ 原始因子计算完成: {len(factor_df)} 行")
    
    # 检查原始因子分布
    print(f"  原始因子均值: {factor_df['factor_raw'].mean():.6f}")
    print(f"  原始因子标准差: {factor_df['factor_raw'].std():.6f}")
    print(f"  原始因子中位数: {factor_df['factor_raw'].median():.6f}")
    
    # 市值中性化: 用log(20日均成交额)做代理
    print("\n⚙️  成交额中性化...")
    
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    
    amt_df = df[['date', 'stock_code', 'log_amount_20d']].drop_duplicates()
    factor_df = factor_df.merge(amt_df, on=['date', 'stock_code'], how='left')
    
    def neutralize_cross_section(group):
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
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(neutralize_cross_section)
    
    # Winsorize (MAD 方法)
    print("⚙️  MAD Winsorize...")
    def winsorize_mad(group, n_mad=5):
        vals = group['factor'].values
        valid = ~np.isnan(vals)
        if valid.sum() < 10:
            return group
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals[valid] - med))
        if mad < 1e-10:
            return group
        lower = med - n_mad * 1.4826 * mad
        upper = med + n_mad * 1.4826 * mad
        group['factor'] = np.clip(vals, lower, upper)
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_mad)
    
    # Z-score
    print("⚙️  Z-score 标准化...")
    def zscore(group):
        vals = group['factor'].values
        valid = ~np.isnan(vals)
        if valid.sum() < 10:
            group['factor'] = np.nan
            return group
        mu = np.nanmean(vals)
        std = np.nanstd(vals)
        if std < 1e-10:
            group['factor'] = 0.0
        else:
            group['factor'] = (vals - mu) / std
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(zscore)
    
    # 输出
    output = factor_df[['date', 'stock_code', 'factor']].dropna()
    output = output.sort_values(['date', 'stock_code'])
    output.to_csv(output_path, index=False)
    
    print(f"\n📊 输出统计:")
    print(f"  行数: {len(output)}")
    print(f"  日期范围: {output['date'].min()} ~ {output['date'].max()}")
    print(f"  截面日期数: {output['date'].nunique()}")
    print(f"  因子均值: {output['factor'].mean():.6f}")
    print(f"  因子标准差: {output['factor'].std():.4f}")
    print(f"\n✅ 因子已保存到: {output_path}")
    
    return output

if __name__ == '__main__':
    kline_path = sys.argv[1] if len(sys.argv) > 1 else 'data/csi1000_kline_raw.csv'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'data/factor_residual_momentum_v1.csv'
    est_window = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    eval_window = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    
    calc_residual_momentum(kline_path, output_path, est_window, eval_window)
