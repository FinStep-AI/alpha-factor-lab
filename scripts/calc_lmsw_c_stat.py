#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LMSW C-statistic Factor (Llorente, Michaely, Saar & Wang, 2002 RFS)

核心思路：
  对每只股票在滚动窗口(60d)内做回归：
    ret_{t+1} = a + b * ret_t + c * ret_t * detrended_vol_t + e
  
  c > 0 → 知情交易主导（volume增强动量延续）→ 信息效率高
  c < 0 → 流动性/噪声交易主导（volume增强反转）→ 信息效率低
  
  我们预期 c 值高的股票后续表现更好（信息质量高 → Quality代理）

论文: Llorente, G., Michaely, R., Saar, G., & Wang, J. (2002). 
       "Dynamic Volume-Return Relation of Individual Stocks."
       Review of Financial Studies, 15(4), 1005-1047.
       https://doi.org/10.1093/rfs/15.4.1005

构造步骤:
  1. 计算日收益率 ret_t
  2. 计算 detrended volume: log(vol_t) - MA(log(vol_t), 20d)
  3. 60日滚动OLS: ret_{t+1} ~ ret_t + ret_t * detrended_vol_t
  4. 提取系数 c (交互项)
  5. 成交额OLS中性化 + MAD winsorize + z-score
"""

import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

def calc_lmsw_c_stat(kline_path, output_path, window=60, vol_detrend_window=20):
    """计算LMSW C统计量因子"""
    
    print(f"[1/5] Loading data from {kline_path}...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # 确保数值列
    for col in ['close', 'volume', 'amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"  Loaded {len(df)} rows, {df['stock_code'].nunique()} stocks")
    
    print(f"[2/5] Computing returns and detrended volume...")
    # 日收益率
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    
    # log volume
    df['log_vol'] = np.log(df['volume'].clip(lower=1))
    
    # detrended volume: log(vol) - MA(log(vol), 20d)
    df['log_vol_ma'] = df.groupby('stock_code')['log_vol'].transform(
        lambda x: x.rolling(vol_detrend_window, min_periods=10).mean()
    )
    df['detrended_vol'] = df['log_vol'] - df['log_vol_ma']
    
    # 次日收益 (forward return)
    df['ret_next'] = df.groupby('stock_code')['ret'].shift(-1)
    
    # 交互项: ret_t * detrended_vol_t
    df['ret_x_vol'] = df['ret'] * df['detrended_vol']
    
    print(f"[3/5] Rolling OLS regression (window={window})...")
    
    # 用向量化滚动方式计算c统计量
    # ret_{t+1} = a + b*ret_t + c*ret_t*detrended_vol_t + e
    # 只需要提取c系数
    
    # 为了效率，用numpy的矩阵运算代替逐个OLS
    results = []
    
    for stock, grp in df.groupby('stock_code'):
        grp = grp.sort_values('date').reset_index(drop=True)
        n = len(grp)
        
        if n < window + 5:
            continue
        
        y = grp['ret_next'].values
        x1 = grp['ret'].values         # ret_t
        x2 = grp['ret_x_vol'].values   # ret_t * detrended_vol_t
        
        # 滚动OLS提取c系数
        c_values = np.full(n, np.nan)
        
        for i in range(window - 1, n):
            idx_start = i - window + 1
            idx_end = i + 1
            
            yi = y[idx_start:idx_end]
            x1i = x1[idx_start:idx_end]
            x2i = x2[idx_start:idx_end]
            
            # 过滤nan
            mask = ~(np.isnan(yi) | np.isnan(x1i) | np.isnan(x2i))
            if mask.sum() < 20:  # 至少20个有效观测
                continue
            
            yi_clean = yi[mask]
            x1i_clean = x1i[mask]
            x2i_clean = x2i[mask]
            
            # OLS: Y = a + b*X1 + c*X2
            X = np.column_stack([np.ones(mask.sum()), x1i_clean, x2i_clean])
            
            try:
                beta = np.linalg.lstsq(X, yi_clean, rcond=None)[0]
                c_values[i] = beta[2]  # c系数
            except:
                continue
        
        dates = grp['date'].values
        for i in range(n):
            if not np.isnan(c_values[i]):
                results.append({
                    'date': dates[i],
                    'stock_code': stock,
                    'factor_raw': c_values[i]
                })
    
    print(f"  Computed {len(results)} factor values")
    
    factor_df = pd.DataFrame(results)
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    
    print(f"[4/5] Neutralizing and standardizing...")
    
    # 合并成交额用于中性化
    amt_df = df[['date', 'stock_code', 'amount']].copy()
    amt_df['log_amount_20d'] = amt_df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.clip(lower=1).rolling(20, min_periods=10).mean())
    )
    
    factor_df = factor_df.merge(
        amt_df[['date', 'stock_code', 'log_amount_20d']],
        on=['date', 'stock_code'], how='left'
    )
    
    # 截面处理：每日做中性化 + winsorize + z-score
    def cross_section_process(group):
        vals = group['factor_raw'].values.copy()
        log_amt = group['log_amount_20d'].values.copy()
        
        mask = ~(np.isnan(vals) | np.isnan(log_amt))
        if mask.sum() < 30:
            group['factor'] = np.nan
            return group
        
        # MAD winsorize
        med = np.nanmedian(vals[mask])
        mad = np.nanmedian(np.abs(vals[mask] - med))
        if mad > 0:
            lower = med - 5 * 1.4826 * mad
            upper = med + 5 * 1.4826 * mad
            vals = np.clip(vals, lower, upper)
        
        # OLS中性化 (对log_amount_20d)
        y = vals[mask]
        X = np.column_stack([np.ones(mask.sum()), log_amt[mask]])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = np.full(len(vals), np.nan)
            residuals[mask] = y - X @ beta
        except:
            residuals = vals.copy()
        
        # z-score
        mu = np.nanmean(residuals)
        std = np.nanstd(residuals)
        if std > 1e-10:
            residuals = (residuals - mu) / std
        
        group['factor'] = residuals
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(cross_section_process)
    
    # 输出
    output = factor_df[['date', 'stock_code', 'factor']].dropna()
    output = output.sort_values(['date', 'stock_code'])
    
    print(f"[5/5] Saving to {output_path}...")
    print(f"  Final: {len(output)} rows, {output['stock_code'].nunique()} stocks, "
          f"{output['date'].nunique()} dates")
    print(f"  Date range: {output['date'].min()} ~ {output['date'].max()}")
    
    output.to_csv(output_path, index=False)
    print("Done!")
    
    return output

if __name__ == '__main__':
    kline_path = sys.argv[1] if len(sys.argv) > 1 else 'data/csi1000_kline_raw.csv'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'data/factor_lmsw_c_stat_v1.csv'
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    
    calc_lmsw_c_stat(kline_path, output_path, window=window)
