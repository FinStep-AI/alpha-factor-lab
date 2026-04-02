#!/usr/bin/env python3
"""
High-Volume Day Return (HVR) Factor
====================================
构造: 过去20日中成交额排名前5天的平均收益率
     正向使用: 高量日均在涨 → 知情交易者看多 → 后续延续

改进: 不用固定阈值(之前highvol_ret_dir_v1用1.5倍中位数),
      而是rank-based top-5, 更稳定

假说: 大成交额日集中了知情交易/机构交易, 
      这些日子的收益方向是市场"真正"的方向信号。

进一步: 也试试"低量日收益"的相反逻辑
"""

import numpy as np
import pandas as pd
import os
import sys

def compute_hvr(df_kline, window=20, top_k=5):
    """计算高成交额日收益因子"""
    
    df = df_kline.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date'])
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    
    results = []
    
    for code, group in df.groupby('stock_code'):
        group = group.sort_values('date').reset_index(drop=True)
        n = len(group)
        rets = group['ret'].values
        amounts = group['amount'].values
        dates = group['date'].values
        
        for i in range(window, n):
            w_rets = rets[i-window+1:i+1]
            w_amts = amounts[i-window+1:i+1]
            
            # 跳过有太多NaN的窗口
            valid = ~(np.isnan(w_rets) | np.isnan(w_amts))
            if np.sum(valid) < window * 0.7:
                continue
            
            # 在有效日中找成交额最大的top_k天
            valid_idx = np.where(valid)[0]
            valid_amts = w_amts[valid_idx]
            valid_rets = w_rets[valid_idx]
            
            if len(valid_idx) < top_k:
                continue
            
            # 排序找top-K
            top_indices = np.argsort(valid_amts)[-top_k:]
            hvr = np.mean(valid_rets[top_indices])
            
            # 20日平均成交额
            avg_amt = np.nanmean(w_amts[valid])
            
            results.append({
                'date': dates[i],
                'stock_code': code,
                'hvr_raw': hvr,
                'log_amount': np.log(avg_amt + 1)
            })
    
    return pd.DataFrame(results)


def neutralize_and_zscore(df, factor_col, neutral_col='log_amount'):
    """截面OLS中性化 + MAD + z-score"""
    all_results = []
    
    for date, group in df.groupby('date'):
        g = group.dropna(subset=[factor_col, neutral_col]).copy()
        if len(g) < 50:
            continue
        
        y = g[factor_col].values
        x = g[neutral_col].values
        x_const = np.column_stack([np.ones(len(x)), x])
        
        try:
            beta = np.linalg.lstsq(x_const, y, rcond=None)[0]
            residual = y - x_const @ beta
        except:
            continue
        
        med = np.median(residual)
        mad = np.median(np.abs(residual - med))
        if mad < 1e-10:
            continue
        residual = np.clip(residual, med - 5*1.4826*mad, med + 5*1.4826*mad)
        
        std = np.std(residual)
        if std < 1e-10:
            continue
        z = (residual - np.mean(residual)) / std
        z = np.clip(z, -3, 3)
        
        g = g.copy()
        g['factor'] = z
        all_results.append(g[['date', 'stock_code', 'factor']])
    
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    print("Loading kline data...")
    df = pd.read_csv(os.path.join(data_dir, 'csi1000_kline_raw.csv'))
    print(f"  {len(df)} rows, {df['stock_code'].nunique()} stocks")
    
    print("\nComputing High-Volume-Day Return factor (top-5 of 20d)...")
    raw = compute_hvr(df, window=20, top_k=5)
    print(f"  Raw: {len(raw)} rows")
    
    # 正向 (高量日收益高 = 做多)
    factor = neutralize_and_zscore(raw, 'hvr_raw')
    out = os.path.join(data_dir, 'factor_hvr_v1.csv')
    factor.to_csv(out, index=False)
    print(f"  Saved: {out} ({len(factor)} rows)")
    
    # 翻转版本
    factor_neg = factor.copy()
    factor_neg['factor'] = -factor_neg['factor']
    out_neg = os.path.join(data_dir, 'factor_hvr_neg_v1.csv')
    factor_neg.to_csv(out_neg, index=False)
    print(f"  Saved flipped: {out_neg}")
    
    print(f"\n  Date range: {factor['date'].min()} ~ {factor['date'].max()}")
    print(f"  Stocks/date: {factor.groupby('date')['stock_code'].count().mean():.0f}")

if __name__ == '__main__':
    main()
