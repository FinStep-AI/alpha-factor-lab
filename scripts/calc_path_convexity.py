#!/usr/bin/env python3
"""
Price-Path Convexity (PPC) Factor
=================================
论文启发: "Price-Path Convexity and Short-Horizon Return Predictability"

构造: 20日窗口内，给每日收益率赋予时间权重 w_t = (2t/T - 1)，
      PPC = sum(w_t * r_t)
      - 正PPC = 先跌后涨（V型路径，凸）
      - 负PPC = 先涨后跌（倒V型路径，凹）

论文发现: 高PPC(凸) → 低未来收益 (反转/均值回复)
         即负向使用: factor = -PPC

假说:
  1. 凸路径(先跌后涨)是短期反弹，动能不可持续
  2. 凹路径(先涨后跌)的股票超卖后有反转机会
  3. 在A股中证1000中，需要验证方向

中性化: 成交额OLS中性化 + MAD winsorize + z-score
"""

import numpy as np
import pandas as pd
import sys
import os

def compute_path_convexity(df_kline, window=20):
    """计算价格路径凸性因子"""
    
    df = df_kline.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date'])
    
    # 计算日收益率
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    
    # 时间权重: w_t = (2*t/T - 1), t从1到T
    # t=1时 w=-0.9, t=T时 w=0.9（近似）
    # 这使得早期收益权重为负，晚期为正
    T = window
    weights = np.array([(2 * t / T - 1) for t in range(1, T + 1)])
    # 归一化使得 sum(|w|) = 1
    weights = weights / np.sum(np.abs(weights))
    
    # 滚动计算加权收益
    results = []
    for code, group in df.groupby('stock_code'):
        group = group.sort_values('date').reset_index(drop=True)
        rets = group['ret'].values
        dates = group['date'].values
        amounts = group['amount'].values
        
        ppc_vals = np.full(len(group), np.nan)
        
        for i in range(window, len(group)):
            window_rets = rets[i-window+1:i+1]
            if np.sum(np.isnan(window_rets)) > window * 0.3:
                continue
            # 用0填充NaN
            window_rets_clean = np.where(np.isnan(window_rets), 0, window_rets)
            ppc_vals[i] = np.dot(weights, window_rets_clean)
        
        for i in range(len(group)):
            if not np.isnan(ppc_vals[i]):
                results.append({
                    'date': dates[i],
                    'stock_code': code,
                    'ppc_raw': ppc_vals[i],
                    'log_amount': np.log(np.nanmean(amounts[max(0,i-window+1):i+1]) + 1)
                })
    
    result_df = pd.DataFrame(results)
    return result_df

def neutralize_and_zscore(df, factor_col='ppc_raw', neutral_col='log_amount'):
    """截面中性化 + MAD winsorize + z-score"""
    from scipy import stats
    
    all_results = []
    
    for date, group in df.groupby('date'):
        g = group.dropna(subset=[factor_col, neutral_col]).copy()
        if len(g) < 50:
            continue
        
        # OLS 中性化
        y = g[factor_col].values
        x = g[neutral_col].values
        x_with_const = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
            residual = y - x_with_const @ beta
        except:
            continue
        
        # MAD winsorize
        med = np.median(residual)
        mad = np.median(np.abs(residual - med))
        if mad < 1e-10:
            continue
        upper = med + 5 * 1.4826 * mad
        lower = med - 5 * 1.4826 * mad
        residual = np.clip(residual, lower, upper)
        
        # z-score
        std = np.std(residual)
        if std < 1e-10:
            continue
        z = (residual - np.mean(residual)) / std
        z = np.clip(z, -3, 3)
        
        g = g.copy()
        g['factor'] = z
        all_results.append(g[['date', 'stock_code', 'factor']])
    
    if not all_results:
        return pd.DataFrame()
    
    return pd.concat(all_results, ignore_index=True)

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    print("Loading kline data...")
    df_kline = pd.read_csv(os.path.join(data_dir, 'csi1000_kline_raw.csv'))
    print(f"  Loaded {len(df_kline)} rows, {df_kline['stock_code'].nunique()} stocks")
    print(f"  Date range: {df_kline['date'].min()} ~ {df_kline['date'].max()}")
    
    # === 正向 PPC (高凸性 = 先跌后涨) ===
    print("\nComputing Path Convexity (PPC) factor, window=20...")
    ppc_df = compute_path_convexity(df_kline, window=20)
    print(f"  Raw PPC computed: {len(ppc_df)} rows")
    
    # 中性化
    print("Neutralizing...")
    factor_df = neutralize_and_zscore(ppc_df, 'ppc_raw', 'log_amount')
    print(f"  Factor output: {len(factor_df)} rows")
    
    if len(factor_df) == 0:
        print("ERROR: No factor values computed!")
        sys.exit(1)
    
    # 保存正向版本（原始PPC，高凸性=高因子值）
    output_path = os.path.join(data_dir, 'factor_ppc_v1.csv')
    factor_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"  Dates: {factor_df['date'].min()} ~ {factor_df['date'].max()}")
    print(f"  Stocks per date (mean): {factor_df.groupby('date')['stock_code'].count().mean():.0f}")
    
    # 保存翻转版本（负PPC，高凹性=高因子值）
    factor_df_neg = factor_df.copy()
    factor_df_neg['factor'] = -factor_df_neg['factor']
    output_path_neg = os.path.join(data_dir, 'factor_ppc_neg_v1.csv')
    factor_df_neg.to_csv(output_path_neg, index=False)
    print(f"Saved flipped version to {output_path_neg}")

if __name__ == '__main__':
    main()
