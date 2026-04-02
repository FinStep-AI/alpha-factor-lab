#!/usr/bin/env python3
"""
Bollinger Band Width (BBW) Factor
==================================
构造: BBW = (BB_upper - BB_lower) / BB_middle
     = 2 * k * std(close, N) / MA(close, N)
     其中 k=2, N=20

本质上是"相对波动率" = 2*CV(close, 20d)

假说:
  A) 高BBW = 高波动 → 波动率溢价 (与amp_level类似方向)
  B) 低BBW = 波动收窄 → 突破前蓄力 → 高收益 (经典技术分析)
  
  需要验证方向。amp_level_v2(60日振幅)已入库，BBW用的是收盘价标准差/均值，
  时间窗口不同，可能捕捉不同信息。

但BBW本质是变异系数，可能与振幅高度相关。让我换个思路：

改做 "布林带位置因子(BB Position / %B)"
  %B = (close - BB_lower) / (BB_upper - BB_lower)
  = (close - MA20 + 2*std20) / (4*std20)
  
  %B高 = 接近上轨 = 超买
  %B低 = 接近下轨 = 超卖
  
  这本质是一个标准化的均线偏离度，但用波动率做标准化而非固定比例。
  
  在截面上，%B可能比BIAS更有区分度，因为每只股票的波动率不同。

进一步改进：用20日 %B 的均值（平滑），而非单日值。

方向：
  做多低%B(接近下轨, 超卖) → 反转逻辑
  做多高%B(接近上轨, 强势) → 动量逻辑
  
  A股CSI1000小盘股之前RSI发现是动量而非反转，所以先试正向(高%B做多)。
"""

import numpy as np
import pandas as pd
import sys
import os

def compute_bb_position(df_kline, window=20, avg_window=5):
    """计算布林带位置因子 %B 的均值"""
    
    df = df_kline.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date'])
    
    results = []
    
    for code, group in df.groupby('stock_code'):
        group = group.sort_values('date').reset_index(drop=True)
        closes = group['close'].values
        amounts = group['amount'].values
        dates = group['date'].values
        
        n = len(group)
        pct_b_vals = np.full(n, np.nan)
        
        for i in range(window - 1, n):
            window_close = closes[i-window+1:i+1]
            if np.sum(np.isnan(window_close)) > 0:
                continue
            
            ma = np.mean(window_close)
            std = np.std(window_close, ddof=1)
            
            if std < 1e-10 or ma < 1e-10:
                continue
            
            bb_upper = ma + 2 * std
            bb_lower = ma - 2 * std
            bb_width = bb_upper - bb_lower
            
            if bb_width < 1e-10:
                continue
            
            pct_b = (closes[i] - bb_lower) / bb_width
            pct_b_vals[i] = pct_b
        
        # 取 avg_window 日均值平滑
        for i in range(window - 1 + avg_window - 1, n):
            vals = pct_b_vals[i-avg_window+1:i+1]
            if np.sum(np.isnan(vals)) > avg_window * 0.3:
                continue
            avg_pct_b = np.nanmean(vals)
            
            # 计算20日平均成交额用于中性化
            amt_window = amounts[max(0, i-19):i+1]
            log_amt = np.log(np.nanmean(amt_window) + 1)
            
            results.append({
                'date': dates[i],
                'stock_code': code,
                'bb_pos_raw': avg_pct_b,
                'log_amount': log_amt
            })
    
    return pd.DataFrame(results)


def neutralize_and_zscore(df, factor_col, neutral_col='log_amount'):
    """截面OLS中性化 + MAD winsorize + z-score"""
    
    all_results = []
    
    for date, group in df.groupby('date'):
        g = group.dropna(subset=[factor_col, neutral_col]).copy()
        if len(g) < 50:
            continue
        
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
    
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    print("Loading kline data...")
    df_kline = pd.read_csv(os.path.join(data_dir, 'csi1000_kline_raw.csv'))
    print(f"  Loaded {len(df_kline)} rows, {df_kline['stock_code'].nunique()} stocks")
    
    print("\nComputing BB Position (%B) factor, window=20, avg=5...")
    raw_df = compute_bb_position(df_kline, window=20, avg_window=5)
    print(f"  Raw computed: {len(raw_df)} rows")
    
    # 正向版本 (高%B=强势=做多)
    print("Neutralizing (positive direction: high %B = buy)...")
    factor_pos = neutralize_and_zscore(raw_df, 'bb_pos_raw')
    output_pos = os.path.join(data_dir, 'factor_bb_pos_v1.csv')
    factor_pos.to_csv(output_pos, index=False)
    print(f"  Saved positive version: {output_pos} ({len(factor_pos)} rows)")
    
    # 翻转版本 (低%B=超卖=做多)
    factor_neg = factor_pos.copy()
    factor_neg['factor'] = -factor_neg['factor']
    output_neg = os.path.join(data_dir, 'factor_bb_pos_neg_v1.csv')
    factor_neg.to_csv(output_neg, index=False)
    print(f"  Saved negative version: {output_neg}")
    
    print(f"\n  Date range: {factor_pos['date'].min()} ~ {factor_pos['date'].max()}")
    print(f"  Stocks/date (mean): {factor_pos.groupby('date')['stock_code'].count().mean():.0f}")

if __name__ == '__main__':
    main()
