#!/usr/bin/env python3
"""
CLV Trend Factor (Close Location Value Trend)
================================================
构造: 
  1. CLV = (2*close - high - low) / (high - low), 范围[-1, 1]
  2. 20日CLV的OLS线性回归斜率
  3. 正向使用: 斜率正 = 收盘位置逐日改善 = 买方力量递增

假说:
  CLV趋势向上 = 每天的收盘价越来越接近当日高点
  = 尾盘买方力量持续增强 = 可能是机构/主力逐步加仓信号
  这是一个微观结构的**趋势**信号，不是level

与已有因子差异:
  - shadow_pressure: 影线比率level (非趋势)
  - close_location_v1: CLV的level (非斜率)
  - pv_corr: 量价相关性 (不同维度)

中性化: 成交额OLS + MAD + z-score
"""

import numpy as np
import pandas as pd
import os

def compute_clv_trend(df_kline, window=20):
    """计算CLV趋势斜率"""
    df = df_kline.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date'])
    
    # CLV
    hl_range = df['high'] - df['low']
    df['clv'] = np.where(hl_range > 0, (2 * df['close'] - df['high'] - df['low']) / hl_range, 0)
    
    results = []
    # 线性回归的X（时间）向量，标准化
    t_vec = np.arange(window, dtype=float)
    t_vec = t_vec - t_vec.mean()
    t_dot = np.dot(t_vec, t_vec)
    
    for code, group in df.groupby('stock_code'):
        group = group.sort_values('date').reset_index(drop=True)
        n = len(group)
        clv = group['clv'].values
        amounts = group['amount'].values
        dates = group['date'].values
        
        for i in range(window - 1, n):
            w_clv = clv[i-window+1:i+1]
            
            if np.sum(np.isnan(w_clv)) > window * 0.3:
                continue
            
            # 填充NaN为0
            w_clv_clean = np.where(np.isnan(w_clv), 0, w_clv)
            
            # OLS斜率: slope = sum((t-t_mean)*(y-y_mean)) / sum((t-t_mean)^2)
            y_centered = w_clv_clean - np.mean(w_clv_clean)
            slope = np.dot(t_vec, y_centered) / t_dot if t_dot > 0 else 0
            
            # 成交额
            amt_w = amounts[max(0, i-19):i+1]
            log_amt = np.log(np.nanmean(amt_w) + 1)
            
            results.append({
                'date': dates[i],
                'stock_code': code,
                'clv_trend_raw': slope,
                'log_amount': log_amt
            })
    
    return pd.DataFrame(results)


def neutralize_zscore(df, fcol, ncol='log_amount'):
    outs = []
    for date, g in df.groupby('date'):
        g = g.dropna(subset=[fcol, ncol]).copy()
        if len(g) < 50: continue
        y = g[fcol].values
        x = np.column_stack([np.ones(len(g)), g[ncol].values])
        try:
            b = np.linalg.lstsq(x, y, rcond=None)[0]
            r = y - x @ b
        except: continue
        med = np.median(r)
        mad = np.median(np.abs(r - med))
        if mad < 1e-10: continue
        r = np.clip(r, med - 5*1.4826*mad, med + 5*1.4826*mad)
        s = np.std(r)
        if s < 1e-10: continue
        z = np.clip((r - np.mean(r)) / s, -3, 3)
        g = g.copy(); g['factor'] = z
        outs.append(g[['date', 'stock_code', 'factor']])
    return pd.concat(outs) if outs else pd.DataFrame()


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    df = pd.read_csv(os.path.join(data_dir, 'csi1000_kline_raw.csv'))
    print(f"Loaded {len(df)} rows, {df['stock_code'].nunique()} stocks")
    
    print("\nComputing CLV Trend (20d OLS slope)...")
    raw = compute_clv_trend(df, window=20)
    print(f"  Raw: {len(raw)}")
    
    fac = neutralize_zscore(raw, 'clv_trend_raw')
    fac.to_csv(os.path.join(data_dir, 'factor_clv_trend_v1.csv'), index=False)
    print(f"  Saved: {len(fac)} rows")
    
    # 翻转
    fac_neg = fac.copy()
    fac_neg['factor'] = -fac_neg['factor']
    fac_neg.to_csv(os.path.join(data_dir, 'factor_clv_trend_neg_v1.csv'), index=False)
    
    print(f"  Dates: {fac['date'].min()} ~ {fac['date'].max()}")
    print(f"  Stocks/date: {fac.groupby('date')['stock_code'].count().mean():.0f}")

if __name__ == '__main__':
    main()
