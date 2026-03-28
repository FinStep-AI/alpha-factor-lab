#!/usr/bin/env python3
"""
Corwin & Schultz (2012) High-Low Spread Estimator
"A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices"
Journal of Finance, 67(2), 719-760.

核心思想：
- 日线最高价几乎总是买入交易，最低价几乎总是卖出交易
- 因此 high-low range 包含两个成分：(1) 真实波动率 (2) 买卖价差
- 利用2日高低价 vs 单日高低价的关系可分离这两个成分
- 波动率随sqrt(时间)增长，而价差是常数

因子假设：
- 高买卖价差 = 流动性差 = 交易成本高 = 风险溢价
- 正向：高spread → 高预期收益（流动性溢价）

构造步骤：
1. 计算每日的 Corwin-Schultz spread 估计值
2. 取20日滚动均值（降噪）
3. 对数变换 + 成交额OLS中性化 + MAD winsorize + z-score
"""

import numpy as np
import pandas as pd
import sys
import os

def calc_corwin_schultz_spread(df):
    """
    计算 Corwin-Schultz (2012) 买卖价差估计值
    
    输入: DataFrame with columns: date, stock_code, high, low, close, amount
    输出: DataFrame with columns: date, stock_code, factor_value
    """
    df = df.sort_values(['stock_code', 'date']).copy()
    
    results = []
    
    for code, g in df.groupby('stock_code'):
        g = g.sort_values('date').reset_index(drop=True)
        
        high = g['high'].values
        low = g['low'].values
        n = len(g)
        
        # 单日 beta: [ln(H_t / L_t)]^2
        ln_hl = np.log(high / low)
        beta_single = ln_hl ** 2  # shape (n,)
        
        # 两日 gamma: [ln(H_{t,t+1} / L_{t,t+1})]^2
        # H_{t,t+1} = max(H_t, H_{t+1}), L_{t,t+1} = min(L_t, L_{t+1})
        high_2d = np.maximum(high[:-1], high[1:])
        low_2d = np.minimum(low[:-1], low[1:])
        gamma = (np.log(high_2d / low_2d)) ** 2  # shape (n-1,)
        
        # beta: sum of two consecutive single-day beta values
        beta = beta_single[:-1] + beta_single[1:]  # shape (n-1,)
        
        # alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
        k = 3 - 2 * np.sqrt(2)  # ≈ 0.1716
        
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        
        # Spread = 2 * (exp(alpha) - 1) / (1 + exp(alpha))
        # alpha < 0 时 spread 设为 0 (原文建议)
        alpha_clipped = np.maximum(alpha, 0)
        spread = 2 * (np.exp(alpha_clipped) - 1) / (1 + np.exp(alpha_clipped))
        
        # 对齐日期 (spread[i] 对应 date[i+1]，因为用了 t 和 t+1 的数据)
        dates = g['date'].values[1:]
        amounts = g['amount'].values[1:]
        
        for i in range(len(spread)):
            results.append({
                'date': dates[i],
                'stock_code': code,
                'cs_spread_raw': spread[i],
                'amount': amounts[i]
            })
    
    result_df = pd.DataFrame(results)
    return result_df


def build_factor(kline_path, output_path, window=20):
    """
    构建 Corwin-Schultz Spread 因子
    
    1. 计算每日CS spread
    2. 取 {window}日滚动均值
    3. log变换（spread本身就是正数，加微小常数避免log(0)）
    4. 成交额OLS中性化
    5. MAD winsorize + z-score
    """
    print(f"读取K线数据: {kline_path}")
    df = pd.read_csv(kline_path, parse_dates=['date'])
    
    # 过滤异常数据
    df = df[df['high'] > 0].copy()
    df = df[df['low'] > 0].copy()
    df = df[df['high'] >= df['low']].copy()
    df = df[df['amount'] > 0].copy()
    
    print(f"数据范围: {df['date'].min()} ~ {df['date'].max()}, {df['stock_code'].nunique()} 只股票")
    
    # Step 1: 计算每日 CS spread
    print("Step 1: 计算 Corwin-Schultz spread...")
    spread_df = calc_corwin_schultz_spread(df[['date', 'stock_code', 'high', 'low', 'close', 'amount']])
    
    print(f"  原始 spread 统计: mean={spread_df['cs_spread_raw'].mean():.6f}, "
          f"median={spread_df['cs_spread_raw'].median():.6f}, "
          f"std={spread_df['cs_spread_raw'].std():.6f}, "
          f"zero_pct={100*(spread_df['cs_spread_raw']==0).mean():.1f}%")
    
    # Step 2: 滚动均值
    print(f"Step 2: {window}日滚动均值...")
    spread_df = spread_df.sort_values(['stock_code', 'date'])
    spread_df['cs_spread_ma'] = spread_df.groupby('stock_code')['cs_spread_raw'].transform(
        lambda x: x.rolling(window, min_periods=max(window//2, 10)).mean()
    )
    
    # 也计算滚动均成交额 (用于中性化)
    spread_df['log_amount_ma'] = spread_df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(window, min_periods=max(window//2, 10)).mean() + 1)
    )
    
    spread_df = spread_df.dropna(subset=['cs_spread_ma', 'log_amount_ma'])
    
    print(f"  MA{window} spread 统计: mean={spread_df['cs_spread_ma'].mean():.6f}, "
          f"median={spread_df['cs_spread_ma'].median():.6f}")
    
    # Step 3: log变换
    print("Step 3: log变换...")
    eps = 1e-8
    spread_df['cs_spread_log'] = np.log(spread_df['cs_spread_ma'] + eps)
    
    # Step 4: 截面成交额OLS中性化 + MAD winsorize + z-score
    print("Step 4: 截面中性化...")
    factor_records = []
    
    dates = sorted(spread_df['date'].unique())
    for dt in dates:
        mask = spread_df['date'] == dt
        day_df = spread_df[mask].copy()
        
        if len(day_df) < 50:
            continue
        
        y = day_df['cs_spread_log'].values
        x = day_df['log_amount_ma'].values
        
        # OLS regression: y = a + b*x + residual
        X = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ beta
        except:
            continue
        
        # MAD winsorize
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        if mad < 1e-10:
            continue
        resid_clipped = np.clip(resid, med - 5 * 1.4826 * mad, med + 5 * 1.4826 * mad)
        
        # z-score
        mean_r = resid_clipped.mean()
        std_r = resid_clipped.std()
        if std_r < 1e-10:
            continue
        z = (resid_clipped - mean_r) / std_r
        
        for code, val in zip(day_df['stock_code'].values, z):
            factor_records.append({
                'date': dt,
                'stock_code': code,
                'factor_value': val
            })
    
    factor_df = pd.DataFrame(factor_records)
    
    # 保存
    factor_df.to_csv(output_path, index=False)
    print(f"\n因子保存到: {output_path}")
    print(f"总记录数: {len(factor_df)}")
    print(f"日期范围: {factor_df['date'].min()} ~ {factor_df['date'].max()}")
    print(f"因子统计: mean={factor_df['factor_value'].mean():.4f}, "
          f"std={factor_df['factor_value'].std():.4f}")
    
    return factor_df


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kline_path = os.path.join(base_dir, 'data', 'csi1000_kline_raw.csv')
    output_path = os.path.join(base_dir, 'data', 'factor_cs_spread_v1.csv')
    
    build_factor(kline_path, output_path, window=20)
