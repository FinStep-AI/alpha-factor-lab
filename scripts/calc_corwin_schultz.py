#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corwin-Schultz (2012) 隐含买卖价差因子

论文: "A Simple Approximation of Bid-Ask Spreads from High and Low Prices"
      Shane A. Corwin & Paul Schultz, Journal of Finance, 2012
      https://doi.org/10.1111/j.1540-6261.2012.01729.x

核心公式:
  beta = sum of [ln(H_t/L_t)]^2 for two consecutive days
  gamma = [ln(max(H_t,H_{t-1})/min(L_t,L_{t-1}))]^2  (2-day range)
  alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma/(3-2*sqrt(2)))
  S = 2*(exp(alpha)-1) / (1+exp(alpha))

中性化: 成交额OLS中性化 + MAD winsorize + z-score
"""

import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")

def compute_corwin_schultz_spread(df, window=20):
    """
    对每只股票计算Corwin-Schultz隐含价差的滚动均值
    
    Parameters
    ----------
    df : DataFrame with columns [date, stock_code, high, low, close, amount]
    window : int, 滚动窗口(天数)
    
    Returns
    -------
    DataFrame with columns [date, stock_code, factor_value]
    """
    k = 3 - 2 * np.sqrt(2)  # ≈ 0.1716
    
    results = []
    
    for code, gdf in df.groupby('stock_code'):
        gdf = gdf.sort_values('date').reset_index(drop=True)
        
        h = gdf['high'].values
        l = gdf['low'].values
        
        n = len(gdf)
        spread = np.full(n, np.nan)
        
        for i in range(1, n):
            # 单日 ln(H/L)^2
            hl_ratio_t = np.log(h[i] / l[i]) if l[i] > 0 and h[i] > 0 else np.nan
            hl_ratio_t1 = np.log(h[i-1] / l[i-1]) if l[i-1] > 0 and h[i-1] > 0 else np.nan
            
            if np.isnan(hl_ratio_t) or np.isnan(hl_ratio_t1):
                continue
            
            beta = hl_ratio_t**2 + hl_ratio_t1**2
            
            # 两日合并范围
            h2 = max(h[i], h[i-1])
            l2 = min(l[i], l[i-1])
            if l2 <= 0 or h2 <= 0:
                continue
            gamma = (np.log(h2 / l2))**2
            
            # alpha
            sqrt_beta = np.sqrt(beta)
            alpha = (np.sqrt(2) * sqrt_beta - sqrt_beta) / k - np.sqrt(gamma / k)
            
            # 当alpha < 0时, 设为0 (价差不能为负)
            alpha = max(alpha, 0)
            
            # spread
            ea = np.exp(alpha)
            s = 2 * (ea - 1) / (1 + ea)
            spread[i] = s
        
        # 滚动窗口均值
        spread_series = pd.Series(spread)
        spread_ma = spread_series.rolling(window=window, min_periods=int(window*0.5)).mean()
        
        for i in range(n):
            if not np.isnan(spread_ma.iloc[i]):
                results.append({
                    'date': gdf['date'].iloc[i],
                    'stock_code': code,
                    'raw_spread': spread_ma.iloc[i]
                })
    
    return pd.DataFrame(results)


def neutralize_and_standardize(factor_df, kline_df, neutralize_col='log_amount_20d'):
    """
    成交额OLS中性化 + MAD winsorize + z-score
    """
    from sklearn.linear_model import LinearRegression
    
    # 计算20日平均成交额
    amt_20d = kline_df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    kline_df = kline_df.copy()
    kline_df['log_amount_20d'] = np.log(amt_20d + 1)
    
    # merge
    merged = factor_df.merge(
        kline_df[['date', 'stock_code', 'log_amount_20d']],
        on=['date', 'stock_code'],
        how='left'
    )
    
    # 按日期截面中性化
    results = []
    for date, gdf in merged.groupby('date'):
        gdf = gdf.dropna(subset=['raw_spread', 'log_amount_20d'])
        if len(gdf) < 50:
            continue
        
        X = gdf[['log_amount_20d']].values
        y = gdf['raw_spread'].values
        
        # MAD winsorize before regression
        median_y = np.median(y)
        mad = np.median(np.abs(y - median_y))
        if mad > 0:
            upper = median_y + 5 * 1.4826 * mad
            lower = median_y - 5 * 1.4826 * mad
            y = np.clip(y, lower, upper)
        
        # OLS中性化
        lr = LinearRegression()
        lr.fit(X, y)
        residual = y - lr.predict(X)
        
        # MAD winsorize residual
        med_r = np.median(residual)
        mad_r = np.median(np.abs(residual - med_r))
        if mad_r > 0:
            upper_r = med_r + 3 * 1.4826 * mad_r
            lower_r = med_r - 3 * 1.4826 * mad_r
            residual = np.clip(residual, lower_r, upper_r)
        
        # z-score
        std_r = np.std(residual)
        if std_r > 0:
            z = (residual - np.mean(residual)) / std_r
        else:
            z = np.zeros_like(residual)
        
        for idx, (_, row) in enumerate(gdf.iterrows()):
            results.append({
                'date': date,
                'stock_code': row['stock_code'],
                'factor_value': z[idx]
            })
    
    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("Corwin-Schultz (2012) 隐含买卖价差因子")
    print("=" * 60)
    
    # 读取数据
    data_dir = '/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab/data'
    print("\n1. 读取行情数据...")
    kline = pd.read_csv(f'{data_dir}/csi1000_kline_raw.csv')
    kline['date'] = pd.to_datetime(kline['date'])
    print(f"   行数: {len(kline)}, 股票数: {kline['stock_code'].nunique()}")
    print(f"   日期范围: {kline['date'].min()} ~ {kline['date'].max()}")
    
    # 计算隐含价差
    print("\n2. 计算Corwin-Schultz隐含价差 (20日滚动均值)...")
    spread_df = compute_corwin_schultz_spread(kline, window=20)
    print(f"   有效记录数: {len(spread_df)}")
    print(f"   价差统计:")
    print(f"     mean={spread_df['raw_spread'].mean():.6f}")
    print(f"     median={spread_df['raw_spread'].median():.6f}")
    print(f"     std={spread_df['raw_spread'].std():.6f}")
    print(f"     min={spread_df['raw_spread'].min():.6f}")
    print(f"     max={spread_df['raw_spread'].max():.6f}")
    
    # 中性化 + 标准化
    print("\n3. 成交额OLS中性化 + MAD winsorize + z-score...")
    factor_df = neutralize_and_standardize(spread_df, kline)
    print(f"   最终因子记录数: {len(factor_df)}")
    
    # 检查方向: 高价差(低流动性) → 可能高收益(流动性溢价)
    # 先用正向(高隐含价差=高因子值), 回测时看方向
    print("\n4. 保存因子值...")
    
    # pivot成宽表
    factor_pivot = factor_df.pivot(index='date', columns='stock_code', values='factor_value')
    output_path = f'{data_dir}/factor_cs_spread_v1.csv'
    factor_pivot.to_csv(output_path)
    print(f"   保存到: {output_path}")
    print(f"   日期数: {len(factor_pivot)}")
    print(f"   股票数: {factor_pivot.shape[1]}")
    
    # 也保存反向版本
    factor_df_neg = factor_df.copy()
    factor_df_neg['factor_value'] = -factor_df_neg['factor_value']
    factor_pivot_neg = factor_df_neg.pivot(index='date', columns='stock_code', values='factor_value')
    output_path_neg = f'{data_dir}/factor_cs_spread_neg_v1.csv'
    factor_pivot_neg.to_csv(output_path_neg)
    print(f"   反向版本: {output_path_neg}")
    
    print("\n✅ 因子计算完成!")
    print("   正向: 高隐含价差 = 高因子值 (流动性溢价假说)")
    print("   反向: 低隐含价差 = 高因子值 (流动性偏好假说)")


if __name__ == '__main__':
    main()
