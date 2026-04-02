#!/usr/bin/env python3
"""
Corwin-Schultz (2012) Spread Alpha 因子

论文: "A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices"
      Corwin & Schultz, Journal of Finance, 2012, 67(2), 719-759
      https://doi.org/10.1111/j.1540-6261.2012.01729.x

核心改动: A股日频数据上α普遍为负(高波动率环境)，无法直接估算正的spread。
但α值本身在截面上有区分度——α越高(接近0)的股票，隐含spread越大，信息不对称越严重。

因子: α_raw (未转换成spread的原始α值)
方向: 待回测确定
  - 正向(高α → 高收益): 流动性风险溢价假说
  - 负向(低α → 高收益): 高波动/高不确定性惩罚

中性化: 成交额OLS中性化
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def compute_cs_alpha(df_stock: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    对单只股票计算Corwin-Schultz α值
    """
    h = df_stock['high'].values.astype(float)
    l = df_stock['low'].values.astype(float)
    
    n = len(h)
    
    # 单日 β_t = [ln(H_t/L_t)]²
    mask_valid = (h > 0) & (l > 0) & (h >= l)
    ln_hl = np.full(n, np.nan)
    valid_hl = h[mask_valid] / l[mask_valid]
    valid_hl = np.maximum(valid_hl, 1.0 + 1e-10)  # 防止h==l时log(1)=0
    ln_hl[mask_valid] = np.log(valid_hl)
    beta_t = ln_hl ** 2
    
    # 两日合并 γ_t
    if n < 2:
        return pd.Series(np.nan, index=df_stock.index)
    
    h2 = np.maximum(h[1:], h[:-1])
    l2 = np.minimum(l[1:], l[:-1])
    mask2 = (h2 > 0) & (l2 > 0) & (h2 >= l2)
    
    gamma_vals = np.full(n, np.nan)
    gl = np.full(n - 1, np.nan)
    valid_hl2 = h2[mask2] / l2[mask2]
    valid_hl2 = np.maximum(valid_hl2, 1.0 + 1e-10)
    gl[mask2] = np.log(valid_hl2)
    gamma_vals[1:] = gl ** 2
    
    # 滚动均值
    idx = df_stock.index
    beta_s = pd.Series(beta_t, index=idx)
    gamma_s = pd.Series(gamma_vals, index=idx)
    
    min_periods = max(int(window * 0.7), 5)
    beta_mean = beta_s.rolling(window, min_periods=min_periods).mean()
    gamma_mean = gamma_s.rolling(window, min_periods=min_periods).mean()
    
    # α = (√(2β) - √β) / (3 - 2√2) - √(γ/(3 - 2√2))
    k = 3 - 2 * np.sqrt(2)  # ≈ 0.1716
    
    alpha = (np.sqrt(2 * beta_mean) - np.sqrt(beta_mean)) / k - np.sqrt(gamma_mean / k)
    
    return alpha


def neutralize_ols(factor: pd.Series, control: pd.Series) -> pd.Series:
    """OLS中性化"""
    mask = factor.notna() & control.notna()
    if mask.sum() < 10:
        return factor
    y = factor[mask].values
    x = control[mask].values.reshape(-1, 1)
    x = np.column_stack([np.ones(len(x)), x])
    try:
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        residuals = y - x @ beta
        result = pd.Series(np.nan, index=factor.index)
        result[mask] = residuals
        return result
    except:
        return factor


def mad_winsorize(s: pd.Series, n_mad: float = 5.0) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if mad < 1e-10:
        return s
    return s.clip(med - n_mad * mad, med + n_mad * mad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/csi1000_kline_raw.csv')
    parser.add_argument('--output', default='data/factor_cs_spread_v1.csv')
    parser.add_argument('--window', type=int, default=20)
    args = parser.parse_args()
    
    print(f"📊 Corwin-Schultz Alpha Factor (v1)")
    print(f"   窗口: {args.window}日")
    
    df = pd.read_csv(args.data)
    df['date'] = pd.to_datetime(df['date'])
    for col in ['high', 'low', 'amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    n_stocks = df['stock_code'].nunique()
    print(f"   股票数: {n_stocks}")
    
    # 计算每只股票的alpha
    print("⏳ 计算Corwin-Schultz Alpha...")
    all_alphas = []
    for code, grp in df.groupby('stock_code'):
        grp = grp.sort_values('date').reset_index(drop=True)
        alpha = compute_cs_alpha(grp, window=args.window)
        for i, (d, a) in enumerate(zip(grp['date'], alpha)):
            if not np.isnan(a):
                all_alphas.append({
                    'date': d,
                    'stock_code': code,
                    'alpha_raw': a
                })
    
    df_alpha = pd.DataFrame(all_alphas)
    print(f"   原始alpha值: {len(df_alpha)} 条")
    print(f"   alpha范围: [{df_alpha['alpha_raw'].min():.4f}, {df_alpha['alpha_raw'].max():.4f}]")
    print(f"   alpha均值: {df_alpha['alpha_raw'].mean():.4f}")
    print(f"   alpha>0比例: {(df_alpha['alpha_raw']>0).mean():.4f}")
    
    # 合并20日均成交额
    df_sort = df.sort_values(['stock_code', 'date'])
    df_sort['amount_20d'] = df_sort.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    df_amt = df_sort[['date', 'stock_code', 'amount_20d']].copy()
    df_alpha = df_alpha.merge(df_amt, on=['date', 'stock_code'], how='left')
    
    # 横截面处理
    print("⏳ 横截面中性化 + 标准化...")
    results = []
    for date, grp in df_alpha.groupby('date'):
        grp = grp.copy()
        grp['log_amt'] = np.log(grp['amount_20d'].clip(lower=1))
        
        # MAD winsorize alpha
        grp['alpha_w'] = mad_winsorize(grp['alpha_raw'], 5.0)
        
        # 成交额中性化
        grp['factor'] = neutralize_ols(grp['alpha_w'], grp['log_amt'])
        
        # z-score
        mu = grp['factor'].mean()
        sd = grp['factor'].std()
        if sd > 1e-10:
            grp['factor'] = (grp['factor'] - mu) / sd
        else:
            continue
        
        for _, row in grp.iterrows():
            if not np.isnan(row['factor']):
                results.append({
                    'date': date,
                    'stock_code': row['stock_code'],
                    'factor_value': row['factor']
                })
    
    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values(['date', 'stock_code'])
    
    print(f"   最终因子值: {len(df_out)} 条")
    vals = df_out['factor_value']
    print(f"   均值: {vals.mean():.4f}, 标准差: {vals.std():.4f}")
    print(f"   偏度: {vals.skew():.4f}, 峰度: {vals.kurtosis():.4f}")
    
    df_out.to_csv(args.output, index=False)
    print(f"\n✅ 因子值已保存到: {args.output}")


if __name__ == '__main__':
    main()
