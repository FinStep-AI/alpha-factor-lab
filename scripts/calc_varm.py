#!/usr/bin/env python3
"""
Volume-Adjusted Return Moderation (VARM) Factor v1
基准：方正金工「适度冒险因子」（2022）的日度适应性改造

核心逻辑改版：用异常收益率（而非成交量激增）来定位信息冲击时刻
  Step 1: 标记异常日（|ret| > MA20(|ret|) + 1σ），代表信息冲击/投资者极端关注时刻
  Step 2: 计算冲击日的|ret|和波动率
  Step 3: 反转型因子：冲击日|ret|越小 → 市场越"适度" → 正alpha
  期望：IC>0.02, t>2.0, Sharpe>0.8, 单调性>0.8
"""
import numpy as np
import pandas as pd

def calc(kline_path='data/csi1000_kline_raw.csv', output_path=None):
    df = pd.read_csv(kline_path, parse_dates=['date'])
    df['code'] = df['stock_code']
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    
    # Daily log return
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['abs_ret'] = df['log_ret'].abs()
    
    grp = df.groupby('code')
    
    # 20-day rolling MA and std of |ret|
    df['ret_ma20'] = grp['abs_ret'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df['ret_std20'] = grp['abs_ret'].transform(lambda x: x.rolling(20, min_periods=10).std())
    
    # Flag "information shock" days: |ret| > MA20 + 1σ  
    # (信息冲击日 = 市场对信息有显著反应的日期)
    df['ret_shock'] = (df['abs_ret'] > (df['ret_ma20'] + df['ret_std20'])).astype(int)
    df['ret_shock_amt'] = df['abs_ret'].where(df['ret_shock'] == 1)
    
    # Factor 1: 20日均冲击|ret| (越低越好) — 等价于"月均耀眼收益率"的相反
    df['shock_ret_ma20'] = grp['ret_shock_amt'].transform(
        lambda x: x.rolling(20, min_periods=3).mean()
    )
    
    # Factor 2: 20日冲击|ret|的变异系数 (越低=反应越一致=越好)
    df['shock_ret_cv20'] = grp['ret_shock_amt'].transform(
        lambda x: x.rolling(20, min_periods=3).mean()
    ) / (grp['ret_shock_amt'].transform(lambda x: x.rolling(20, min_periods=3).std()) + 1e-8)
    df['shock_ret_cv20'] = -df['shock_ret_cv20']  # 低CV = 好
    
    # Factor 3 (合成): 原始因子 = -(冲击|ret|均值 + 冲击|ret|标准差)
    # 等权合成"月均"和"月稳"维度 (对应原文公式)
    df['shock_ret_std20'] = grp['ret_shock_amt'].transform(
        lambda x: x.rolling(20, min_periods=3).std()
    )
    df['raw_factor'] = -(df['shock_ret_ma20'] + df['shock_ret_std20']) / 2
    
    # 市值中性化 (用20日均成交额)
    df['log_amount_20'] = grp['amount'].transform(lambda x: np.log(x.rolling(20).mean().replace(0, np.nan)))
    
    result = df[['date', 'code', 'raw_factor', 'log_amount_20']].dropna(subset=['raw_factor', 'log_amount_20']).copy()
    
    # OLS neutralize against market cap proxy
    from sklearn.linear_model import LinearRegression
    all_results = []
    for date, group in result.groupby('date'):
        X = group['log_amount_20'].values.reshape(-1, 1)
        y = group['raw_factor'].values
        lr = LinearRegression().fit(X, y)
        residual = y - lr.predict(X)
        sub = group[['date', 'code']].copy()
        sub['varm_v1'] = residual
        all_results.append(sub)
    
    result = pd.concat(all_results, ignore_index=True)
    
    # Cross-section z-score
    def zscore(g):
        mu, std = g['varm_v1'].mean(), g['varm_v1'].std()
        if std > 0:
            g['varm_v1'] = (g['varm_v1'] - mu) / std
        else:
            g['varm_v1'] = 0
        return g
    
    result = result.groupby('date', group_keys=False).apply(zscore)
    
    if output_path is None:
        output_path = 'data/factor_varm_v1.csv'
    
    result.to_csv(output_path, index=False)
    print(f"✅ Factor saved: {output_path} | {len(result)} rows, {result['code'].nunique()} stocks")
    print(f"   Last date: {result['date'].max()}, factor mean: {result['varm_v1'].mean():.4f}")
    return output_path

if __name__ == '__main__':
    calc()
