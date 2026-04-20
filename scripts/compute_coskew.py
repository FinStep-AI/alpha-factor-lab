#!/usr/bin/env python3
"""
因子: 市场协偏度 (Coskewness v1)
reference: Harvey & Siddique (2000) JF, Bali, Cakici & Whitelaw (2011) JFE

逻辑:
  协偏度 = Cov(r_i, r_mkt^2) / Var(r_mkt^2)
  衡量个股对市场极端收益的暴露度
  高coskewness = 市场大涨时弹性更大 = 承担系统性跳跃风险 → 风险溢价 → 高收益
  方向: 正向

公式:
  ret_mkt_sq = 等权中证1000日收益的平方 (20日均)
  coskew_20d = corr(ret_i, ret_mkt_sq, 20d)
  中性化: log(1+|coskew|) × sign(coskew) → OLS neutralized by log_amount_20d

Barra风格: Risk
"""

import numpy as np
import pandas as pd
import sys

def compute_coskewness_factor(data_path, returns_path):
    print("[INFO] 加载K线数据...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print("[INFO] 计算市场等权收益...")
    # 等权市场日收益
    mkt_ret = df.groupby('date').apply(
        lambda x: x['pct_change'].mean() / 100 if 'pct_change' in x.columns else x['close'].pct_change().mean()
    )
    # 用close重新计算
    daily_close = df.groupby('date')['close'].mean()
    mkt_ret = daily_close.pct_change()
    mkt_ret.name = 'mkt_ret'
    mkt_ret.index = pd.to_datetime(mkt_ret.index)
    
    # 市场收益平方
    mkt_ret_sq = mkt_ret ** 2
    mkt_ret_sq.name = 'mkt_ret_sq'
    
    # 合并到个股数据
    df = df.merge(mkt_ret_sq, left_on='date', right_index=True, how='left')
    
    print("[INFO] 计算个股收益率...")
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    grouped = df.groupby('stock_code')
    df['ret'] = grouped['close'].pct_change()
    
    print("[INFO] 计算20日协偏度 (corr(ret_i, r_mkt^2))...")
    # 20日滚动corr(ret, market_ret_squared)
    def rolling_coskew(group):
        ret = group['ret']
        mkt_sq = group['mkt_ret_sq']
        return ret.rolling(20, min_periods=10).corr(mkt_sq)
    
    df['coskew_raw'] = grouped.apply(
        lambda g: g['ret'].rolling(20, min_periods=10).corr(g['mkt_ret_sq'])
    ).reset_index(level=0, drop=True)
    
    # 处理NaN: 全NaN日子会留下，需要drop
    df['factor_raw'] = df['coskew_raw']
    
    # 成交额中性化
    df['log_amount_20d'] = np.log(df['amount'].rolling(20, min_periods=10).mean() + 1)
    
    # 截面中性化
    results = []
    for date, group in df.groupby('date'):
        g = group.dropna(subset=['factor_raw', 'log_amount_20d'])
        if len(g) < 30:
            continue
        x = g['log_amount_20d'].values
        y = g['factor_raw'].values
        
        X = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residual = y - X @ beta
        except:
            continue
        
        # MAD winsorize
        med = np.median(residual)
        mad = np.median(np.abs(residual - med)) * 1.4826
        if mad < 1e-10:
            continue
        upper, lower = med + 3 * mad, med - 3 * mad
        residual = np.clip(residual, lower, upper)
        
        # z-score
        mu, sigma = residual.mean(), residual.std()
        if sigma < 1e-10:
            continue
        z = (residual - mu) / sigma
        
        g = g.copy()
        g['factor_neutral'] = z
        results.append(g[['date', 'stock_code', 'factor_neutral']])
    
    if not results:
        print("ERROR: no data after neutralization", file=sys.stderr)
        sys.exit(1)
    
    result_df = pd.concat(results, ignore_index=True)
    print(f"[INFO] Coskewness 因子: {result_df['date'].min()} ~ {result_df['date'].max()}, "
          f"{result_df['stock_code'].nunique()} stocks")
    print(f"[INFO] stats: mean={result_df['factor_neutral'].mean():.4f}, std={result_df['factor_neutral'].std():.4f}")
    
    out_path = 'data/factor_coskew_v1.csv'
    result_df.to_csv(out_path, index=False)
    print(f"[INFO] 保存到 {out_path}")
    return out_path

if __name__ == '__main__':
    compute_coskewness_factor(
        'data/csi1000_kline_raw.csv',
        'data/csi1000_returns.csv'
    )
