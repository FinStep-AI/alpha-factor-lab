#!/usr/bin/env python3
"""
因子：收益极值不对称性 (Return Extreme Asymmetry)  
ID: ret_asym_v1

灵感：CVaR因子(看最差2天)是全库最强(Sharpe=2.04)
本因子：看最差N天和最好N天的不对称性

逻辑：
  worst_k = mean(bottom K returns, window)  (负数)
  best_k = mean(top K returns, window)  (正数)
  asymmetry = best_k + worst_k  
    (注意best_k>0, worst_k<0)
    
  asymmetry > 0: 极端正收益 > 极端负收益 → "右偏" → 可能过度乐观
  asymmetry < 0: 极端负收益 > 极端正收益 → "左偏" → 可能过度悲观
  
  反向使用：做多左偏(asymmetry<0)，做空右偏(asymmetry>0)
  即：极端亏损>极端盈利 的股票 → 被市场过度惩罚 → 后续反弹
  
  这与CVaR不同：CVaR只看下行极端，本因子看上下极端的不对称性
  
  参数: window=10, K=2 (与CVaR一致)
"""

import pandas as pd
import numpy as np
import os

def compute_factor(kline_path, output_path):
    print("Loading kline data...")
    df = pd.read_csv(kline_path, dtype={'stock_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    df['daily_ret'] = df.groupby('stock_code')['close'].pct_change()
    
    window = 10
    k = 2
    
    print(f"Computing return extreme asymmetry (window={window}, k={k})...")
    
    g = df.groupby('stock_code')
    
    # 需要自定义rolling function来取top/bottom k
    def rolling_asymmetry(series):
        result = np.full(len(series), np.nan)
        vals = series.values
        for i in range(window, len(vals)+1):
            w = vals[i-window:i]
            valid = w[~np.isnan(w)]
            if len(valid) < window * 0.8:
                continue
            sorted_v = np.sort(valid)
            worst_k = np.mean(sorted_v[:k])  # bottom k (most negative)
            best_k = np.mean(sorted_v[-k:])   # top k (most positive)
            # asymmetry: best + worst
            # negative when worst is more extreme than best
            result[i-1] = -(best_k + worst_k)  # negate: we want to buy "left-skewed"
        return pd.Series(result, index=series.index)
    
    df['factor_raw'] = g['daily_ret'].transform(rolling_asymmetry)
    
    # log(20日平均成交额)
    df['mean_amt'] = g['amount'].transform(
        lambda x: x.rolling(20, min_periods=16).mean()
    )
    df['log_amount_20d'] = np.log(df['mean_amt'].clip(lower=1))
    
    factor_df = df[['date', 'stock_code', 'factor_raw', 'log_amount_20d']].dropna().copy()
    print(f"Raw factor: {len(factor_df)} rows, {factor_df['date'].nunique()} dates")
    
    # 截面中性化
    print("Neutralizing...")
    def neutralize_cs(group):
        y = group['factor_raw'].values.copy()
        x = group['log_amount_20d'].values.copy()
        valid = np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 30:
            return pd.Series(np.nan, index=group.index, name='factor')
        y_v, x_v = y[valid], x[valid]
        med = np.median(y_v)
        mad = np.median(np.abs(y_v - med)) * 1.4826
        if mad > 0:
            y_v = np.clip(y_v, med - 3*mad, med + 3*mad)
        x_mat = np.column_stack([np.ones(len(x_v)), x_v])
        beta = np.linalg.lstsq(x_mat, y_v, rcond=None)[0]
        resid = y_v - x_mat @ beta
        std = np.std(resid)
        if std > 0:
            resid = (resid - np.mean(resid)) / std
        out = np.full(len(y), np.nan)
        out[valid] = resid
        return pd.Series(out, index=group.index, name='factor')
    
    factor_df['factor'] = factor_df.groupby('date', group_keys=False).apply(
        lambda g: neutralize_cs(g)
    ).values
    
    output = factor_df[['date', 'stock_code', 'factor']].dropna().copy()
    output['stock_code'] = output['stock_code'].astype(str).str.zfill(6)
    output = output.sort_values(['date', 'stock_code'])
    output.to_csv(output_path, index=False)
    print(f"Factor saved: {output_path} ({len(output)} rows)")
    print(f"Date range: {output['date'].min()} ~ {output['date'].max()}")
    print(f"Stocks/date: {output.groupby('date')['stock_code'].count().mean():.0f}")

if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    compute_factor(
        os.path.join(base, 'data', 'csi1000_kline_raw.csv'),
        os.path.join(base, 'data', 'factor_ret_asym_v1.csv')
    )
