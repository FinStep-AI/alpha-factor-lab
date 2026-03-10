#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
收益率自相关非对称性因子 (Return Autocorrelation Asymmetry v1)
===============================================================

核心思路:
  分别计算上涨日和下跌日的收益率自相关系数(lag-1)，取差值。
  
  autocorr_up = corr(ret_t, ret_{t+1}) when ret_t > 0
  autocorr_down = corr(ret_t, ret_{t+1}) when ret_t < 0
  
  asymmetry = autocorr_down - autocorr_up
  
  经济学直觉:
  - 高不对称性(下跌自相关 > 上涨自相关) → 下跌有惯性/恐慌连锁 → 风险高
  - 低不对称性(上涨自相关 ≥ 下跌自相关) → 上涨更有持续性 → 看好
  
  反向使用: 做多低不对称性（上涨有延续性的股票）
  
  这与Glosten, Jagannathan & Runkle (1993) 的"波动率非对称效应"相关。
  
  20日窗口，实际计算时用条件收益率序列的相关系数。

替代方案 - 更简洁的定义:
  对于20日窗口内的收益序列，计算:
    up_continuation_ratio = P(ret_t > 0 | ret_{t-1} > 0) - 条件涨幅
    down_continuation_ratio = P(ret_t < 0 | ret_{t-1} < 0) - 条件跌幅
  
  factor = up_continuation_ratio - down_continuation_ratio
  
  正值 = 上涨有惯性但下跌无惯性 → 趋势性好 → 做多
"""

import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def compute_ret_autocorr_asym(kline_path: str, output_path: str, window: int = 20):
    print(f"{'='*60}")
    print(f"Return Autocorrelation Asymmetry (window={window})")
    print(f"{'='*60}")
    
    # 1. Load data
    print("\n[1] 加载数据...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    n_stocks = df['stock_code'].nunique()
    n_dates = df['date'].nunique()
    print(f"    股票数: {n_stocks}, 日期数: {n_dates}")
    
    # 2. Daily returns
    print("[2] 计算日收益率...")
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    
    # 3. Rolling asymmetry
    print(f"[3] 计算{window}日滚动自相关不对称性...")
    
    def calc_asym(group):
        group = group.sort_values('date').copy()
        ret = group['ret'].values
        n = len(ret)
        asym_vals = np.full(n, np.nan)
        
        for i in range(window, n):
            r = ret[i - window:i + 1]  # window+1 values for lag-1 pairs
            valid = ~np.isnan(r)
            if valid.sum() < window * 0.75:
                continue
            
            # Get consecutive pairs
            r_prev = r[:-1]
            r_curr = r[1:]
            pair_valid = ~np.isnan(r_prev) & ~np.isnan(r_curr)
            r_prev = r_prev[pair_valid]
            r_curr = r_curr[pair_valid]
            
            if len(r_prev) < 10:
                continue
            
            # Up continuation: P(curr > 0 | prev > 0)
            prev_up = r_prev > 0
            prev_down = r_prev < 0
            
            n_prev_up = prev_up.sum()
            n_prev_down = prev_down.sum()
            
            if n_prev_up < 3 or n_prev_down < 3:
                continue
            
            # Conditional probability approach
            up_cont = (r_curr[prev_up] > 0).sum() / n_prev_up
            down_cont = (r_curr[prev_down] < 0).sum() / n_prev_down
            
            # Also consider magnitude
            # Up continuation: mean return after up day
            up_mean_ret = r_curr[prev_up].mean()
            down_mean_ret = r_curr[prev_down].mean()
            
            # Asymmetry: combines both probability and magnitude
            # Positive = up continuation strong, down continuation weak → bullish
            asym_vals[i] = (up_cont - down_cont) + (up_mean_ret + down_mean_ret) * 10
            # Note: +down_mean_ret because if mean ret after down day is positive,
            # it means the stock bounces back (good sign)
        
        group['raw_asym'] = asym_vals
        return group
    
    df = df.groupby('stock_code', group_keys=False).apply(calc_asym)
    
    # 4. log_amount for neutralization
    print("[4] 计算市值代理...")
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(window, min_periods=int(window*0.75)).mean() + 1)
    )
    
    # 5. Winsorize 5%
    print("[5] 5%缩尾...")
    def winsorize_cs(group):
        v = group['raw_asym']
        m = v.notna()
        if m.sum() < 10:
            return v
        lower = v[m].quantile(0.05)
        upper = v[m].quantile(0.95)
        return v.clip(lower, upper)
    
    df['raw_asym'] = df.groupby('date', group_keys=False).apply(winsorize_cs)
    
    # 6. OLS neutralization
    print("[6] OLS市值中性化...")
    def neutralize(group):
        y = group['raw_asym']
        x = group['log_amount_20d']
        mask = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            return pd.Series(np.nan, index=group.index)
        y_c = y[mask].values
        x_c = x[mask].values
        X = np.column_stack([np.ones(len(x_c)), x_c])
        try:
            beta = np.linalg.lstsq(X, y_c, rcond=None)[0]
            residuals = y_c - X @ beta
            result = pd.Series(np.nan, index=group.index)
            result[mask] = residuals
            return result
        except Exception:
            return pd.Series(np.nan, index=group.index)
    
    df['factor'] = df.groupby('date', group_keys=False).apply(neutralize)
    
    # 7. Cross-sectional z-score
    print("[7] 截面z-score标准化...")
    def zscore_cs(group):
        v = group['factor']
        m = v.notna()
        if m.sum() < 10:
            return v
        mu = v[m].mean()
        s = v[m].std()
        if s < 1e-10:
            return v * 0
        return (v - mu) / s
    
    df['factor'] = df.groupby('date', group_keys=False).apply(zscore_cs)
    
    # 8. Output
    print("[8] 输出因子值...")
    output = df[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    output = output.rename(columns={'factor': 'factor_value'})
    output.to_csv(output_path, index=False)
    
    n_valid = output.shape[0]
    n_dates_valid = output['date'].nunique()
    avg_stocks = output.groupby('date')['stock_code'].nunique().median()
    
    print(f"\n✅ 因子计算完成!")
    print(f"   输出: {output_path}")
    print(f"   有效记录: {n_valid:,}")
    print(f"   有效日期: {n_dates_valid}")
    print(f"   平均每日股票数: {avg_stocks:.0f}")


if __name__ == "__main__":
    kline_path = sys.argv[1] if len(sys.argv) > 1 else "data/csi1000_kline_raw.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/factor_ret_autocorr_asym_v1.csv"
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    compute_ret_autocorr_asym(kline_path, output_path, window)
