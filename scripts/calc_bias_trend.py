#!/usr/bin/env python3
"""
因子: 均线乖离率趋势 (MA Bias Trend / MA Deviation Slope)
ID: bias_trend_v1

逻辑:
  1. 计算收盘价相对于MA20的乖离率: bias = (close - MA20) / MA20
  2. 对过去10日的bias做线性回归取斜率, 衡量乖离率的变化速度
  3. 正值=价格加速远离MA(趋势加速), 负值=价格回归MA(趋势减速/反转)
  4. 成交额OLS中性化 + MAD winsorize + z-score

假设:
  - 乖离率加速上升的股票趋势延续(动量), 或回归加速的反转
  - 方向待回测确认

Barra: 趋势/Momentum
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calc_bias_trend(kline_path, output_path, ma_window=20, slope_window=10):
    """计算均线乖离率趋势因子"""
    
    print(f"加载数据: {kline_path}")
    df = pd.read_csv(kline_path, parse_dates=['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # 过滤无效数据
    df = df[df['close'] > 0].copy()
    
    results = []
    
    for code, g in df.groupby('stock_code'):
        g = g.sort_values('date').copy()
        
        # 1. MA
        g['ma'] = g['close'].rolling(ma_window, min_periods=ma_window).mean()
        
        # 2. 乖离率
        g['bias'] = (g['close'] - g['ma']) / g['ma']
        
        # 3. 乖离率线性回归斜率 (slope_window日)
        bias_vals = g['bias'].values
        slopes = np.full(len(g), np.nan)
        
        x = np.arange(slope_window)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        
        for i in range(slope_window - 1 + ma_window - 1, len(g)):
            window = bias_vals[i - slope_window + 1: i + 1]
            if np.isnan(window).any():
                continue
            y_mean = window.mean()
            slopes[i] = ((x * (window - y_mean)).sum()) / x_var
        
        g['factor_raw'] = slopes
        
        # 4. 20日平均成交额 (用于中性化)
        g['log_amount_20d'] = np.log(g['amount'].rolling(20, min_periods=10).mean() + 1)
        
        results.append(g[['date', 'stock_code', 'factor_raw', 'log_amount_20d']].copy())
    
    all_data = pd.concat(results, ignore_index=True)
    all_data = all_data.dropna(subset=['factor_raw', 'log_amount_20d'])
    
    print(f"原始因子: {len(all_data)} rows, {all_data['stock_code'].nunique()} stocks")
    
    # 5. 截面处理: 成交额OLS中性化 + MAD winsorize + z-score
    final_rows = []
    for dt, group in all_data.groupby('date'):
        if len(group) < 50:
            continue
        
        vals = group['factor_raw'].values.copy()
        amt = group['log_amount_20d'].values.copy()
        
        # MAD winsorize (先做，减少极端值对OLS的影响)
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        if mad < 1e-10:
            continue
        upper = med + 5 * 1.4826 * mad
        lower = med - 5 * 1.4826 * mad
        vals = np.clip(vals, lower, upper)
        
        # OLS 成交额中性化
        mask = ~(np.isnan(vals) | np.isnan(amt) | np.isinf(vals) | np.isinf(amt))
        if mask.sum() < 30:
            continue
        
        v = vals[mask]
        a = amt[mask]
        
        # OLS: factor = alpha + beta * log_amount + residual
        X = np.column_stack([np.ones(len(a)), a])
        try:
            beta = np.linalg.lstsq(X, v, rcond=None)[0]
            residuals = v - X @ beta
        except:
            continue
        
        # MAD winsorize on residuals
        med_r = np.median(residuals)
        mad_r = np.median(np.abs(residuals - med_r))
        if mad_r < 1e-10:
            continue
        upper_r = med_r + 3 * 1.4826 * mad_r
        lower_r = med_r - 3 * 1.4826 * mad_r
        residuals = np.clip(residuals, lower_r, upper_r)
        
        # z-score
        std_r = residuals.std()
        if std_r < 1e-10:
            continue
        z = (residuals - residuals.mean()) / std_r
        
        sub = group.iloc[np.where(mask)[0]].copy()
        sub['factor'] = z
        final_rows.append(sub[['date', 'stock_code', 'factor']])
    
    result = pd.concat(final_rows, ignore_index=True)
    result = result.sort_values(['date', 'stock_code']).reset_index(drop=True)
    
    print(f"最终因子: {len(result)} rows, dates: {result['date'].min()} ~ {result['date'].max()}")
    print(f"因子统计: mean={result['factor'].mean():.4f}, std={result['factor'].std():.4f}")
    
    result.to_csv(output_path, index=False)
    print(f"已保存: {output_path}")
    
    return result

if __name__ == '__main__':
    base = Path(__file__).resolve().parent.parent
    kline_path = base / 'data' / 'csi1000_kline_raw.csv'
    output_path = base / 'data' / 'factor_bias_trend_v1.csv'
    
    calc_bias_trend(kline_path, output_path, ma_window=20, slope_window=10)
