#!/usr/bin/env python3
"""
共偏度因子 (Coskewness Factor)
来源: Harvey & Siddique (2000) "Conditional Skewness in Asset Pricing Tests", Journal of Finance
公式: coskew_i = E[(r_i - μ_i)(r_m - μ_m)²] / [σ_i × σ_m²]
窗口: 60日滚动
方向: 负向（低共偏度 → 高预期收益）
中性化: 成交额OLS中性化 + MAD winsorize + z-score
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

print("=== 共偏度因子 (Coskewness) ===")
print("来源: Harvey & Siddique (2000) JF\n")

# 1. 读取数据
kline = pd.read_csv("data/csi1000_kline_raw.csv")
kline['date'] = pd.to_datetime(kline['date'])
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)

# 计算日收益率
kline['returns'] = kline.groupby('stock_code')['close'].pct_change()

# 计算市场等权收益率
mkt = kline.groupby('date')['returns'].mean().rename('mkt_ret')
kline = kline.merge(mkt, on='date', how='left')

print(f"数据: {kline['stock_code'].nunique()} 只股票, {kline['date'].nunique()} 个交易日")
print(f"日期范围: {kline['date'].min().date()} ~ {kline['date'].max().date()}")

# 2. 计算60日滚动共偏度
WINDOW = 60
MIN_PERIODS = 40

def rolling_coskewness(group):
    """对单只股票计算滚动共偏度"""
    r_i = group['returns'].values
    r_m = group['mkt_ret'].values
    dates = group['date'].values
    n = len(r_i)
    
    coskew = np.full(n, np.nan)
    
    for t in range(MIN_PERIODS - 1, n):
        start = max(0, t - WINDOW + 1)
        ri_w = r_i[start:t+1]
        rm_w = r_m[start:t+1]
        
        mask = ~(np.isnan(ri_w) | np.isnan(rm_w))
        if mask.sum() < MIN_PERIODS:
            continue
        
        ri_w = ri_w[mask]
        rm_w = rm_w[mask]
        
        ri_dm = ri_w - ri_w.mean()
        rm_dm = rm_w - rm_w.mean()
        
        sigma_i = ri_w.std(ddof=1)
        sigma_m = rm_w.std(ddof=1)
        
        if sigma_i < 1e-10 or sigma_m < 1e-10:
            continue
        
        # coskewness = E[(r_i - μ_i)(r_m - μ_m)²] / [σ_i × σ_m²]
        coskew_val = (ri_dm * rm_dm**2).mean() / (sigma_i * sigma_m**2)
        coskew[t] = coskew_val
    
    return pd.Series(coskew, index=group.index)

print("\n计算60日滚动共偏度...")
coskew_values = kline.groupby('stock_code', group_keys=False).apply(rolling_coskewness)
kline['coskew_raw'] = coskew_values

# 取负值：低共偏度 → 高因子值 → 高预期收益
kline['factor_raw'] = -kline['coskew_raw']

# 3. 成交额中性化 + MAD winsorize + z-score
print("成交额中性化...")

# 20日平均成交额(对数)
kline['log_amount_20d'] = kline.groupby('stock_code')['amount'].transform(
    lambda x: np.log(x.rolling(20, min_periods=10).mean().replace(0, np.nan))
)

from sklearn.linear_model import LinearRegression

def neutralize_and_standardize(df_day):
    """单日截面中性化+标准化"""
    mask = df_day[['factor_raw', 'log_amount_20d']].notna().all(axis=1)
    sub = df_day[mask].copy()
    
    if len(sub) < 50:
        return pd.Series(np.nan, index=df_day.index, name='factor')
    
    X = sub[['log_amount_20d']].values
    y = sub['factor_raw'].values
    
    # MAD winsorize on raw factor
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad > 0:
        scaled_mad = 1.4826 * mad
        y = np.clip(y, med - 3 * scaled_mad, med + 3 * scaled_mad)
    
    # OLS中性化
    lr = LinearRegression()
    lr.fit(X, y)
    residuals = y - lr.predict(X)
    
    # z-score
    mu, sigma = residuals.mean(), residuals.std()
    if sigma > 0:
        z = (residuals - mu) / sigma
    else:
        z = np.zeros_like(residuals)
    z = np.clip(z, -3, 3)
    
    result = pd.Series(np.nan, index=df_day.index, name='factor')
    result.iloc[mask.values] = z
    return result

factor_vals = kline.groupby('date', group_keys=False).apply(neutralize_and_standardize)
kline['factor'] = factor_vals.values if hasattr(factor_vals, 'values') else factor_vals

# 4. 输出
output = kline[['date', 'stock_code', 'factor']].dropna(subset=['factor']).copy()
output['date'] = output['date'].dt.strftime('%Y-%m-%d')
output = output.rename(columns={'factor': 'factor_value'})

print(f"\n因子覆盖: {output['stock_code'].nunique()} 只股票, {output['date'].nunique()} 个交易日")
print(f"日期范围: {output['date'].min()} ~ {output['date'].max()}")
print(f"\n因子统计:")
print(output['factor_value'].describe())

output.to_csv("data/factor_coskewness_v1.csv", index=False)
print(f"\n✅ 输出: data/factor_coskewness_v1.csv ({len(output)} 行)")
