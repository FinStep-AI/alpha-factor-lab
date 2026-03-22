"""
因子: 换手率条件收益差 (Turnover-Conditional Return Spread, TCRS)
构造: 过去20日内，高换手日(高于中位数)的平均收益 - 低换手日的平均收益
正向: 高值=高换手日涨多/跌少=资金推动型，做多
中性化: 成交额OLS中性化 + MAD winsorize + z-score
逻辑: 当放量日收益好于缩量日，说明主力资金在推升价格；反之说明放量卖出
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("加载数据...")
df = pd.read_csv('data/csi1000_kline_raw.csv', parse_dates=['date'])
df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
df = df[df['date'] <= '2026-03-19']

# 计算日收益率
df['ret'] = df['pct_change'] / 100.0
# 如果pct_change为空，用close/close_prev-1
df['ret'] = df.groupby('stock_code').apply(
    lambda g: g['close'].pct_change()
).reset_index(level=0, drop=True)

WINDOW = 20

print(f"计算 Turnover-Conditional Return Spread (窗口={WINDOW}d)...")

results = []
for code, grp in df.groupby('stock_code'):
    grp = grp.sort_values('date').reset_index(drop=True)
    rets = grp['ret'].values
    turnover = grp['turnover'].values
    amount = grp['amount'].values
    
    factor_vals = np.full(len(grp), np.nan)
    log_amt_20d = np.full(len(grp), np.nan)
    
    for i in range(WINDOW - 1, len(grp)):
        window_ret = rets[i - WINDOW + 1:i + 1]
        window_to = turnover[i - WINDOW + 1:i + 1]
        window_amt = amount[i - WINDOW + 1:i + 1]
        
        # 跳过有NaN的窗口
        mask = np.isfinite(window_ret) & np.isfinite(window_to) & (window_to > 0)
        if mask.sum() < 10:
            continue
        
        w_ret = window_ret[mask]
        w_to = window_to[mask]
        
        # 以换手率中位数分高低
        median_to = np.median(w_to)
        high_mask = w_to >= median_to
        low_mask = w_to < median_to
        
        if high_mask.sum() < 3 or low_mask.sum() < 3:
            continue
        
        high_ret_mean = np.mean(w_ret[high_mask])
        low_ret_mean = np.mean(w_ret[low_mask])
        
        factor_vals[i] = high_ret_mean - low_ret_mean
        
        # log(20日均成交额)用于中性化
        amt_valid = window_amt[np.isfinite(window_amt) & (window_amt > 0)]
        if len(amt_valid) > 0:
            log_amt_20d[i] = np.log(np.mean(amt_valid))
    
    grp_result = grp[['date', 'stock_code']].copy()
    grp_result['raw_factor'] = factor_vals
    grp_result['log_amount_20d'] = log_amt_20d
    results.append(grp_result)

factor_df = pd.concat(results, ignore_index=True)
factor_df = factor_df.dropna(subset=['raw_factor', 'log_amount_20d'])

print(f"原始因子: {len(factor_df)} 行, mean={factor_df['raw_factor'].mean():.6f}, std={factor_df['raw_factor'].std():.6f}")

# === 成交额OLS中性化 ===
print("成交额OLS中性化...")

def neutralize_cross_section(group):
    y = group['raw_factor'].values
    x = group['log_amount_20d'].values
    
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 30:
        group['factor'] = np.nan
        return group
    
    x_clean = x[mask]
    y_clean = y[mask]
    x_mat = np.column_stack([np.ones(len(x_clean)), x_clean])
    
    try:
        beta = np.linalg.lstsq(x_mat, y_clean, rcond=None)[0]
        residuals = np.full(len(y), np.nan)
        residuals[mask] = y_clean - x_mat @ beta
        group['factor'] = residuals
    except:
        group['factor'] = np.nan
    
    return group

factor_df = factor_df.groupby('date', group_keys=False).apply(neutralize_cross_section)
factor_df = factor_df.dropna(subset=['factor'])

# === MAD Winsorize + z-score ===
print("MAD winsorize + z-score...")

def mad_winsorize_zscore(group, n_mad=5):
    vals = group['factor'].values.copy()
    median = np.median(vals)
    mad = np.median(np.abs(vals - median))
    if mad < 1e-10:
        group['factor'] = 0.0
        return group
    
    upper = median + n_mad * 1.4826 * mad
    lower = median - n_mad * 1.4826 * mad
    vals = np.clip(vals, lower, upper)
    
    mean = np.mean(vals)
    std = np.std(vals)
    if std < 1e-10:
        group['factor'] = 0.0
    else:
        group['factor'] = (vals - mean) / std
    
    return group

factor_df = factor_df.groupby('date', group_keys=False).apply(mad_winsorize_zscore)

# === 输出 ===
output = factor_df[['date', 'stock_code', 'factor']].copy()
output = output.sort_values(['date', 'stock_code'])

print(f"\n最终因子: {len(output)} 行, {output['stock_code'].nunique()} 只股票")
print(f"日期范围: {output['date'].min()} ~ {output['date'].max()}")
print(f"因子分布: mean={output['factor'].mean():.4f}, std={output['factor'].std():.4f}")

output.to_csv('data/factor_tcrs_v1.csv', index=False)
print("因子已保存到 data/factor_tcrs_v1.csv")
