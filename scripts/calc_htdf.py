"""
因子: 高换手日频率 (High Turnover Day Frequency, HTDF)
构造: 过去20日中换手率 > 1.5倍20日均值的天数占比
正向: 做多频繁出现高换手的股票（关注度脉冲频繁→信息活跃→价格发现充分）
中性化: 成交额OLS中性化 + MAD winsorize + z-score
Barra风格: Liquidity / Sentiment
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("加载数据...")
df = pd.read_csv('data/csi1000_kline_raw.csv', parse_dates=['date'])
df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
df = df[df['date'] <= '2026-03-19']

WINDOW = 20
THRESHOLD = 1.5  # 换手率超过均值的倍数

print(f"计算 High Turnover Day Frequency (窗口={WINDOW}d, 阈值={THRESHOLD}x)...")

results = []
for code, grp in df.groupby('stock_code'):
    grp = grp.sort_values('date').reset_index(drop=True)
    to = grp['turnover'].values
    amount = grp['amount'].values
    
    factor_vals = np.full(len(grp), np.nan)
    log_amt_20d = np.full(len(grp), np.nan)
    
    for i in range(WINDOW - 1, len(grp)):
        window_to = to[i - WINDOW + 1:i + 1]
        window_amt = amount[i - WINDOW + 1:i + 1]
        
        valid = np.isfinite(window_to) & (window_to > 0)
        if valid.sum() < 15:
            continue
        
        wto = window_to[valid]
        mean_to = np.mean(wto)
        
        if mean_to <= 0:
            continue
        
        # 高换手日占比
        high_days = np.sum(wto > THRESHOLD * mean_to)
        factor_vals[i] = high_days / len(wto)
        
        # log(20日均成交额)
        wamt = window_amt[np.isfinite(window_amt) & (window_amt > 0)]
        if len(wamt) > 0:
            log_amt_20d[i] = np.log(np.mean(wamt))
    
    grp_result = grp[['date', 'stock_code']].copy()
    grp_result['raw_factor'] = factor_vals
    grp_result['log_amount_20d'] = log_amt_20d
    results.append(grp_result)

factor_df = pd.concat(results, ignore_index=True)
factor_df = factor_df.dropna(subset=['raw_factor', 'log_amount_20d'])

print(f"原始因子: {len(factor_df)} 行, mean={factor_df['raw_factor'].mean():.4f}, std={factor_df['raw_factor'].std():.4f}")

# === 中性化 ===
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

output = factor_df[['date', 'stock_code', 'factor']].sort_values(['date', 'stock_code'])
print(f"\n最终因子: {len(output)} 行, {output['stock_code'].nunique()} 只股票")
print(f"日期范围: {output['date'].min()} ~ {output['date'].max()}")

output.to_csv('data/factor_htdf_v1.csv', index=False)
print("因子已保存到 data/factor_htdf_v1.csv")
