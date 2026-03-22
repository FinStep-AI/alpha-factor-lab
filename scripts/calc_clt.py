"""
因子: 收盘位置趋势 (Close Location Trend, CLT)
构造: CLV = (2*close - high - low) / (high - low + eps)
      对20日CLV做线性回归，取斜率的t-stat
正向: 高CLT=收盘持续向日高靠拢=买方力量趋势性增强
中性化: 成交额OLS中性化 + MAD winsorize + z-score
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("加载数据...")
df = pd.read_csv('data/csi1000_kline_raw.csv', parse_dates=['date'])
df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
df = df[df['date'] <= '2026-03-19']

# CLV
df['clv'] = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'] + 1e-8)
df['clv'] = df['clv'].clip(-1, 1)

WINDOW = 20

print(f"计算 Close Location Trend (窗口={WINDOW}d)...")

results = []
for code, grp in df.groupby('stock_code'):
    grp = grp.sort_values('date').reset_index(drop=True)
    clv = grp['clv'].values
    amount = grp['amount'].values
    
    factor_vals = np.full(len(grp), np.nan)
    log_amt_20d = np.full(len(grp), np.nan)
    
    for i in range(WINDOW - 1, len(grp)):
        window_clv = clv[i - WINDOW + 1:i + 1]
        window_amt = amount[i - WINDOW + 1:i + 1]
        
        valid = np.isfinite(window_clv)
        if valid.sum() < 15:
            continue
        
        wclv = window_clv[valid]
        x = np.arange(len(wclv))
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, wclv)
            if std_err > 0 and not np.isnan(std_err):
                factor_vals[i] = slope / std_err  # t-stat of slope
        except:
            pass
        
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

# 中性化
print("成交额OLS中性化...")

def neutralize_cross_section(group):
    y = group['raw_factor'].values
    x = group['log_amount_20d'].values
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 30:
        group['factor'] = np.nan
        return group
    x_clean, y_clean = x[mask], y[mask]
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
    mean, std = np.mean(vals), np.std(vals)
    group['factor'] = (vals - mean) / std if std > 1e-10 else 0.0
    return group

factor_df = factor_df.groupby('date', group_keys=False).apply(mad_winsorize_zscore)

output = factor_df[['date', 'stock_code', 'factor']].sort_values(['date', 'stock_code'])
print(f"\n最终因子: {len(output)} 行, {output['stock_code'].nunique()} 只股票")
output.to_csv('data/factor_clt_v1.csv', index=False)

# 反向
output_rev = output.copy()
output_rev['factor'] = -output_rev['factor']
output_rev.to_csv('data/factor_clt_rev_v1.csv', index=False)
print("因子已保存: factor_clt_v1.csv (正向), factor_clt_rev_v1.csv (反向)")
