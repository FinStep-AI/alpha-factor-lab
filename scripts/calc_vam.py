"""
因子: 波动率调整动量 (Volatility-Adjusted Momentum, VAM)
构造: 20日累计收益率 / 20日已实现波动率 (= 时序Sharpe形式)
反向使用: 负因子值=低VAM=超卖/趋势反转候选 (短期反转)
正向使用: 高VAM=高效上涨 (动量延续)
先测试两个方向
中性化: 成交额OLS中性化 + MAD winsorize + z-score
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
df['ret'] = df.groupby('stock_code')['close'].pct_change()

WINDOW = 20

print(f"计算 Volatility-Adjusted Momentum (窗口={WINDOW}d)...")

results = []
for code, grp in df.groupby('stock_code'):
    grp = grp.sort_values('date').reset_index(drop=True)
    rets = grp['ret'].values
    amount = grp['amount'].values
    
    factor_vals = np.full(len(grp), np.nan)
    log_amt_20d = np.full(len(grp), np.nan)
    
    for i in range(WINDOW, len(grp)):
        window_ret = rets[i - WINDOW + 1:i + 1]
        window_amt = amount[i - WINDOW + 1:i + 1]
        
        valid = np.isfinite(window_ret)
        if valid.sum() < 15:
            continue
        
        wr = window_ret[valid]
        cum_ret = np.sum(wr)  # 累计收益（简单求和近似）
        vol = np.std(wr, ddof=1)
        
        if vol < 1e-8:
            continue
        
        factor_vals[i] = cum_ret / vol  # Sharpe-like ratio
        
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

# 保存正向版本
output = factor_df[['date', 'stock_code', 'factor']].sort_values(['date', 'stock_code'])
print(f"\n最终因子: {len(output)} 行, {output['stock_code'].nunique()} 只股票")
print(f"日期范围: {output['date'].min()} ~ {output['date'].max()}")
output.to_csv('data/factor_vam_v1.csv', index=False)

# 保存反转版本
output_rev = output.copy()
output_rev['factor'] = -output_rev['factor']
output_rev.to_csv('data/factor_vam_rev_v1.csv', index=False)

print("因子已保存: factor_vam_v1.csv (正向/动量), factor_vam_rev_v1.csv (反向/反转)")
