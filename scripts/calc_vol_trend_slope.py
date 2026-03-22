"""
因子: 成交量趋势斜率 (Volume Trend Slope, VTS)
构造: 对20日log(turnover)做OLS线性回归，取标准化斜率(t-stat)
正向: 做多放量趋势股票（关注度上升→动量延续）
中性化: 成交额OLS中性化 + MAD winsorize + z-score
Barra风格: Momentum / Sentiment
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# === 加载数据 ===
print("加载数据...")
df = pd.read_csv('data/csi1000_kline_raw.csv', parse_dates=['date'])
df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)

# 过滤掉异常数据
df = df[df['date'] <= '2026-03-19']

# === 计算因子 ===
WINDOW = 20  # 回看窗口

print(f"计算 Volume Trend Slope (窗口={WINDOW}d)...")

def calc_slope_tstat(series):
    """对series做线性回归，返回斜率的t统计量"""
    y = series.values
    if len(y) < WINDOW or np.any(np.isnan(y)) or np.any(np.isinf(y)):
        return np.nan
    x = np.arange(len(y))
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        if std_err == 0 or np.isnan(std_err):
            return np.nan
        return slope / std_err  # t-stat of slope
    except:
        return np.nan

# 用log(turnover)
df['log_turnover'] = np.log(df['turnover'].clip(lower=0.001))

# 也计算log(amount)用于中性化
df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
    lambda x: np.log(x.rolling(20).mean().clip(lower=1))
)

# 对每只股票滚动计算
results = []
for code, grp in df.groupby('stock_code'):
    grp = grp.sort_values('date')
    log_to = grp['log_turnover'].values
    
    slopes = np.full(len(grp), np.nan)
    for i in range(WINDOW - 1, len(grp)):
        y = log_to[i - WINDOW + 1:i + 1]
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            continue
        x = np.arange(WINDOW)
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            if std_err > 0 and not np.isnan(std_err):
                slopes[i] = slope / std_err  # t-stat
        except:
            pass
    
    grp_result = grp[['date', 'stock_code']].copy()
    grp_result['raw_factor'] = slopes
    grp_result['log_amount_20d'] = grp['log_amount_20d'].values
    results.append(grp_result)

factor_df = pd.concat(results, ignore_index=True)
factor_df = factor_df.dropna(subset=['raw_factor', 'log_amount_20d'])

print(f"原始因子统计: mean={factor_df['raw_factor'].mean():.4f}, std={factor_df['raw_factor'].std():.4f}")
print(f"记录数: {len(factor_df)}")

# === 成交额OLS中性化 ===
print("成交额OLS中性化...")

def neutralize_cross_section(group):
    y = group['raw_factor'].values
    x = group['log_amount_20d'].values
    
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 30:
        group['factor'] = np.nan
        return group
    
    # OLS回归
    x_clean = x[mask]
    y_clean = y[mask]
    x_mat = np.column_stack([np.ones(len(x_clean)), x_clean])
    
    try:
        beta = np.linalg.lstsq(x_mat, y_clean, rcond=None)[0]
        residuals = np.full(len(y), np.nan)
        x_full = np.column_stack([np.ones(mask.sum()), x_clean])
        residuals[mask] = y_clean - x_full @ beta
        group['factor'] = residuals
    except:
        group['factor'] = np.nan
    
    return group

factor_df = factor_df.groupby('date', group_keys=False).apply(neutralize_cross_section)
factor_df = factor_df.dropna(subset=['factor'])

# === MAD Winsorize ===
print("MAD winsorize + z-score...")

def mad_winsorize_zscore(group, n_mad=5):
    vals = group['factor'].values
    median = np.median(vals)
    mad = np.median(np.abs(vals - median))
    if mad < 1e-10:
        group['factor'] = 0.0
        return group
    
    upper = median + n_mad * 1.4826 * mad
    lower = median - n_mad * 1.4826 * mad
    vals = np.clip(vals, lower, upper)
    
    # z-score
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
output.columns = ['date', 'stock_code', 'factor']
output = output.sort_values(['date', 'stock_code'])

print(f"\n最终因子: {len(output)} 行, {output['stock_code'].nunique()} 只股票")
print(f"日期范围: {output['date'].min()} ~ {output['date'].max()}")
print(f"因子分布: mean={output['factor'].mean():.4f}, std={output['factor'].std():.4f}")
print(f"  25%={output['factor'].quantile(0.25):.4f}, 50%={output['factor'].quantile(0.5):.4f}, 75%={output['factor'].quantile(0.75):.4f}")

output.to_csv('data/factor_vol_trend_slope_v1.csv', index=False)
print("\n因子已保存到 data/factor_vol_trend_slope_v1.csv")
