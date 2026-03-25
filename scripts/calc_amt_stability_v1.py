"""
成交额稳定性因子 (Amount Stability / 成交额变异系数取负)
==========================================================
逻辑：
  - CV(amount, 20d) = std(amount) / mean(amount)
  - 因子值 = -CV（负向：低CV=高因子值=成交额稳定=机构特征）
  - 成交额OLS中性化 + MAD winsorize + z-score

假说：
  成交额波动小的股票,日间交易行为更有序,
  可能反映机构投资者的分散下单策略(TWAP/VWAP算法交易).
  这类股票信息效率高,价格发现有序,后续收益更好.
  
Barra风格: Liquidity/Quality交叉
"""

import pandas as pd
import numpy as np
import warnings, sys
warnings.filterwarnings('ignore')

# ── 读数据 ──────────────────────────────────────────────
kline = pd.read_csv('data/csi1000_kline_raw.csv', parse_dates=['date'])
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)

print(f"原始行情: {len(kline)} 行, {kline['stock_code'].nunique()} 只股票")
print(f"日期范围: {kline['date'].min()} ~ {kline['date'].max()}")

# ── 计算20日成交额CV ────────────────────────────────────
# 滚动20日std和mean
kline['amt_mean_20'] = kline.groupby('stock_code')['amount'].transform(
    lambda x: x.rolling(20, min_periods=15).mean()
)
kline['amt_std_20'] = kline.groupby('stock_code')['amount'].transform(
    lambda x: x.rolling(20, min_periods=15).std()
)

# CV = std/mean, 取负
kline['amt_cv'] = kline['amt_std_20'] / (kline['amt_mean_20'] + 1e-10)
kline['factor_raw'] = -kline['amt_cv']  # 负向：低CV=高因子值

# 去掉NaN
df = kline.dropna(subset=['factor_raw', 'amount']).copy()
print(f"有效行: {len(df)}")

# ── 截面中性化 ──────────────────────────────────────────
# 用20日平均成交额的对数做中性化
df['log_amt_20'] = np.log(df['amt_mean_20'] + 1)

from sklearn.linear_model import LinearRegression

def neutralize_cross_section(group):
    """截面OLS中性化"""
    y = group['factor_raw'].values
    X = group['log_amt_20'].values.reshape(-1, 1)
    mask = np.isfinite(y) & np.isfinite(X.ravel())
    if mask.sum() < 30:
        return pd.Series(np.nan, index=group.index)
    reg = LinearRegression().fit(X[mask], y[mask])
    resid = np.full(len(y), np.nan)
    resid[mask] = y[mask] - reg.predict(X[mask])
    return pd.Series(resid, index=group.index)

print("开始截面中性化...")
df['factor_neutral'] = df.groupby('date').apply(neutralize_cross_section).reset_index(level=0, drop=True)

# ── MAD winsorize + z-score ──────────────────────────────
def mad_winsorize_zscore(group):
    vals = group['factor_neutral']
    median = vals.median()
    mad = (vals - median).abs().median()
    if mad < 1e-10:
        return pd.Series(0.0, index=group.index)
    upper = median + 5.2 * mad
    lower = median - 5.2 * mad
    clipped = vals.clip(lower, upper)
    mean = clipped.mean()
    std = clipped.std()
    if std < 1e-10:
        return pd.Series(0.0, index=group.index)
    return (clipped - mean) / std

df['factor'] = df.groupby('date').apply(mad_winsorize_zscore).reset_index(level=0, drop=True)

# ── 输出 ────────────────────────────────────────────────
output = df[['date', 'stock_code', 'factor']].dropna()
output = output.rename(columns={'factor': 'factor_value'})
output.to_csv('data/factor_amt_stability_v1.csv', index=False)

print(f"\n输出: {len(output)} 行")
print(f"日期范围: {output['date'].min()} ~ {output['date'].max()}")
print(f"股票数: {output['stock_code'].nunique()}")
print(f"\n因子分布:")
print(output['factor_value'].describe())

# 快速IC检查
ret = pd.read_csv('data/csi1000_returns.csv', parse_dates=['date'])
merged = output.merge(ret, on=['date', 'stock_code'])
if 'return_5d' in merged.columns:
    ic = merged.groupby('date').apply(lambda g: g['factor_value'].corr(g['return_5d']))
    print(f"\n快速IC检查 (5d forward):")
    print(f"  IC均值: {ic.mean():.4f}")
    print(f"  IC标准差: {ic.std():.4f}")
    print(f"  IC_t: {ic.mean() / ic.std() * np.sqrt(len(ic)):.2f}")
    print(f"  正比例: {(ic > 0).mean():.3f}")
