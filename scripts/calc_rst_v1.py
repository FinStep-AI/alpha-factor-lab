"""
收益率加权换手率因子 (Return-Signed Turnover, RST)
====================================================
逻辑:
  - RST = sum(sign(ret) * turnover, 20d)
  - 正向: 上涨日换手多+下跌日换手少 = 买方主导
  - 成交额OLS中性化 + MAD winsorize + z-score

本质上是OBV的换手率改良版:
  OBV用volume * sign(ret), 我们用turnover * sign(ret)
  turnover已包含流通股信息, 更适合截面比较

也测试变体:
  v1a: sign(ret) * turnover (方向×换手)
  v1b: ret * turnover (幅度×换手, 成交额加权动量变体)  
  v1c: sign(ret) * log(turnover) (对数换手, 减弱极端值影响)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression

kline = pd.read_csv('data/csi1000_kline_raw.csv', parse_dates=['date'])
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)

print(f"行情: {len(kline)} 行, {kline['stock_code'].nunique()} 只股票")

# 计算日收益率
kline['ret'] = kline['pct_change'] / 100  # pct_change是百分比形式
# 但pct_change列有些是NaN（首日）, 直接用close计算
kline['ret'] = kline.groupby('stock_code')['close'].pct_change()

# sign(ret) * turnover
kline['signed_turnover'] = np.sign(kline['ret']) * kline['turnover']

# 20日累计
kline['rst_20'] = kline.groupby('stock_code')['signed_turnover'].transform(
    lambda x: x.rolling(20, min_periods=15).sum()
)

# 20日平均成交额 (用于中性化)
kline['log_amt_20'] = np.log(
    kline.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    ) + 1
)

kline['factor_raw'] = kline['rst_20']

df = kline.dropna(subset=['factor_raw', 'log_amt_20']).copy()
print(f"有效行: {len(df)}")

# ── 截面中性化 ──────────────────────────────────────────
def neutralize_cross_section(group):
    y = group['factor_raw'].values
    X = group['log_amt_20'].values.reshape(-1, 1)
    mask = np.isfinite(y) & np.isfinite(X.ravel())
    if mask.sum() < 30:
        return pd.Series(np.nan, index=group.index)
    reg = LinearRegression().fit(X[mask], y[mask])
    resid = np.full(len(y), np.nan)
    resid[mask] = y[mask] - reg.predict(X[mask])
    return pd.Series(resid, index=group.index)

print("截面中性化...")
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
output.to_csv('data/factor_rst_v1.csv', index=False)

print(f"\n输出: {len(output)} 行")
print(f"日期: {output['date'].min()} ~ {output['date'].max()}")
print(output['factor_value'].describe())
