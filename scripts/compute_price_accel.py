#!/usr/bin/env python3
"""
价格加速度因子 (Price Acceleration / Momentum Change)
来源思路: 动量二阶效应
  - Grinblatt & Moskowitz (2004) "Predicting stock price movements from past returns"
  - 类似 Chen & Lu (2019) "中国A股二阶动量效应"
公式: 多种变体
  v1: accel = cum_ret(1-5d) - cum_ret(6-10d)  近5日收益 - 前5日收益
  v2: accel = cum_ret(1-10d) - cum_ret(11-20d) 近10日 vs 前10日
  v3: trend_break = (MA5 - MA10) - (MA10 - MA20) 均线加速度
  
方向: 待验证
中性化: 成交额OLS中性化 + MAD winsorize + z-score
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

print("=== 价格加速度因子 (Price Acceleration) ===\n")

# 读取数据
kline = pd.read_csv("data/csi1000_kline_raw.csv")
kline['date'] = pd.to_datetime(kline['date'])
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)

kline['returns'] = kline.groupby('stock_code')['close'].pct_change()

print(f"数据: {kline['stock_code'].nunique()} 只股票, {kline['date'].nunique()} 个交易日")

# 计算多种价格加速度变体
g = kline.groupby('stock_code')

# 累积收益
kline['cum_ret_5'] = g['returns'].transform(lambda x: x.rolling(5, min_periods=3).sum())
kline['cum_ret_5_lag5'] = g['cum_ret_5'].shift(5)
kline['cum_ret_10'] = g['returns'].transform(lambda x: x.rolling(10, min_periods=7).sum())
kline['cum_ret_10_lag10'] = g['cum_ret_10'].shift(10)

# MA
kline['ma5'] = g['close'].transform(lambda x: x.rolling(5, min_periods=3).mean())
kline['ma10'] = g['close'].transform(lambda x: x.rolling(10, min_periods=7).mean())
kline['ma20'] = g['close'].transform(lambda x: x.rolling(20, min_periods=14).mean())

# 变体
# v1: 近5日收益 - 前5日收益
kline['accel_v1'] = kline['cum_ret_5'] - kline['cum_ret_5_lag5']

# v2: 近10日 - 前10日  
kline['accel_v2'] = kline['cum_ret_10'] - kline['cum_ret_10_lag10']

# v3: 均线加速度 (MA5-MA10)/MA10 - (MA10-MA20)/MA20
kline['accel_v3'] = (kline['ma5'] - kline['ma10']) / kline['ma10'] - \
                     (kline['ma10'] - kline['ma20']) / kline['ma20']

# 20日平均成交额(对数)
kline['log_amount_20d'] = g['amount'].transform(
    lambda x: np.log(x.rolling(20, min_periods=10).mean().replace(0, np.nan))
)

# 中性化函数
def neutralize_and_standardize(df_day, col):
    mask = df_day[[col, 'log_amount_20d']].notna().all(axis=1)
    sub = df_day[mask].copy()
    
    if len(sub) < 50:
        return pd.Series(np.nan, index=df_day.index)
    
    y = sub[col].values
    X = sub[['log_amount_20d']].values
    
    # MAD winsorize
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
    
    result = pd.Series(np.nan, index=df_day.index)
    result.iloc[mask.values] = z
    return result

# 对每个变体做中性化，取反（反转方向：加速下跌 → 高因子值 → 反弹）
for v_name in ['accel_v1', 'accel_v2', 'accel_v3']:
    print(f"\n处理 {v_name}...")
    # 先正向(加速 → 做多)
    factor_pos = kline.groupby('date', group_keys=False).apply(
        lambda d: neutralize_and_standardize(d, v_name)
    )
    kline[f'{v_name}_pos'] = factor_pos.values if hasattr(factor_pos, 'values') else factor_pos
    
    # 再反向(减速/反转 → 做多)
    kline[f'{v_name}_neg_raw'] = -kline[v_name]
    factor_neg = kline.groupby('date', group_keys=False).apply(
        lambda d: neutralize_and_standardize(d, f'{v_name}_neg_raw')
    )
    kline[f'{v_name}_neg'] = factor_neg.values if hasattr(factor_neg, 'values') else factor_neg

# 输出所有变体
for suffix in ['v1_pos', 'v1_neg', 'v2_pos', 'v2_neg', 'v3_pos', 'v3_neg']:
    col = f'accel_{suffix}'
    out = kline[['date', 'stock_code', col]].dropna(subset=[col]).copy()
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    out = out.rename(columns={col: 'factor_value'})
    fname = f"data/factor_price_accel_{suffix}.csv"
    out.to_csv(fname, index=False)
    print(f"✅ {fname}: {len(out)} 行, {out['stock_code'].nunique()} 股票")
    print(f"   统计: mean={out['factor_value'].mean():.4f}, std={out['factor_value'].std():.4f}")

print("\n✅ 全部变体计算完成!")
