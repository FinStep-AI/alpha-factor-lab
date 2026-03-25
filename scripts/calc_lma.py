"""
大幅涨跌不对称性因子 (Large Move Asymmetry)
============================================
构造:
  count(ret > 2%, 20d) - count(ret < -2%, 20d)
  正向使用: 大涨天数多 = 正面信息多 = 动量延续

也测试:
  v1: 阈值2%, 正向(动量)
  v2: 阈值3%, 正向
  v3: 阈值2%, 反向(反转: 大跌多→反弹)
  v4: 极端涨跌比: count(ret>3%) / (count(ret>3%) + count(ret<-3%))
  
成交额OLS中性化 + MAD + z-score
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression

kline = pd.read_csv('data/csi1000_kline_raw.csv', parse_dates=['date'])
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)

# 计算日收益率
kline['ret'] = kline.groupby('stock_code')['close'].pct_change()

# 标记大涨/大跌
for thresh in [0.02, 0.03]:
    kline[f'big_up_{int(thresh*100)}'] = (kline['ret'] > thresh).astype(float)
    kline[f'big_dn_{int(thresh*100)}'] = (kline['ret'] < -thresh).astype(float)

# 20日滚动计数
for col in ['big_up_2', 'big_dn_2', 'big_up_3', 'big_dn_3']:
    kline[f'{col}_20'] = kline.groupby('stock_code')[col].transform(
        lambda x: x.rolling(20, min_periods=15).sum())

# 成交额中性化用
kline['log_amt_20'] = np.log(
    kline.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=15).mean()) + 1)

# ── 构造变体 ──────────────────────────────────────────
# v1: 2%阈值, 差值
kline['v1_raw'] = kline['big_up_2_20'] - kline['big_dn_2_20']

# v2: 3%阈值
kline['v2_raw'] = kline['big_up_3_20'] - kline['big_dn_3_20']

# v3: 反向(大跌多→高值)
kline['v3_raw'] = -(kline['big_up_2_20'] - kline['big_dn_2_20'])

# v4: 极端涨跌比 count(>3%) / total_extreme
total_extreme = kline['big_up_3_20'] + kline['big_dn_3_20']
kline['v4_raw'] = kline['big_up_3_20'] / (total_extreme + 1e-10)
# 无极端天=0.5(中性), 有的按比例
kline.loc[total_extreme < 0.5, 'v4_raw'] = 0.5  # 无极端天设为中性

# v5: 上涨日占比(不限阈值) - 之前down_day_ratio_v1(IC=0, mono=0.8)失败了
# 但上涨日占比和大幅涨跌比是不同的，试试

def neutralize_and_zscore(df_in, col):
    df = df_in.dropna(subset=[col, 'log_amt_20']).copy()
    
    def neutralize(group):
        y = group[col].values
        X = group['log_amt_20'].values.reshape(-1, 1)
        mask = np.isfinite(y) & np.isfinite(X.ravel())
        if mask.sum() < 30:
            return pd.Series(np.nan, index=group.index)
        reg = LinearRegression().fit(X[mask], y[mask])
        resid = np.full(len(y), np.nan)
        resid[mask] = y[mask] - reg.predict(X[mask])
        return pd.Series(resid, index=group.index)
    
    df['neutral'] = df.groupby('date').apply(neutralize).reset_index(level=0, drop=True)
    
    def mad_zscore(group):
        vals = group['neutral']
        median = vals.median()
        mad = (vals - median).abs().median()
        if mad < 1e-10:
            return pd.Series(0.0, index=group.index)
        upper = median + 5.2 * mad
        lower = median - 5.2 * mad
        clipped = vals.clip(lower, upper)
        m, s = clipped.mean(), clipped.std()
        if s < 1e-10:
            return pd.Series(0.0, index=group.index)
        return (clipped - m) / s
    
    df['factor_value'] = df.groupby('date').apply(mad_zscore).reset_index(level=0, drop=True)
    return df[['date', 'stock_code', 'factor_value']].dropna()

for name, col in [('v1', 'v1_raw'), ('v2', 'v2_raw'), ('v3', 'v3_raw'), ('v4', 'v4_raw')]:
    print(f"处理 {name}...")
    out = neutralize_and_zscore(kline, col)
    out.to_csv(f'data/factor_lma_{name}.csv', index=False)
    print(f"  {len(out)} 行")

print("完成!")
