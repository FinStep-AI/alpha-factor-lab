"""
换手率冷却因子优化 (Turnover Cooling)
======================================
v1_neg发现: IC=0.019(t=2.93), mono=1.0完美, 但IC差0.001不达标

优化方向:
1. log比率代替原始比率: log(MA5/MA20) → 减弱极端值
2. 窗口优化: 3/20, 5/20, 5/40, 10/20
3. 换手率取对数再算比率
4. 加入市值(log_mktcap)做双变量中性化
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression

kline = pd.read_csv('data/csi1000_kline_raw.csv', parse_dates=['date'])
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)

# 各种换手率均值
for w in [3, 5, 10, 20, 40]:
    kline[f'to_ma{w}'] = kline.groupby('stock_code')['turnover'].transform(
        lambda x: x.rolling(w, min_periods=max(w//2, 2)).mean())

# 市值近似 = close * volume / turnover (流通股数≈volume/turnover*100)
kline['mktcap_approx'] = kline['close'] * kline['volume'] / (kline['turnover'] / 100 + 1e-10)
kline['log_mktcap'] = np.log(kline.groupby('stock_code')['mktcap_approx'].transform(
    lambda x: x.rolling(20, min_periods=15).mean()) + 1)

# 20日成交额均值
kline['log_amt_20'] = np.log(
    kline.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=15).mean()) + 1)

def neutralize_and_zscore(df_in, col, neutralize_cols=['log_amt_20']):
    """多变量中性化 + MAD + z-score"""
    df = df_in.dropna(subset=[col] + neutralize_cols).copy()
    
    def neutralize(group):
        y = group[col].values
        X = group[neutralize_cols].values
        mask = np.all(np.isfinite(np.column_stack([y.reshape(-1,1), X])), axis=1)
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

# ── 变体矩阵 ──────────────────────────────────────────
variants = {}

# A: 原始比率, 各种窗口 (取负=冷却)
for short, long in [(3,20), (5,20), (5,40), (10,20)]:
    name = f'raw_{short}_{long}'
    kline[name] = -(kline[f'to_ma{short}'] / (kline[f'to_ma{long}'] + 1e-10) - 1)
    variants[name] = name

# B: log比率
for short, long in [(3,20), (5,20), (5,40)]:
    name = f'log_{short}_{long}'
    kline[name] = -np.log(kline[f'to_ma{short}'] / (kline[f'to_ma{long}'] + 1e-10) + 1e-10)
    variants[name] = name

# C: 双变量中性化(成交额+市值)
kline['best_raw'] = -(kline['to_ma5'] / (kline['to_ma20'] + 1e-10) - 1)

print(f"共 {len(variants)} 个变体 + 1个双中性化版本")

# 批量计算
results = {}
for vname, col in variants.items():
    out = neutralize_and_zscore(kline, col)
    out.to_csv(f'data/factor_to_cool_{vname}.csv', index=False)
    results[vname] = out
    print(f"  {vname}: {len(out)} 行")

# 双中性化
out_dual = neutralize_and_zscore(kline, 'best_raw', ['log_amt_20', 'log_mktcap'])
out_dual.to_csv('data/factor_to_cool_dual.csv', index=False)
print(f"  dual_5_20: {len(out_dual)} 行")

print("\n计算完成!")
