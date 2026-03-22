#!/usr/bin/env python3
"""
Betting Against Beta (BAB) 因子
来源: Frazzini & Pedersen (2014) "Betting Against Beta", JFE
公式: -beta_60d (做多低Beta, 做空高Beta)
变体:
  v1: 标准60日Beta取负
  v2: 波动率调整Beta = beta / idio_vol (信噪比)
  v3: 收缩Beta = 0.6*beta + 0.4*1.0 (Vasicek shrinkage toward cross-sectional mean)
中性化: 成交额OLS中性化 + MAD winsorize + z-score
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

print("=== BAB因子 (Betting Against Beta) ===")
print("来源: Frazzini & Pedersen (2014) JFE\n")

# 读取数据
kline = pd.read_csv("data/csi1000_kline_raw.csv")
kline['date'] = pd.to_datetime(kline['date'])
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
kline['returns'] = kline.groupby('stock_code')['close'].pct_change()

# 市场收益
mkt = kline.groupby('date')['returns'].mean().rename('mkt_ret')
kline = kline.merge(mkt, on='date', how='left')

print(f"数据: {kline['stock_code'].nunique()} 只股票, {kline['date'].nunique()} 个交易日")

# 60日滚动Beta
WINDOW = 60
MIN_P = 40

def rolling_beta_stats(group):
    """计算滚动beta, idio_vol"""
    ri = group['returns'].values
    rm = group['mkt_ret'].values
    n = len(ri)
    
    beta_arr = np.full(n, np.nan)
    ivol_arr = np.full(n, np.nan)
    
    for t in range(MIN_P - 1, n):
        start = max(0, t - WINDOW + 1)
        ri_w = ri[start:t+1]
        rm_w = rm[start:t+1]
        
        mask = ~(np.isnan(ri_w) | np.isnan(rm_w))
        if mask.sum() < MIN_P:
            continue
        
        ri_w = ri_w[mask]
        rm_w = rm_w[mask]
        
        cov = np.cov(ri_w, rm_w)
        if cov[1,1] > 1e-12:
            beta = cov[0,1] / cov[1,1]
            resid = ri_w - (np.mean(ri_w) - beta * np.mean(rm_w)) - beta * rm_w
            ivol = resid.std(ddof=1)
        else:
            continue
        
        beta_arr[t] = beta
        ivol_arr[t] = ivol
    
    return pd.DataFrame({'beta': beta_arr, 'ivol': ivol_arr}, index=group.index)

print("计算60日滚动Beta...")
stats = kline.groupby('stock_code', group_keys=False).apply(rolling_beta_stats)
kline['beta'] = stats['beta']
kline['ivol'] = stats['ivol']

# 变体构造
# v1: -beta (做多低beta)
kline['bab_v1'] = -kline['beta']

# v2: -beta / ivol (低beta高信噪比)
kline['bab_v2'] = -kline['beta'] / kline['ivol'].replace(0, np.nan)

# v3: 收缩beta取负
kline['bab_v3'] = -(0.6 * kline['beta'] + 0.4 * 1.0)

# 20日平均成交额(对数)
kline['log_amount_20d'] = kline.groupby('stock_code')['amount'].transform(
    lambda x: np.log(x.rolling(20, min_periods=10).mean().replace(0, np.nan))
)

def neutralize_and_standardize(df_day, col):
    mask = df_day[[col, 'log_amount_20d']].notna().all(axis=1)
    sub = df_day[mask].copy()
    if len(sub) < 50:
        return pd.Series(np.nan, index=df_day.index)
    y = sub[col].values
    X = sub[['log_amount_20d']].values
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad > 0:
        y = np.clip(y, med - 3 * 1.4826 * mad, med + 3 * 1.4826 * mad)
    lr = LinearRegression()
    lr.fit(X, y)
    residuals = y - lr.predict(X)
    mu, sigma = residuals.mean(), residuals.std()
    if sigma > 0:
        z = (residuals - mu) / sigma
    else:
        z = np.zeros_like(residuals)
    z = np.clip(z, -3, 3)
    result = pd.Series(np.nan, index=df_day.index)
    result.iloc[mask.values] = z
    return result

for v in ['bab_v1', 'bab_v2', 'bab_v3']:
    print(f"\n处理 {v}...")
    factor = kline.groupby('date', group_keys=False).apply(
        lambda d: neutralize_and_standardize(d, v)
    )
    kline[f'{v}_final'] = factor.values if hasattr(factor, 'values') else factor
    
    out = kline[['date', 'stock_code', f'{v}_final']].dropna(subset=[f'{v}_final']).copy()
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    out = out.rename(columns={f'{v}_final': 'factor_value'})
    fname = f"data/factor_{v}.csv"
    out.to_csv(fname, index=False)
    print(f"✅ {fname}: {len(out)} 行, {out['stock_code'].nunique()} 股票")

# 也输出正向beta (做多高beta, 高波动溢价)
for v_neg in ['bab_v1']:
    kline['beta_pos'] = kline['beta']  # 不取负
    print(f"\n处理 beta_pos (做多高Beta)...")
    factor = kline.groupby('date', group_keys=False).apply(
        lambda d: neutralize_and_standardize(d, 'beta_pos')
    )
    kline['beta_pos_final'] = factor.values if hasattr(factor, 'values') else factor
    out = kline[['date', 'stock_code', 'beta_pos_final']].dropna(subset=['beta_pos_final']).copy()
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    out = out.rename(columns={'beta_pos_final': 'factor_value'})
    out.to_csv("data/factor_beta_pos.csv", index=False)
    print(f"✅ data/factor_beta_pos.csv: {len(out)} 行")

print("\n✅ 全部完成!")
