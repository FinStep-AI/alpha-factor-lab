#!/usr/bin/env python3
"""
成交量趋势/异常成交量因子
参考:
  - Gervais, Kaniel & Mingelgrin (2001) "The High-Volume Return Premium", JF
  - Campbell, Grossman & Wang (1993) "Trading Volume and Serial Correlation", QJE
  
变体:
  v1: vol_trend = log(MA5_volume/MA20_volume) 近期放量信号
  v2: vol_price_diverge = vol_trend - price_trend 量增价平(蓄势)
  v3: high_vol_streak = 近20日中成交量>MA60的天数占比
  v4: vol_momentum = vol变化率的20日均值 (成交量持续放量)
  v5: smart_money_flow = 上涨日成交量趋势 vs 下跌日成交量趋势
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

print("=== 成交量趋势因子 ===\n")

kline = pd.read_csv("data/csi1000_kline_raw.csv")
kline['date'] = pd.to_datetime(kline['date'])
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
kline['returns'] = kline.groupby('stock_code')['close'].pct_change()

g = kline.groupby('stock_code')
print(f"数据: {kline['stock_code'].nunique()} 只股票")

# 衍生字段
kline['ma5_vol'] = g['volume'].transform(lambda x: x.rolling(5, min_periods=3).mean())
kline['ma20_vol'] = g['volume'].transform(lambda x: x.rolling(20, min_periods=14).mean())
kline['ma60_vol'] = g['volume'].transform(lambda x: x.rolling(60, min_periods=40).mean())
kline['ma5_close'] = g['close'].transform(lambda x: x.rolling(5, min_periods=3).mean())
kline['ma20_close'] = g['close'].transform(lambda x: x.rolling(20, min_periods=14).mean())

# v1: 成交量趋势
kline['vol_trend_v1'] = np.log(kline['ma5_vol'] / kline['ma20_vol'].replace(0, np.nan))

# v2: 量增价平 = vol趋势 - price趋势 (正向: 量增价不涨=蓄势)
price_trend = np.log(kline['ma5_close'] / kline['ma20_close'].replace(0, np.nan))
kline['vol_price_div_v2'] = kline['vol_trend_v1'] - price_trend

# v3: 高成交量天数占比 (近20日中vol > MA60的天数比例)
kline['vol_above_ma60'] = (kline['volume'] > kline['ma60_vol']).astype(float)
kline['high_vol_ratio_v3'] = g['vol_above_ma60'].transform(lambda x: x.rolling(20, min_periods=14).mean())

# v4: 成交量动量 = volume pct_change的20日均值
kline['vol_pct_change'] = g['volume'].pct_change()
kline['vol_momentum_v4'] = g['vol_pct_change'].transform(lambda x: x.rolling(20, min_periods=14).mean())

# v5: 分日累计: 上涨日vol趋势 vs 下跌日vol趋势
kline['up_vol'] = np.where(kline['returns'] > 0, kline['volume'], 0)
kline['dn_vol'] = np.where(kline['returns'] <= 0, kline['volume'], 0)
kline['up_vol_ma20'] = g['up_vol'].transform(lambda x: x.rolling(20, min_periods=14).sum())
kline['dn_vol_ma20'] = g['dn_vol'].transform(lambda x: x.rolling(20, min_periods=14).sum())
# 上涨日成交量 / 下跌日成交量  (高 = 买方主导)
kline['smart_flow_v5'] = np.log(kline['up_vol_ma20'] / kline['dn_vol_ma20'].replace(0, np.nan))

# 成交额用于中性化
kline['log_amount_20d'] = g['amount'].transform(
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
        y = np.clip(y, med - 3*1.4826*mad, med + 3*1.4826*mad)
    lr = LinearRegression()
    lr.fit(X, y)
    residuals = y - lr.predict(X)
    mu, sigma = residuals.mean(), residuals.std()
    z = (residuals - mu) / sigma if sigma > 0 else np.zeros_like(residuals)
    z = np.clip(z, -3, 3)
    result = pd.Series(np.nan, index=df_day.index)
    result.iloc[mask.values] = z
    return result

# 处理每个变体（正向和反向都试）
variants = {
    'vol_trend_v1': 'vol_trend_v1',
    'vol_price_div_v2': 'vol_price_div_v2',
    'high_vol_ratio_v3': 'high_vol_ratio_v3',
    'vol_momentum_v4': 'vol_momentum_v4',
    'smart_flow_v5': 'smart_flow_v5',
}

for name, col in variants.items():
    print(f"\n处理 {name} (正向)...")
    factor = kline.groupby('date', group_keys=False).apply(
        lambda d, c=col: neutralize_and_standardize(d, c)
    )
    kline[f'{name}_pos'] = factor.values if hasattr(factor, 'values') else factor
    
    out = kline[['date', 'stock_code', f'{name}_pos']].dropna(subset=[f'{name}_pos']).copy()
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    out = out.rename(columns={f'{name}_pos': 'factor_value'})
    fname = f"data/factor_{name}_pos.csv"
    out.to_csv(fname, index=False)
    print(f"  ✅ {fname}: {len(out)} 行")

    # 反向
    print(f"处理 {name} (反向)...")
    kline[f'{name}_neg_raw'] = -kline[col]
    factor_neg = kline.groupby('date', group_keys=False).apply(
        lambda d, c=f'{name}_neg_raw': neutralize_and_standardize(d, c)
    )
    kline[f'{name}_neg'] = factor_neg.values if hasattr(factor_neg, 'values') else factor_neg
    
    out_neg = kline[['date', 'stock_code', f'{name}_neg']].dropna(subset=[f'{name}_neg']).copy()
    out_neg['date'] = out_neg['date'].dt.strftime('%Y-%m-%d')
    out_neg = out_neg.rename(columns={f'{name}_neg': 'factor_value'})
    fname_neg = f"data/factor_{name}_neg.csv"
    out_neg.to_csv(fname_neg, index=False)
    print(f"  ✅ {fname_neg}: {len(out_neg)} 行")

print("\n✅ 全部完成!")
