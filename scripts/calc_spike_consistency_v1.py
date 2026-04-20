#!/usr/bin/env python3
"""量价 Spike 一致性 v2

构造逻辑（与vol_ret_align_v1的区分）：
  - v1 用的是 sign(vol_change) × sign(ret) → 剧烈变动方向一致性
  - v2 用的是 corr(|vol_change|, |ret|) → **量价同步度**（幅度一致性）
    高相关 = 大成交量日出现大波动 = 信息驱动 (vs 低相关 = 噪音驱动)

  与已有因子区分度：
  - pv_corr_v1: 用的是 sign 方向相关性 (同向/反向)
  - vol_ret_align_v1: 用的是 sign(vol_chg)×sign(ret) 方向一致性
  - 本因子: 用 |vol_chg| 与 |ret| 的幅度协同，反映信息密度
  三者角度不同

构造：20日 rolling corr(|d_volume/mean|, |returns|)，MAD winsorize + z-score
方向：正向 (高协同 = 信息驱动 = 高收益)
"""
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'skills', 'alpha-factor-lab', 'scripts'))
from factor_calculator import load_data, zscore_cross_section
import numpy as np
import pandas as pd

print('[信息] 加载数据...')
df = load_data('data/csi1000_kline_raw.csv').sort_values(['stock_code', 'date']).reset_index(drop=True)
g = df.groupby('stock_code')

# 成交量变化率
df['vol_chg'] = df['volume'] / g['volume'].shift(1).replace(0, np.nan) - 1
df['vol_chg_abs'] = df['vol_chg'].abs().clip(upper=5)
df['ret_abs'] = df['close'].pct_change().abs()

# 20日滚动相关系数: |vol_chg| vs |returns|
print('[信息] 计算20日rolling corr(|vol_chg|, |ret|)...')
df['_raw'] = g['vol_chg_abs'].transform(
    lambda x: x.rolling(20, min_periods=10).corr(g['ret_abs'].iloc[x.index])
)

# 用 shift 避免 future leak: 计算当天 T-1 的值
df['_raw'] = g['_raw'].shift(1)

# MAD Winsorize + z-score (截面)
df['factor_value'] = df.groupby('date')['_raw'].transform(lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)))
df['factor_value'] = zscore_cross_section(df, 'factor_value')

out = df[['date', 'stock_code', 'factor_value']].dropna(subset=['factor_value']).copy()
out['date'] = out['date'].dt.strftime('%Y-%m-%d')
out.to_csv('data/factor_spike_consistency_v1.csv', index=False)

fv = df['factor_value'].dropna()
print(f'\n[结果] data/factor_spike_consistency_v1.csv')
print(f'N={fv.notna().sum()} mean={fv.mean():.4f} std={fv.std():.4f}')

