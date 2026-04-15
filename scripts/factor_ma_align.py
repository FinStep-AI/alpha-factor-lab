#!/usr/bin/env python3
"""
因子: 均线多头排列度 (MA Alignment Score)
逻辑: 衡量MA5/MA10/MA20/MA60的多头排列一致性
  - 计算所有pair-wise比较中短期MA > 长期MA的比例
  - 完美多头(MA5>MA10>MA20>MA60) = 1.0
  - 完美空头(MA5<MA10<MA20<MA60) = 0.0
  - 混乱排列 ≈ 0.5
与ma_disp(离散度)互补: 离散度衡量发散程度, alignment衡量方向一致性
中性化: 成交额OLS中性化 + MAD winsorize + z-score
"""

import pandas as pd
import numpy as np
import sys

OUTPUT_PATH = "data/factor_ma_align_v1.csv"

MA_WINDOWS = [5, 10, 20, 60]

def mad_winsorize(s, n_mad=5):
    median = s.median()
    mad = (s - median).abs().median()
    if mad == 0:
        return s
    upper = median + n_mad * 1.4826 * mad
    lower = median - n_mad * 1.4826 * mad
    return s.clip(lower=lower, upper=upper)

def neutralize_ols(factor_series, neutralizer_series):
    from numpy.linalg import lstsq
    mask = factor_series.notna() & neutralizer_series.notna()
    if mask.sum() < 10:
        return factor_series
    X = np.column_stack([np.ones(mask.sum()), neutralizer_series[mask].values])
    y = factor_series[mask].values
    beta, _, _, _ = lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    result = factor_series.copy()
    result[mask] = residuals
    return result

def main():
    print("Loading data...")
    kline = pd.read_csv("data/csi1000_kline_raw.csv")
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # 计算各均线
    print("Computing moving averages...")
    for w in MA_WINDOWS:
        kline[f'ma{w}'] = kline.groupby('stock_code')['close'].transform(
            lambda x: x.rolling(w, min_periods=max(1, int(w*0.8))).mean()
        )
    
    # 计算20日平均成交额用于中性化
    amt_20d = kline.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )
    kline['log_amount_20d'] = np.log(amt_20d.clip(lower=1))
    
    # 计算alignment score
    print("Computing MA alignment score...")
    n_pairs = len(MA_WINDOWS) * (len(MA_WINDOWS) - 1) // 2  # C(4,2)=6
    
    # 所有pair-wise比较(短期 vs 长期)
    pairs = []
    for i in range(len(MA_WINDOWS)):
        for j in range(i + 1, len(MA_WINDOWS)):
            pairs.append((MA_WINDOWS[i], MA_WINDOWS[j]))  # (短期, 长期)
    
    # 对每个pair，检查短期MA > 长期MA (多头)
    alignment = np.zeros(len(kline))
    valid_mask = np.ones(len(kline), dtype=bool)
    
    for short_w, long_w in pairs:
        short_ma = kline[f'ma{short_w}'].values
        long_ma = kline[f'ma{long_w}'].values
        
        valid = ~(np.isnan(short_ma) | np.isnan(long_ma))
        valid_mask &= valid
        
        # 短期 > 长期 得1分
        alignment += np.where(valid & (short_ma > long_ma), 1.0, 0.0)
    
    kline['raw_factor'] = alignment / n_pairs  # [0, 1]
    kline.loc[~valid_mask, 'raw_factor'] = np.nan
    
    # 过滤: 需要MA60有效(至少60天数据)
    kline.loc[kline['ma60'].isna(), 'raw_factor'] = np.nan
    
    df = kline[kline['raw_factor'].notna()][['date', 'stock_code', 'raw_factor', 'log_amount_20d']].copy()
    print(f"Raw factor computed: {len(df)} rows")
    
    # 截面处理
    print("Cross-sectional processing...")
    processed = []
    
    for date, group in df.groupby('date'):
        g = group.copy()
        g['factor'] = mad_winsorize(g['raw_factor'])
        g['factor'] = neutralize_ols(g['factor'], g['log_amount_20d'])
        mean = g['factor'].mean()
        std = g['factor'].std()
        if std > 0:
            g['factor'] = (g['factor'] - mean) / std
        else:
            g['factor'] = 0
        processed.append(g[['date', 'stock_code', 'factor']])
    
    result = pd.concat(processed, ignore_index=True)
    result = result.sort_values(['date', 'stock_code']).reset_index(drop=True)
    
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}: {len(result)} rows")
    print(f"Date range: {result['date'].min()} ~ {result['date'].max()}")
    print(f"Factor stats:\n{result['factor'].describe()}")

if __name__ == "__main__":
    main()
