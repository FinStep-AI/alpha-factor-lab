#!/usr/bin/env python3
"""
因子: 振幅变异系数(反) (Amplitude CV Negative)
逻辑: -CV(amplitude, 20d) = -std(amp)/mean(amp)
低振幅波动 = 价格行为规律性强 = 信息不对称低 = 高预期收益
与vol_cv_neg(成交量变异系数)同族但不同维度
中性化: 成交额OLS中性化 + MAD winsorize + z-score
"""

import pandas as pd
import numpy as np

OUTPUT_PATH = "data/factor_amp_cv_neg_v1.csv"
LOOKBACK = 20

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
    
    # 计算振幅 = (high - low) / prev_close
    kline['prev_close'] = kline.groupby('stock_code')['close'].shift(1)
    kline['amplitude'] = (kline['high'] - kline['low']) / kline['prev_close']
    
    # 20日振幅CV
    print(f"Computing {LOOKBACK}d amplitude CV...")
    
    amp_mean = kline.groupby('stock_code')['amplitude'].transform(
        lambda x: x.rolling(LOOKBACK, min_periods=15).mean()
    )
    amp_std = kline.groupby('stock_code')['amplitude'].transform(
        lambda x: x.rolling(LOOKBACK, min_periods=15).std()
    )
    
    kline['raw_factor'] = -(amp_std / amp_mean.clip(lower=1e-6))  # 负号: 低CV=高因子值
    
    # 20日成交额
    amt_20d = kline.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )
    kline['log_amount_20d'] = np.log(amt_20d.clip(lower=1))
    
    df = kline[kline['raw_factor'].notna() & kline['log_amount_20d'].notna()][
        ['date', 'stock_code', 'raw_factor', 'log_amount_20d']
    ].copy()
    
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
