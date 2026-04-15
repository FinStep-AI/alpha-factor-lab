#!/usr/bin/env python3
"""
因子: 高低价位成交额比 (High-Low Price Volume Ratio)
逻辑: 过去20天中, 收盘价高于MA20的日子的平均成交额 vs 低于MA20的日子的平均成交额
log(above_avg / below_avg)
正向: 高于均线时放量 = 上方有资金支撑 = 看多信号
这是一种成交额在不同价位的分布特征

中性化: 成交额OLS中性化 + MAD winsorize + z-score
"""

import pandas as pd
import numpy as np

OUTPUT_PATH = "data/factor_hlpvr_v1.csv"
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
    
    # MA20
    kline['ma20'] = kline.groupby('stock_code')['close'].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )
    
    # 20日成交额
    amt_20d = kline.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )
    kline['log_amount_20d'] = np.log(amt_20d.clip(lower=1))
    
    # 标记高于/低于MA20
    kline['above_ma'] = (kline['close'] > kline['ma20']).astype(float)
    
    print(f"Computing {LOOKBACK}d high-low price volume ratio...")
    
    all_results = []
    stocks = kline['stock_code'].unique()
    
    for i, stock in enumerate(stocks):
        if (i + 1) % 200 == 0:
            print(f"  Processing {i+1}/{len(stocks)} stocks...")
        
        stock_data = kline[kline['stock_code'] == stock].sort_values('date')
        closes = stock_data['close'].values
        amounts = stock_data['amount'].values
        ma20s = stock_data['ma20'].values
        dates = stock_data['date'].values
        codes = stock_data['stock_code'].values
        
        for j in range(LOOKBACK - 1, len(closes)):
            c_window = closes[j - LOOKBACK + 1: j + 1]
            a_window = amounts[j - LOOKBACK + 1: j + 1]
            m_window = ma20s[j - LOOKBACK + 1: j + 1]
            
            valid = ~(np.isnan(c_window) | np.isnan(a_window) | np.isnan(m_window))
            if valid.sum() < 15:
                continue
            
            c_v = c_window[valid]
            a_v = a_window[valid]
            m_v = m_window[valid]
            
            above = c_v > m_v
            below = ~above
            
            if above.sum() < 3 or below.sum() < 3:
                continue
            
            avg_amt_above = np.mean(a_v[above])
            avg_amt_below = np.mean(a_v[below])
            
            if avg_amt_above > 0 and avg_amt_below > 0:
                ratio = np.log(avg_amt_above / avg_amt_below)
            else:
                continue
            
            all_results.append({
                'date': dates[j],
                'stock_code': codes[j],
                'raw_factor': ratio
            })
    
    df = pd.DataFrame(all_results)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Raw factor computed: {len(df)} rows")
    
    df = df.merge(
        kline[['date', 'stock_code', 'log_amount_20d']],
        on=['date', 'stock_code'],
        how='left'
    )
    
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
