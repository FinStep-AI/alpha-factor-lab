#!/usr/bin/env python3
"""
因子: 成交额峰度因子 (Amount Kurtosis)
逻辑: 20日成交额的峰度(kurtosis)取负
低峰度(负因子) = 成交额分布均匀 = 稳定交易模式
高峰度 = 成交额集中在少数日 = 脉冲式交易 = 噪声/事件驱动

其实换个思路 - 让我试一个更创新的:

因子: 涨跌缺口不对称性 (Gap Asymmetry)
逻辑: 20天中上涨日跳空幅度均值 vs 下跌日跳空幅度均值的差
正值 = 上涨时跳空大,下跌时跳空小 = 看多力量在集合竞价占优
"""

import pandas as pd
import numpy as np

OUTPUT_PATH = "data/factor_gap_asym_v1.csv"
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
    
    # 计算跳空幅度 = (open - prev_close) / prev_close
    kline['prev_close'] = kline.groupby('stock_code')['close'].shift(1)
    kline['gap'] = (kline['open'] - kline['prev_close']) / kline['prev_close']
    
    # 日收益率
    kline['ret'] = kline['close'].pct_change()  # 简化版
    kline['ret'] = kline.groupby('stock_code')['close'].pct_change()
    
    # 成交额
    amt_20d = kline.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )
    kline['log_amount_20d'] = np.log(amt_20d.clip(lower=1))
    
    print(f"Computing {LOOKBACK}d gap asymmetry...")
    
    all_results = []
    stocks = kline['stock_code'].unique()
    
    for i, stock in enumerate(stocks):
        if (i + 1) % 200 == 0:
            print(f"  Processing {i+1}/{len(stocks)} stocks...")
        
        stock_data = kline[kline['stock_code'] == stock].sort_values('date')
        gaps = stock_data['gap'].values
        rets = stock_data['ret'].values
        dates = stock_data['date'].values
        codes = stock_data['stock_code'].values
        
        for j in range(LOOKBACK - 1, len(gaps)):
            g_window = gaps[j - LOOKBACK + 1: j + 1]
            r_window = rets[j - LOOKBACK + 1: j + 1]
            
            valid = ~(np.isnan(g_window) | np.isnan(r_window))
            if valid.sum() < 15:
                continue
            
            g_valid = g_window[valid]
            r_valid = r_window[valid]
            
            # 上涨日(全天收益>0)的平均跳空
            up_mask = r_valid > 0
            down_mask = r_valid < 0
            
            if up_mask.sum() < 3 or down_mask.sum() < 3:
                continue
            
            up_gap_mean = np.mean(np.abs(g_valid[up_mask]))  # 上涨日跳空绝对幅度
            down_gap_mean = np.mean(np.abs(g_valid[down_mask]))  # 下跌日跳空绝对幅度
            
            # 不对称性: 上涨日跳空大 vs 下跌日跳空大
            # 用比值: log(up/down), 正=上涨日跳空更大
            if up_gap_mean > 0 and down_gap_mean > 0:
                asym = np.log(up_gap_mean / down_gap_mean)
            else:
                continue
            
            all_results.append({
                'date': dates[j],
                'stock_code': codes[j],
                'raw_factor': asym
            })
    
    df = pd.DataFrame(all_results)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Raw factor computed: {len(df)} rows")
    
    df = df.merge(
        kline[['date', 'stock_code', 'log_amount_20d']],
        on=['date', 'stock_code'],
        how='left'
    )
    
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
