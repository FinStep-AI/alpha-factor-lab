#!/usr/bin/env python3
"""
因子: volume_price_trend_div_v1 (量价趋势背离因子)
优化版: 用pandas rolling + apply 加速
"""

import numpy as np
import pandas as pd
from pathlib import Path

def rolling_slope_fast(series, window=20):
    """用rolling apply计算标准化斜率 = slope / std"""
    
    def _slope_std(y):
        n = len(y)
        if np.isnan(y).sum() > n * 0.3:
            return np.nan
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        x_var = ((x - x_mean)**2).sum()
        y_clean = np.where(np.isnan(y), np.nanmean(y), y)
        y_mean = y_clean.mean()
        slope = ((x - x_mean) * (y_clean - y_mean)).sum() / x_var
        std = y_clean.std()
        if std < 1e-10:
            return 0.0
        return slope / std
    
    return series.rolling(window, min_periods=window).apply(_slope_std, raw=True)

def mad_winsorize(series, n_mad=5):
    median = series.median()
    mad = (series - median).abs().median()
    if mad < 1e-10:
        return series
    upper = median + n_mad * 1.4826 * mad
    lower = median - n_mad * 1.4826 * mad
    return series.clip(lower, upper)

def neutralize_ols(factor_series, neutralize_series):
    valid = factor_series.notna() & neutralize_series.notna()
    if valid.sum() < 10:
        return factor_series
    X = np.column_stack([neutralize_series[valid].values, np.ones(valid.sum())])
    y = factor_series[valid].values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ beta
    result = pd.Series(np.nan, index=factor_series.index)
    result[valid] = residual
    return result

def main():
    data_dir = Path(__file__).parent.parent / "data"
    kline_file = data_dir / "csi1000_kline_raw.csv"
    output_file = data_dir / "factor_vol_price_trend_div_v1.csv"
    
    print("读取K线数据...")
    df = pd.read_csv(kline_file, dtype={'stock_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    df = df[df['date'] <= '2026-03-17']
    
    print(f"数据: {df['date'].min()} ~ {df['date'].max()}, 股票数: {df['stock_code'].nunique()}")
    
    window = 20
    
    # 按股票分组计算
    print("计算量价趋势背离...")
    
    def calc_per_stock(sdf):
        if len(sdf) < window + 5:
            return pd.DataFrame()
        sdf = sdf.sort_values('date').copy()
        
        # 价格标准化趋势斜率
        price_trend = rolling_slope_fast(sdf['close'], window)
        
        # 成交量标准化趋势斜率
        vol_trend = rolling_slope_fast(sdf['volume'], window)
        
        # 背离 = 量趋势 - 价趋势
        sdf['factor_raw'] = vol_trend - price_trend
        sdf['log_amount_20d'] = np.log1p(sdf['amount'].rolling(window).mean())
        
        return sdf[['date', 'stock_code', 'factor_raw', 'log_amount_20d']].dropna()
    
    results = []
    stocks = df['stock_code'].unique()
    for i, stock in enumerate(stocks):
        if (i + 1) % 200 == 0:
            print(f"  进度: {i+1}/{len(stocks)}")
        sdf = df[df['stock_code'] == stock]
        r = calc_per_stock(sdf)
        if len(r) > 0:
            results.append(r)
    
    factor_df = pd.concat(results, ignore_index=True)
    print(f"原始因子: {len(factor_df)}条记录")
    
    # 截面处理
    print("截面中性化+标准化...")
    final = []
    for date, gdf in factor_df.groupby('date'):
        if len(gdf) < 50:
            continue
        gdf = gdf.copy()
        factor_neutral = neutralize_ols(gdf['factor_raw'], gdf['log_amount_20d'])
        factor_win = mad_winsorize(factor_neutral)
        std = factor_win.std()
        if std > 0:
            factor_z = (factor_win - factor_win.mean()) / std
        else:
            continue
        gdf['factor_value'] = factor_z.values
        final.append(gdf[['date', 'stock_code', 'factor_value']])
    
    result_df = pd.concat(final, ignore_index=True).sort_values(['date', 'stock_code'])
    result_df.to_csv(output_file, index=False)
    
    print(f"\n输出: {output_file}")
    print(f"日期: {result_df['date'].min()} ~ {result_df['date'].max()}")
    print(f"每日平均: {result_df.groupby('date')['stock_code'].count().mean():.0f}只")
    print(f"统计: mean={result_df['factor_value'].mean():.4f}, std={result_df['factor_value'].std():.4f}")

if __name__ == '__main__':
    main()
