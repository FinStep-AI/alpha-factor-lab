"""
适度冒险因子（日线近似版 v1）
来源：方正金工"成交量激增时刻蕴含的alpha信息"
用日线OHLCV近似构造，向量化实现。

逻辑：
1. 成交量激增日：日成交量增量 > 过去20日增量均值 + 1倍标准差
2. 激增日耀眼收益率 = 当日收益率
3. 激增日耀眼波动率 = 当日日内波动率 (high-low)/close
4. 适度偏离 = |个股耀眼指标 - 截面均值|
5. 过去20个交易日：对激增日求平均适度偏离
6. 因子 = 适度收益率偏离 + 适度波动率偏离（等权）
7. 负向因子：值越高越不适度，预期收益越低
"""

import pandas as pd
import numpy as np
import os, sys

def compute_factor(kline_path, output_path, lookback=20, surge_window=20):
    print(f"读取K线: {kline_path}")
    df = pd.read_csv(kline_path)
    
    # 标准化列名
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ('code', 'stock_code', 'symbol'): col_map[c] = 'code'
        elif cl in ('date', 'trade_date'): col_map[c] = 'date'
        elif cl in ('close',): col_map[c] = 'close'
        elif cl in ('volume', 'vol'): col_map[c] = 'volume'
        elif cl in ('high',): col_map[c] = 'high'
        elif cl in ('low',): col_map[c] = 'low'
    df = df.rename(columns=col_map)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    
    print(f"股票数: {df['code'].nunique()}, 总行数: {len(df)}")
    
    # 1. 计算基础指标（向量化，按code分组）
    df['ret'] = df.groupby('code')['close'].pct_change()
    df['vol_delta'] = df.groupby('code')['volume'].diff()
    df['intraday_vol'] = (df['high'] - df['low']) / df['close']
    
    # 2. 滚动统计判断激增
    df['vd_mean'] = df.groupby('code')['vol_delta'].transform(
        lambda x: x.rolling(surge_window, min_periods=10).mean()
    )
    df['vd_std'] = df.groupby('code')['vol_delta'].transform(
        lambda x: x.rolling(surge_window, min_periods=10).std()
    )
    df['is_surge'] = (df['vol_delta'] > (df['vd_mean'] + df['vd_std'])).astype(int)
    
    print(f"激增日比例: {df['is_surge'].mean():.3f}")
    
    # 3. 激增日的耀眼指标（直接用当日）
    df['dazzle_ret'] = np.where(df['is_surge'] == 1, df['ret'], np.nan)
    df['dazzle_vol'] = np.where(df['is_surge'] == 1, df['intraday_vol'], np.nan)
    
    # 4. 截面均值（每日所有激增股的均值）
    daily_means = df[df['is_surge'] == 1].groupby('date').agg(
        ret_xmean=('dazzle_ret', 'mean'),
        vol_xmean=('dazzle_vol', 'mean')
    ).reset_index()
    
    df = df.merge(daily_means, on='date', how='left')
    
    # 5. 适度偏离（仅激增日有值）
    df['mod_ret'] = np.where(df['is_surge'] == 1,
        (df['dazzle_ret'] - df['ret_xmean']).abs(), np.nan)
    df['mod_vol'] = np.where(df['is_surge'] == 1,
        (df['dazzle_vol'] - df['vol_xmean']).abs(), np.nan)
    
    # 6. 过去lookback个交易日内，激增日适度偏离的均值
    # 用rolling + min_periods=1, 对NaN自动跳过
    df['factor_ret'] = df.groupby('code')['mod_ret'].transform(
        lambda x: x.rolling(lookback, min_periods=3).mean()
    )
    df['factor_vol'] = df.groupby('code')['mod_vol'].transform(
        lambda x: x.rolling(lookback, min_periods=3).mean()
    )
    
    # 7. 合成因子
    df['factor_value'] = (df['factor_ret'].fillna(0) + df['factor_vol'].fillna(0)) / 2
    
    # 只保留有有效因子值的
    valid = df[df['factor_value'] > 0][['code', 'date', 'factor_value']].copy()
    valid = valid.dropna()
    
    print(f"有效因子记录数: {len(valid)}")
    print(f"覆盖股票数: {valid['code'].nunique()}")
    print(f"日期范围: {valid['date'].min()} ~ {valid['date'].max()}")
    
    valid.to_csv(output_path, index=False)
    print(f"因子已保存: {output_path}")
    return valid


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kline_path = os.path.join(base_dir, 'data', 'csi1000_kline_raw.csv')
    output_path = os.path.join(base_dir, 'data', 'factor_moderate_risk_v1.csv')
    compute_factor(kline_path, output_path)
