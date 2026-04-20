#!/usr/bin/env python3
"""大单占比代理因子 v1 (Large Order Proxy v1)

构造逻辑（基于日线OHLCV近似主力资金流向）：
  CLV = (2*close - high - low) / (high - low)
       ≈ 收盘价在日内振幅中的位置 [-1, 1]
       >0: 收在当日高位（买方支撑强）
       <0: 收在当日低位（卖方压力强）
  
  用 CLV × volume 作为"大单方向信号"的日频代理
  20日滚动累计和 = 资金方向持续性
  对数成交额中性化（OLS）+ MAD缩尾 + z-score
  
正向使用: 高因子值 = 近期持续大单买入 = 惯性延续
"""
import numpy as np
import pandas as pd
from factor_calculator import load_data, neutralize_cross_section, zscore_cross_section, winsorize_mad

def calc_large_order_proxy(df_kline: pd.DataFrame) -> pd.Series:
    df = df_kline.copy()
    
    # CLV: 收盘位置 [-1, 1]
    hl_range = df['high'] - df['low']
    # 防止除以0
    hl_range = hl_range.replace(0, np.nan)
    clv = (2 * df['close'] - df['high'] - df['low']) / hl_range
    clv = clv.clip(-1, 1)
    
    # 成交量加权 CLV（近似"大单方向"信号）
    vol_weighted_clv = clv * df['volume']
    
    # 20日滚动累计和（资金方向持续性）
    g = df.groupby('stock_code')
    cumsum_clv = g['vol_weighted_clv'].transform(
        lambda x: x.rolling(20, min_periods=10).sum()
    )
    
    # 以20日平均成交额中性化
    amt_20d = g['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    log_amt = np.log(amt_20d.clip(lower=1))
    
    df['_factor_raw'] = cumsum_clv
    df['_log_amount'] = log_amt
    
    # 横截面对数成交额中性化（OLS残差）
    neutralized = neutralize_cross_section(
        df, '_factor_raw', neutralize_cols=['_log_amount']
    )
    
    # MAD Winsorize + z-score
    neutralized = neutralized.clip(-3, 3)
    neutralized = zscore_cross_section(df.assign(_neutralized=neutralized), '_neutralized')
    
    return neutralized

def main():
    import sys
    
    print("[信息] 加载K线数据...")
    df = load_data('data/csi1000_kline_raw.csv')
    
    # 预计算 vol_weighted_clv
    print("[信息] 计算 CLV × volume...")
    hl_range = df['high'] - df['low']
    hl_range = hl_range.replace(0, np.nan)
    df['vol_weighted_clv'] = ((2 * df['close'] - df['high'] - df['low']) / hl_range).clip(-1, 1) * df['volume']
    
    print("[信息] 计算因子: 20日 CLV×volume 累计和 (大单占比代理)")
    factor = calc_large_order_proxy(df)
    
    df['factor_value'] = factor
    
    # 输出
    out = df[['date', 'stock_code', 'factor_value']].dropna(subset=['factor_value']).copy()
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    out.to_csv('data/factor_large_order_proxy_v1.csv', index=False)
    
    print(f"\n[结果] 因子保存: data/factor_large_order_proxy_v1.csv")
    print(f"有效值: {factor.notna().sum()} / {len(factor)} ({factor.notna().mean()*100:.1f}%)")
    print(f"均值: {factor.mean():.4f}  标准差: {factor.std():.4f}")
    print(f"最小值: {factor.min():.4f}  最大值: {factor.max():.4f}")
    print(f"偏度: {factor.skew():.4f}  峰度: {factor.kurtosis():.4f}")

if __name__ == '__main__':
    main()
