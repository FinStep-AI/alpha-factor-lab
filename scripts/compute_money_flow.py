#!/usr/bin/env python3
"""
因子：资金流向强度 (Money Flow Intensity)
ID: money_flow_v1

逻辑：
  MFI = sum(volume_up, 20d) / sum(volume_total, 20d) - 0.5
  其中：
    volume_up = 上涨日的成交量
    volume_down = 下跌日的成交量  
  
  MFI > 0: 上涨日的成交量占比更大 → 买方力量主导
  MFI < 0: 下跌日的成交量占比更大 → 卖方力量主导
  
  改进版：加入close location权重
  上涨日weight = clv = (close-low)/(high-low), 下跌日weight同样
  这样即使同样是"上涨"，收在高位的日子权重更大（真正的买方主导）
  
  final_factor = sum(clv * amount, 20d) / sum(amount, 20d) - 0.5
  其中clv = 2*(close-low)/(high-low) - 1, 范围[-1, 1]
  CLV > 0 = 收在日内高位半区
  CLV < 0 = 收在日内低位半区
  
  本质上是 Chaikin Money Flow (CMF) 指标
  
  假设：CMF正值（资金净流入）在中证1000有正向预测力

中性化：成交额 OLS残差
"""

import pandas as pd
import numpy as np
import os

def compute_factor(kline_path, output_path):
    print("Loading kline data...")
    df = pd.read_csv(kline_path, dtype={'stock_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    window = 20
    
    # CLV (Close Location Value) = 2*(close-low)/(high-low) - 1
    # 范围 [-1, 1]
    range_ = df['high'] - df['low']
    df['clv'] = np.where(range_ > 0, 2 * (df['close'] - df['low']) / range_ - 1, 0)
    
    # Money Flow Volume = CLV × amount
    df['mf_volume'] = df['clv'] * df['amount']
    
    g = df.groupby('stock_code')
    
    # 20日CMF = sum(MF_volume, 20) / sum(amount, 20)
    df['sum_mf_vol'] = g['mf_volume'].transform(
        lambda x: x.rolling(window, min_periods=16).sum()
    )
    df['sum_amount'] = g['amount'].transform(
        lambda x: x.rolling(window, min_periods=16).sum()
    )
    
    df['factor_raw'] = df['sum_mf_vol'] / df['sum_amount']
    
    # log(20日平均成交额)
    df['mean_amt'] = g['amount'].transform(
        lambda x: x.rolling(window, min_periods=16).mean()
    )
    df['log_amount_20d'] = np.log(df['mean_amt'].clip(lower=1))
    
    factor_df = df[['date', 'stock_code', 'factor_raw', 'log_amount_20d']].dropna().copy()
    print(f"Raw factor: {len(factor_df)} rows, {factor_df['date'].nunique()} dates")
    
    # 截面中性化
    print("Neutralizing...")
    def neutralize_cs(group):
        y = group['factor_raw'].values.copy()
        x = group['log_amount_20d'].values.copy()
        valid = np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 30:
            return pd.Series(np.nan, index=group.index, name='factor')
        y_v, x_v = y[valid], x[valid]
        med = np.median(y_v)
        mad = np.median(np.abs(y_v - med)) * 1.4826
        if mad > 0:
            y_v = np.clip(y_v, med - 3*mad, med + 3*mad)
        x_mat = np.column_stack([np.ones(len(x_v)), x_v])
        beta = np.linalg.lstsq(x_mat, y_v, rcond=None)[0]
        resid = y_v - x_mat @ beta
        std = np.std(resid)
        if std > 0:
            resid = (resid - np.mean(resid)) / std
        out = np.full(len(y), np.nan)
        out[valid] = resid
        return pd.Series(out, index=group.index, name='factor')
    
    factor_df['factor'] = factor_df.groupby('date', group_keys=False).apply(
        lambda g: neutralize_cs(g)
    ).values
    
    output = factor_df[['date', 'stock_code', 'factor']].dropna().copy()
    output['stock_code'] = output['stock_code'].astype(str).str.zfill(6)
    output = output.sort_values(['date', 'stock_code'])
    output.to_csv(output_path, index=False)
    print(f"Factor saved: {output_path} ({len(output)} rows)")
    print(f"Date range: {output['date'].min()} ~ {output['date'].max()}")
    print(f"Stocks/date: {output.groupby('date')['stock_code'].count().mean():.0f}")

if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    compute_factor(
        os.path.join(base, 'data', 'csi1000_kline_raw.csv'),
        os.path.join(base, 'data', 'factor_money_flow_v1.csv')
    )
