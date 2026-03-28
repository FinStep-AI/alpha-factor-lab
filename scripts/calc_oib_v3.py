#!/usr/bin/env python3
"""
OIB加速度因子 v3：短期(20日)OIB - 长期(60日)OIB 的差值
逻辑：如果近期卖压相比长期在加速 → 更强的反转信号

也测试：CLV 的 20日 variance（不平衡波动性）
"""

import numpy as np
import pandas as pd
import os
from scipy.stats import rankdata

def build_factor(kline_path, output_path):
    print(f"读取K线数据: {kline_path}")
    df = pd.read_csv(kline_path, parse_dates=['date'])
    df = df[(df['amount'] > 0) & (df['volume'] > 0) & (df['high'] > df['low'])].copy()
    
    print(f"数据: {df['date'].min()} ~ {df['date'].max()}, {df['stock_code'].nunique()} stocks")
    
    df['clv'] = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'])
    df['signed_vol'] = df['clv'] * df['volume']
    
    df = df.sort_values(['stock_code', 'date'])
    
    # 20日和60日OIB
    print("计算 20日/60日 OIB...")
    for w, name in [(20, 'oib_20'), (60, 'oib_60')]:
        df[f'sv_sum_{w}'] = df.groupby('stock_code')['signed_vol'].transform(
            lambda x: x.rolling(w, min_periods=max(w//2, 10)).sum()
        )
        df[f'vol_sum_{w}'] = df.groupby('stock_code')['volume'].transform(
            lambda x: x.rolling(w, min_periods=max(w//2, 10)).sum()
        )
        df[name] = df[f'sv_sum_{w}'] / (df[f'vol_sum_{w}'] + 1e-10)
    
    # OIB加速度 = OIB_20 - OIB_60
    df['oib_accel'] = df['oib_20'] - df['oib_60']
    # 翻转（卖压加速 → 正因子，反转逻辑）
    df['oib_accel_neg'] = -df['oib_accel']
    
    # 成交额
    df['log_amount_ma'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    
    df = df.dropna(subset=['oib_accel_neg', 'log_amount_ma'])
    
    print(f"OIB_accel 统计: mean={df['oib_accel'].mean():.4f}, std={df['oib_accel'].std():.4f}")
    
    # 截面中性化
    print("截面中性化...")
    factor_records = []
    
    for dt, day_df in df.groupby('date'):
        if len(day_df) < 50:
            continue
        
        y_raw = day_df['oib_accel_neg'].values.astype(float)
        x = day_df['log_amount_ma'].values.astype(float)
        
        valid = np.isfinite(y_raw) & np.isfinite(x)
        if valid.sum() < 50:
            continue
        
        y_v = y_raw[valid]
        x_v = x[valid]
        
        # Rank transform
        y_rank = rankdata(y_v) / (len(y_v) + 1)
        
        # OLS 中性化
        X = np.column_stack([np.ones(len(x_v)), x_v])
        try:
            beta = np.linalg.lstsq(X, y_rank, rcond=None)[0]
            resid = y_rank - X @ beta
        except:
            continue
        
        # MAD winsorize + z-score
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        if mad < 1e-10:
            continue
        resid = np.clip(resid, med - 5*1.4826*mad, med + 5*1.4826*mad)
        
        mean_r = resid.mean()
        std_r = resid.std()
        if std_r < 1e-10:
            continue
        z = (resid - mean_r) / std_r
        
        codes = day_df['stock_code'].values[valid]
        for c, v in zip(codes, z):
            factor_records.append({'date': dt, 'stock_code': c, 'factor_value': v})
    
    factor_df = pd.DataFrame(factor_records)
    factor_df.to_csv(output_path, index=False)
    print(f"\n保存: {output_path} ({len(factor_df)} rows)")
    print(f"因子统计: mean={factor_df['factor_value'].mean():.4f}, std={factor_df['factor_value'].std():.4f}")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kline_path = os.path.join(base_dir, 'data', 'csi1000_kline_raw.csv')
    output_path = os.path.join(base_dir, 'data', 'factor_oib_accel_v3.csv')
    build_factor(kline_path, output_path)
