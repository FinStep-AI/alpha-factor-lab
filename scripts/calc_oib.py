#!/usr/bin/env python3
"""
签名成交量不平衡因子 (Signed Volume Imbalance)
灵感来源：
- Chordia & Subrahmanyam (2004) "Order Imbalance and Individual Stock Returns" JFE
- Brennan, Huh & Subrahmanyam (2013) "An Analysis of the Amihud Illiquidity Premium" RFS

构造思路：
方法1 - OIB (Order Imbalance)：
  signed_vol_t = volume_t * sign(close_t - open_t)
  OIB_20d = sum(signed_vol) / sum(volume)  ∈ [-1, 1]
  高OIB = 近期持续净买入 → 可能是知情交易的持续积累

方法2 - 改进版：用 CLV * volume 作为更精细的签名
  signed_vol_t = CLV_t * volume_t
  CLV = (2*close - high - low) / (high - low)  ∈ [-1, 1]

两种方法都做市值/成交额中性化。
"""

import numpy as np
import pandas as pd
import os

def build_signed_vol_imbalance(kline_path, output_path, window=20):
    """构建签名成交量不平衡因子"""
    print(f"读取K线数据: {kline_path}")
    df = pd.read_csv(kline_path, parse_dates=['date'])
    df = df[df['amount'] > 0].copy()
    df = df[df['volume'] > 0].copy()
    df = df[df['high'] > df['low']].copy()
    
    print(f"数据: {df['date'].min()} ~ {df['date'].max()}, {df['stock_code'].nunique()} stocks")
    
    # Method 1: sign(close-open) * volume
    df['sign_co'] = np.sign(df['close'] - df['open'])
    # close == open 时用 sign(pct_change) 或 0
    mask_zero = df['sign_co'] == 0
    if 'pct_change' in df.columns:
        df.loc[mask_zero, 'sign_co'] = np.sign(df.loc[mask_zero, 'pct_change'].fillna(0))
    
    df['signed_vol_co'] = df['sign_co'] * df['volume']
    
    # Method 2: CLV * volume  
    hl_range = df['high'] - df['low']
    df['clv'] = (2 * df['close'] - df['high'] - df['low']) / hl_range
    df['signed_vol_clv'] = df['clv'] * df['volume']
    
    df = df.sort_values(['stock_code', 'date'])
    
    # Rolling OIB = sum(signed_vol) / sum(vol) over window
    print(f"计算 {window}日滚动 OIB...")
    
    for method, sv_col, oib_col in [
        ('close-open', 'signed_vol_co', 'oib_co'),
        ('CLV', 'signed_vol_clv', 'oib_clv'),
    ]:
        df[oib_col] = df.groupby('stock_code').apply(
            lambda g: g[sv_col].rolling(window, min_periods=window//2).sum() / 
                       g['volume'].rolling(window, min_periods=window//2).sum()
        ).reset_index(level=0, drop=True)
    
    # 成交额用于中性化
    df['log_amount_ma'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(window, min_periods=window//2).mean() + 1)
    )
    
    df = df.dropna(subset=['oib_co', 'oib_clv', 'log_amount_ma'])
    
    print(f"OIB_CO 统计: mean={df['oib_co'].mean():.4f}, std={df['oib_co'].std():.4f}")
    print(f"OIB_CLV 统计: mean={df['oib_clv'].mean():.4f}, std={df['oib_clv'].std():.4f}")
    
    # 截面中性化 + z-score
    for factor_col, out_name in [('oib_co', 'factor_oib_co'), ('oib_clv', 'factor_oib_clv')]:
        print(f"\n处理 {factor_col}...")
        records = []
        
        for dt, day_df in df.groupby('date'):
            if len(day_df) < 50:
                continue
            
            y = day_df[factor_col].values.astype(float)
            x = day_df['log_amount_ma'].values.astype(float)
            
            # 去除NaN/Inf
            valid = np.isfinite(y) & np.isfinite(x)
            if valid.sum() < 50:
                continue
            
            y_v = y[valid]
            x_v = x[valid]
            
            # OLS 中性化
            X = np.column_stack([np.ones(len(x_v)), x_v])
            try:
                beta = np.linalg.lstsq(X, y_v, rcond=None)[0]
                resid = y_v - X @ beta
            except:
                continue
            
            # MAD winsorize
            med = np.median(resid)
            mad = np.median(np.abs(resid - med))
            if mad < 1e-10:
                continue
            resid = np.clip(resid, med - 5*1.4826*mad, med + 5*1.4826*mad)
            
            # z-score
            mean_r = resid.mean()
            std_r = resid.std()
            if std_r < 1e-10:
                continue
            z = (resid - mean_r) / std_r
            
            codes = day_df['stock_code'].values[valid]
            for c, v in zip(codes, z):
                records.append({'date': dt, 'stock_code': c, 'factor_value': v})
        
        result_df = pd.DataFrame(records)
        path = output_path.replace('.csv', f'_{out_name}.csv')
        result_df.to_csv(path, index=False)
        print(f"  保存: {path} ({len(result_df)} rows)")
        print(f"  统计: mean={result_df['factor_value'].mean():.4f}, std={result_df['factor_value'].std():.4f}")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kline_path = os.path.join(base_dir, 'data', 'csi1000_kline_raw.csv')
    output_path = os.path.join(base_dir, 'data', 'factor_oib_v1.csv')
    
    build_signed_vol_imbalance(kline_path, output_path, window=20)
