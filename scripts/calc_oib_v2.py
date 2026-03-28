#!/usr/bin/env python3
"""
OIB改进版：
1. 60日窗口（捕获更持久的资金流动方向）
2. 成交量衰减加权（近期权重更大）
3. 用 close 位于 high-low range 的位置 (CLV) 作为签名
4. 对 OIB 做 rank 变换后再中性化（更稳健）
"""

import numpy as np
import pandas as pd
import os

def exponential_weights(n, halflife=20):
    """指数衰减权重，最近的权重最大"""
    w = np.exp(-np.log(2) / halflife * np.arange(n-1, -1, -1))
    return w / w.sum()

def build_factor(kline_path, output_path, window=60, halflife=20):
    print(f"读取K线数据: {kline_path}")
    df = pd.read_csv(kline_path, parse_dates=['date'])
    df = df[(df['amount'] > 0) & (df['volume'] > 0) & (df['high'] > df['low'])].copy()
    
    print(f"数据: {df['date'].min()} ~ {df['date'].max()}, {df['stock_code'].nunique()} stocks")
    
    # CLV签名
    df['clv'] = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'])
    df['signed_vol'] = df['clv'] * df['volume']
    
    df = df.sort_values(['stock_code', 'date'])
    
    # 滚动加权OIB
    print(f"计算 {window}日EWM OIB (halflife={halflife})...")
    
    records = []
    
    for code, g in df.groupby('stock_code'):
        g = g.sort_values('date').reset_index(drop=True)
        sv = g['signed_vol'].values
        vol = g['volume'].values
        amt = g['amount'].values
        dates = g['date'].values
        
        for i in range(window-1, len(g)):
            # 取 [i-window+1, i] 的数据
            sv_win = sv[i-window+1:i+1]
            vol_win = vol[i-window+1:i+1]
            amt_win = amt[i-window+1:i+1]
            
            # 指数衰减权重
            w = exponential_weights(window, halflife)
            
            oib = np.sum(w * sv_win) / (np.sum(w * vol_win) + 1e-10)
            log_amt = np.log(np.mean(amt_win) + 1)
            
            records.append({
                'date': dates[i],
                'stock_code': code,
                'oib': oib,
                'log_amount': log_amt
            })
    
    result = pd.DataFrame(records)
    print(f"计算完成: {len(result)} rows")
    print(f"OIB 统计: mean={result['oib'].mean():.4f}, std={result['oib'].std():.4f}")
    
    # 翻转方向（卖出不平衡 → 正因子，反转逻辑）
    result['oib_neg'] = -result['oib']
    
    # 截面 rank 变换 + 中性化
    print("截面中性化...")
    factor_records = []
    
    for dt, day_df in result.groupby('date'):
        if len(day_df) < 50:
            continue
        
        # rank transform (更稳健)
        y_raw = day_df['oib_neg'].values.astype(float)
        x = day_df['log_amount'].values.astype(float)
        
        valid = np.isfinite(y_raw) & np.isfinite(x)
        if valid.sum() < 50:
            continue
        
        y_v = y_raw[valid]
        x_v = x[valid]
        
        # Rank transform
        from scipy.stats import rankdata
        y_rank = rankdata(y_v) / (len(y_v) + 1)  # uniform [0,1]
        
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
    
    return factor_df


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kline_path = os.path.join(base_dir, 'data', 'csi1000_kline_raw.csv')
    output_path = os.path.join(base_dir, 'data', 'factor_oib_ewm_v2.csv')
    
    build_factor(kline_path, output_path, window=60, halflife=20)
