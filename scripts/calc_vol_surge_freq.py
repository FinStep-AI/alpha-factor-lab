#!/usr/bin/env python3
"""
因子: Volume Surge Frequency (成交量激增频率) v1

论文基础: Gervais, Kaniel & Mingelgrin (2001) "The High-Volume Return Premium", JF
核心思路: 过去N日中成交量>自身MA(N)×阈值的天数占比
离散化因子, 参考neg_day_freq_v1的成功范式

做市值/成交额中性化后衡量纯粹的"交易事件频率"
"""

import numpy as np
import pandas as pd
import sys

def calc_vol_surge_freq(kline_path, output_path, window=20, threshold=1.5, neutralize_col='log_amount_20d'):
    """
    计算成交量激增频率因子
    
    Parameters:
    - window: 回看窗口 (默认20日)
    - threshold: 成交量激增阈值倍数 (默认1.5倍)
    - neutralize_col: 中性化变量
    """
    print(f"[1/5] 加载数据...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    print(f"  数据量: {len(df):,} 行, {df['stock_code'].nunique()} 只股票")
    print(f"  日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    
    # 确保volume列存在
    if 'volume' not in df.columns:
        print("[错误] 数据中无volume列")
        sys.exit(1)
    
    print(f"[2/5] 计算成交量激增频率 (window={window}, threshold={threshold}x)...")
    
    # 每只股票独立计算
    results = []
    for code, grp in df.groupby('stock_code'):
        grp = grp.sort_values('date').copy()
        
        # 20日均量
        ma_vol = grp['volume'].rolling(window, min_periods=window//2).mean()
        
        # 是否高于阈值
        is_surge = (grp['volume'] > ma_vol * threshold).astype(float)
        
        # 高量日频率
        surge_freq = is_surge.rolling(window, min_periods=window//2).mean()
        
        # 用于中性化的变量: 20日平均成交额(对数)
        log_amount_20d = np.log(grp['amount'].rolling(20, min_periods=10).mean() + 1)
        
        # 20日平均市值代理(用成交额)
        grp_result = grp[['date', 'stock_code']].copy()
        grp_result['raw_factor'] = surge_freq
        grp_result['log_amount_20d'] = log_amount_20d
        results.append(grp_result)
    
    df_factor = pd.concat(results, ignore_index=True)
    df_factor = df_factor.dropna(subset=['raw_factor', 'log_amount_20d'])
    
    print(f"  有效因子值: {df_factor['raw_factor'].notna().sum():,}")
    print(f"  因子统计: mean={df_factor['raw_factor'].mean():.4f}, std={df_factor['raw_factor'].std():.4f}")
    print(f"  因子分布: min={df_factor['raw_factor'].min():.4f}, median={df_factor['raw_factor'].median():.4f}, max={df_factor['raw_factor'].max():.4f}")
    
    print(f"[3/5] MAD Winsorize + 成交额OLS中性化 + Z-score...")
    
    factor_values = []
    for date, day_df in df_factor.groupby('date'):
        if len(day_df) < 50:
            continue
        
        vals = day_df['raw_factor'].copy()
        
        # MAD winsorize
        median = vals.median()
        mad = (vals - median).abs().median()
        if mad > 0:
            upper = median + 5 * 1.4826 * mad
            lower = median - 5 * 1.4826 * mad
            vals = vals.clip(lower, upper)
        
        # OLS中性化 (对成交额回归取残差)
        X = day_df['log_amount_20d'].values
        mask = np.isfinite(X) & np.isfinite(vals.values)
        if mask.sum() < 50:
            continue
        
        X_clean = X[mask]
        y_clean = vals.values[mask]
        
        # OLS: y = a + b*X + residual
        X_mat = np.column_stack([np.ones(len(X_clean)), X_clean])
        try:
            beta = np.linalg.lstsq(X_mat, y_clean, rcond=None)[0]
            residuals = y_clean - X_mat @ beta
        except:
            continue
        
        # Z-score
        std = residuals.std()
        if std > 0:
            z = (residuals - residuals.mean()) / std
        else:
            continue
        
        # 回填
        day_result = day_df.iloc[np.where(mask)[0]][['date', 'stock_code']].copy()
        day_result['factor_value'] = z
        factor_values.append(day_result)
    
    df_out = pd.concat(factor_values, ignore_index=True)
    
    print(f"[4/5] 输出因子...")
    print(f"  总行数: {len(df_out):,}")
    print(f"  日期范围: {df_out['date'].min()} ~ {df_out['date'].max()}")
    print(f"  因子统计: mean={df_out['factor_value'].mean():.4f}, std={df_out['factor_value'].std():.4f}")
    
    df_out.to_csv(output_path, index=False)
    print(f"  保存到: {output_path}")
    
    print(f"[5/5] 完成!")
    return df_out

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    parser.add_argument('--output', default='data/factor_vol_surge_freq_v1.csv')
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=1.5)
    args = parser.parse_args()
    
    calc_vol_surge_freq(args.kline, args.output, args.window, args.threshold)
