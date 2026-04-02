#!/usr/bin/env python3
"""
因子: Turnover-Intensity Reversal (换手强度反转) v1

论文基础: 
- Nagel (2012) "Evaporating Liquidity" - 短期反转与流动性的交互
- Campbell, Grossman & Wang (1993) "Trading Volume and Serial Correlation" 
  - 放量下跌后反转更强(流动性供给者获得溢价)

公式: -return_5d × relative_turnover_5d
  其中 relative_turnover_5d = mean(turnover, 5d) / mean(turnover, 60d)

正因子值 = 近期跌 × 换手放大 → 恐慌卖出后反弹信号
"""

import numpy as np
import pandas as pd
import sys

def calc_turnover_intensity_reversal(kline_path, output_path, ret_window=5, to_short=5, to_long=60):
    print(f"[1/5] 加载数据...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    print(f"  数据量: {len(df):,} 行, {df['stock_code'].nunique()} 只股票")
    
    print(f"[2/5] 计算换手强度反转因子...")
    
    results = []
    for code, grp in df.groupby('stock_code'):
        grp = grp.sort_values('date').copy()
        
        # 5日累计收益
        log_ret = np.log(grp['close'] / grp['close'].shift(1))
        cum_ret_5d = log_ret.rolling(ret_window, min_periods=ret_window).sum()
        
        # 短期换手 / 长期换手 (相对换手率)
        to_ma_short = grp['turnover'].rolling(to_short, min_periods=to_short).mean()
        to_ma_long = grp['turnover'].rolling(to_long, min_periods=to_long//2).mean()
        relative_to = to_ma_short / to_ma_long.replace(0, np.nan)
        
        # 因子 = -收益 × 相对换手率
        # 跌幅大(负收益) × 放量(高相对换手) → 正值 → 做多
        factor_raw = -cum_ret_5d * relative_to
        
        # 中性化变量
        log_amount_20d = np.log(grp['amount'].rolling(20, min_periods=10).mean() + 1)
        
        grp_result = grp[['date', 'stock_code']].copy()
        grp_result['raw_factor'] = factor_raw.values
        grp_result['log_amount_20d'] = log_amount_20d.values
        results.append(grp_result)
    
    df_factor = pd.concat(results, ignore_index=True)
    df_factor = df_factor.dropna(subset=['raw_factor', 'log_amount_20d'])
    
    print(f"  有效因子值: {df_factor['raw_factor'].notna().sum():,}")
    print(f"  因子统计: mean={df_factor['raw_factor'].mean():.6f}, std={df_factor['raw_factor'].std():.6f}")
    
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
        
        # OLS中性化
        X = day_df['log_amount_20d'].values
        mask = np.isfinite(X) & np.isfinite(vals.values)
        if mask.sum() < 50:
            continue
        
        X_clean = X[mask]
        y_clean = vals.values[mask]
        
        X_mat = np.column_stack([np.ones(len(X_clean)), X_clean])
        try:
            beta = np.linalg.lstsq(X_mat, y_clean, rcond=None)[0]
            residuals = y_clean - X_mat @ beta
        except:
            continue
        
        std = residuals.std()
        if std > 0:
            z = (residuals - residuals.mean()) / std
        else:
            continue
        
        day_result = day_df.iloc[np.where(mask)[0]][['date', 'stock_code']].copy()
        day_result['factor_value'] = z
        factor_values.append(day_result)
    
    df_out = pd.concat(factor_values, ignore_index=True)
    
    print(f"[4/5] 输出因子...")
    print(f"  总行数: {len(df_out):,}")
    print(f"  因子统计: mean={df_out['factor_value'].mean():.4f}, std={df_out['factor_value'].std():.4f}")
    
    df_out.to_csv(output_path, index=False)
    print(f"  保存到: {output_path}")
    
    print(f"[5/5] 完成!")
    return df_out

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    parser.add_argument('--output', default='data/factor_to_intensity_rev_v1.csv')
    args = parser.parse_args()
    
    calc_turnover_intensity_reversal(args.kline, args.output)
