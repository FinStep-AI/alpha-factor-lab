#!/usr/bin/env python3
"""
因子：跳空/日内振幅比 (Gap-to-Range Ratio)
ID: gap_range_ratio_v1

构造：
  mean(|open/prev_close - 1| / ((high-low)/prev_close + 0.001), 20d)
  对数变换 + 成交额OLS中性化 + MAD winsorize + z-score

逻辑：
  高值 = 隔夜跳空大、日内波动小 = 信息主要在盘后/集合竞价释放 = 机构主导
  低值 = 隔夜跳空小、日内波动大 = 日内散户博弈为主
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def compute_factor(kline):
    kline = kline.sort_values(['stock_code', 'date']).copy()
    results = []
    
    for code, grp in kline.groupby('stock_code'):
        grp = grp.sort_values('date').copy()
        
        prev_close = grp['close'].shift(1)
        gap = np.abs(grp['open'].values / prev_close.values - 1)
        intra_range = (grp['high'].values - grp['low'].values) / prev_close.values + 0.001
        
        ratio = pd.Series(gap / intra_range, index=grp.index)
        factor_raw = ratio.rolling(20, min_periods=10).mean()
        
        sub = grp[['date', 'stock_code']].copy()
        sub['raw_factor'] = factor_raw.values
        results.append(sub)
    
    return pd.concat(results, ignore_index=True)


def neutralize(df, kline):
    df = df.copy()
    kline_amt = kline[['date', 'stock_code', 'amount']].copy()
    kline_amt['date'] = pd.to_datetime(kline_amt['date'])
    df['date'] = pd.to_datetime(df['date'])
    
    kline_amt = kline_amt.sort_values(['stock_code', 'date'])
    kline_amt['log_amount_20d'] = kline_amt.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    
    df = df.merge(kline_amt[['date', 'stock_code', 'log_amount_20d']], on=['date', 'stock_code'], how='left')
    df['log_factor'] = np.log(df['raw_factor'].clip(lower=1e-10))
    
    output = []
    for dt, cs in df.groupby('date'):
        cs = cs.dropna(subset=['log_factor', 'log_amount_20d']).copy()
        if len(cs) < 50:
            continue
        
        X = np.column_stack([np.ones(len(cs)), cs['log_amount_20d'].values])
        y = cs['log_factor'].values
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residual = y - X @ beta
        except:
            continue
        
        cs['factor_neutral'] = residual
        
        med = np.nanmedian(cs['factor_neutral'])
        mad = np.nanmedian(np.abs(cs['factor_neutral'] - med))
        if mad < 1e-10:
            cs['factor'] = cs['factor_neutral']
        else:
            upper = med + 5.0 * 1.4826 * mad
            lower = med - 5.0 * 1.4826 * mad
            cs['factor'] = cs['factor_neutral'].clip(lower=lower, upper=upper)
        
        mean = cs['factor'].mean()
        std = cs['factor'].std()
        cs['factor'] = (cs['factor'] - mean) / std if std > 1e-10 else 0.0
        
        output.append(cs[['date', 'stock_code', 'factor']])
    
    return pd.concat(output, ignore_index=True)


if __name__ == '__main__':
    print("加载数据...")
    kline = pd.read_csv('data/csi1000_kline_raw.csv')
    kline['date'] = pd.to_datetime(kline['date'])
    
    print("计算跳空/日内振幅比...")
    raw = compute_factor(kline)
    
    print("中性化 + 标准化...")
    final = neutralize(raw, kline)
    
    output_path = 'data/factor_gap_range_ratio.csv'
    final.to_csv(output_path, index=False)
    
    print(f"因子计算完成! 保存到 {output_path}")
    print(f"  日期范围: {final['date'].min()} ~ {final['date'].max()}")
    print(f"  股票数: {final['stock_code'].nunique()}")
    print(f"  总记录: {len(final)}")
    print(f"  因子统计: mean={final['factor'].mean():.4f}, std={final['factor'].std():.4f}")
    
    # Also output negated version
    final_neg = final.copy()
    final_neg['factor'] = -final_neg['factor']
    final_neg.to_csv('data/factor_gap_range_ratio_neg.csv', index=False)
    print("  同时保存了反向因子: data/factor_gap_range_ratio_neg.csv")
