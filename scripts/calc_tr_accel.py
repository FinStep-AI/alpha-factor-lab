#!/usr/bin/env python3
"""
Turnover Acceleration Factor
==============================
构造: log(MA5_turnover) - log(MA60_turnover)
     = log(MA5_turnover / MA60_turnover)
     正向: 近期换手率加速 → 关注度急升 → 动量延续

注意: turnover_ratio_v1(MA5/MA60反向)已经测试过(IC=0.020,mono=0.7)
      那次是反向使用(低比值做多),这次试正向。
      而且turnover_trend_v1(线性斜率)也失败了(IC=-0.017方向反)

但那次的IC=0.020(t=2.68)几乎达标！问题是mono=0.7不足。
让我试试把窗口调整：MA10/MA40, 成交额中性化。

核心创新: 
  1. 用MA10/MA40而非MA5/MA60（更稳定）
  2. 正向使用（高换手加速做多）
  3. 也尝试10d forward

额外尝试: "换手率离差因子" 
  = MA20_turnover - MA60_turnover（绝对差而非比值）
  可能比值过于极端而绝对差更稳定
"""

import numpy as np
import pandas as pd
import os

def compute_turnover_accel(df_kline, short_w=10, long_w=40):
    """换手率加速因子"""
    df = df_kline.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date'])
    
    results = []
    for code, group in df.groupby('stock_code'):
        group = group.sort_values('date').reset_index(drop=True)
        n = len(group)
        turnover = group['turnover'].values
        amounts = group['amount'].values
        dates = group['date'].values
        
        for i in range(long_w, n):
            short_vals = turnover[i-short_w+1:i+1]
            long_vals = turnover[i-long_w+1:i+1]
            
            ma_short = np.nanmean(short_vals)
            ma_long = np.nanmean(long_vals)
            
            if ma_short <= 0 or ma_long <= 0 or np.isnan(ma_short) or np.isnan(ma_long):
                continue
            
            # 比值
            ratio = np.log(ma_short / ma_long)
            
            # 60日平均成交额
            amt_vals = amounts[max(0,i-59):i+1]
            log_amt = np.log(np.nanmean(amt_vals) + 1)
            
            results.append({
                'date': dates[i],
                'stock_code': code,
                'tr_accel_raw': ratio,
                'log_amount': log_amt
            })
    
    return pd.DataFrame(results)


def neutralize_zscore(df, fcol, ncol='log_amount'):
    """OLS中性化 + MAD + z-score"""
    outs = []
    for date, g in df.groupby('date'):
        g = g.dropna(subset=[fcol, ncol]).copy()
        if len(g) < 50:
            continue
        y = g[fcol].values
        x = np.column_stack([np.ones(len(g)), g[ncol].values])
        try:
            b = np.linalg.lstsq(x, y, rcond=None)[0]
            r = y - x @ b
        except:
            continue
        med, mad = np.median(r), np.median(np.abs(r - np.median(r)))
        if mad < 1e-10: continue
        r = np.clip(r, med - 5*1.4826*mad, med + 5*1.4826*mad)
        s = np.std(r)
        if s < 1e-10: continue
        z = np.clip((r - np.mean(r)) / s, -3, 3)
        g = g.copy()
        g['factor'] = z
        outs.append(g[['date', 'stock_code', 'factor']])
    return pd.concat(outs) if outs else pd.DataFrame()


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    df = pd.read_csv(os.path.join(data_dir, 'csi1000_kline_raw.csv'))
    print(f"Loaded {len(df)} rows, {df['stock_code'].nunique()} stocks")
    
    # === Variant 1: MA10/MA40 ===
    print("\nComputing TR acceleration: log(MA10/MA40)...")
    raw = compute_turnover_accel(df, short_w=10, long_w=40)
    print(f"  Raw: {len(raw)}")
    
    # 正向
    fac = neutralize_zscore(raw, 'tr_accel_raw')
    fac.to_csv(os.path.join(data_dir, 'factor_tr_accel_v1.csv'), index=False)
    print(f"  Positive saved: {len(fac)} rows")
    
    # 翻转
    fac_neg = fac.copy()
    fac_neg['factor'] = -fac_neg['factor']
    fac_neg.to_csv(os.path.join(data_dir, 'factor_tr_accel_neg_v1.csv'), index=False)
    print(f"  Negative saved")
    
    print(f"  Dates: {fac['date'].min()} ~ {fac['date'].max()}")
    print(f"  Stocks/date: {fac.groupby('date')['stock_code'].count().mean():.0f}")

if __name__ == '__main__':
    main()
