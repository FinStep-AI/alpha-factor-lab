#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
隔夜收益一致性因子 (Overnight Return Consistency)

思路来源:
  - Lou, Polk, Skouras (2019) "A Tug of War: Overnight vs Intraday Returns" JFE
  - 我们已有 overnight_momentum_v1 (隔夜均值)
  - 新维度: 隔夜收益的**方向一致性** = 正隔夜收益天数占比
  
逻辑:
  - A股隔夜收益主要反映知情交易者/机构的信息消化
  - 如果一只股票连续多天隔夜上涨(即使每天涨幅小)，说明有持续的正面信息流入
  - 这种信息渗透是渐进的(underreaction)，后续会继续上涨
  - 与隔夜均值不同：一致性衡量的是信号的可靠度，不是强度

构造:
  1. overnight_ret = open_t / close_{t-1} - 1
  2. oc_pos_ratio_20d = mean(overnight_ret > 0, 20d) — 近20天正隔夜收益占比
  3. 成交额 OLS 中性化 + MAD winsorize + z-score

也测试变体:
  - oc_signed: sum(sign(overnight_ret), 20d) / 20 — 等效但更直观
  - oc_weighted: 用量加权的隔夜收益一致性
"""

import numpy as np
import pandas as pd
import sys

def calc_overnight_consistency(kline_path, output_path, window=20):
    print(f"📥 读取: {kline_path}")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # 隔夜收益: open_t / close_{t-1} - 1
    df['prev_close'] = df.groupby('stock_code')['close'].shift(1)
    df['overnight_ret'] = df['open'] / df['prev_close'] - 1
    
    # 隔夜方向: 1 if positive, 0 if zero/negative
    df['overnight_pos'] = (df['overnight_ret'] > 0).astype(float)
    
    # 滚动窗口: 正隔夜收益占比
    df['oc_pos_ratio'] = df.groupby('stock_code')['overnight_pos'].transform(
        lambda x: x.rolling(window, min_periods=int(window * 0.6)).mean()
    )
    
    print(f"📊 隔夜收益统计:")
    print(f"  均值: {df['overnight_ret'].mean():.6f}")
    print(f"  正比例: {(df['overnight_ret'] > 0).mean():.3f}")
    print(f"  负比例: {(df['overnight_ret'] < 0).mean():.3f}")
    
    # 也计算一个变体: 隔夜收益的均值/标准差 (隔夜IR)
    df['oc_mean'] = df.groupby('stock_code')['overnight_ret'].transform(
        lambda x: x.rolling(window, min_periods=int(window * 0.6)).mean()
    )
    df['oc_std'] = df.groupby('stock_code')['overnight_ret'].transform(
        lambda x: x.rolling(window, min_periods=int(window * 0.6)).std()
    )
    df['oc_ir'] = df['oc_mean'] / df['oc_std'].replace(0, np.nan)
    
    # 输出主因子: oc_pos_ratio
    factor_raw = df[['date', 'stock_code', 'oc_pos_ratio']].dropna().copy()
    factor_raw.columns = ['date', 'stock_code', 'factor_raw']
    
    print(f"📊 因子原始值:")
    print(f"  行数: {len(factor_raw)}")
    print(f"  均值: {factor_raw['factor_raw'].mean():.4f}")
    print(f"  标准差: {factor_raw['factor_raw'].std():.4f}")
    
    # 成交额中性化
    print("⚙️  成交额中性化...")
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    amt_df = df[['date', 'stock_code', 'log_amount_20d']].drop_duplicates()
    factor_raw = factor_raw.merge(amt_df, on=['date', 'stock_code'], how='left')
    
    def neutralize(group):
        y = group['factor_raw'].values
        x = group['log_amount_20d'].values
        valid = ~(np.isnan(y) | np.isnan(x))
        if valid.sum() < 30:
            group['factor'] = np.nan
            return group
        X = np.column_stack([np.ones(valid.sum()), x[valid]])
        try:
            beta = np.linalg.lstsq(X, y[valid], rcond=None)[0]
            resid = np.full(len(y), np.nan)
            resid[valid] = y[valid] - X @ beta
            group['factor'] = resid
        except:
            group['factor'] = np.nan
        return group
    
    factor_raw = factor_raw.groupby('date', group_keys=False).apply(neutralize)
    
    # MAD Winsorize + Z-score
    print("⚙️  Winsorize + Z-score...")
    def winsorize_zscore(group):
        vals = group['factor'].values.copy()
        valid = ~np.isnan(vals)
        if valid.sum() < 10:
            group['factor'] = np.nan
            return group
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals[valid] - med))
        if mad > 1e-10:
            lower = med - 5 * 1.4826 * mad
            upper = med + 5 * 1.4826 * mad
            vals = np.clip(vals, lower, upper)
        mu = np.nanmean(vals)
        std = np.nanstd(vals)
        if std < 1e-10:
            group['factor'] = 0.0
        else:
            group['factor'] = (vals - mu) / std
        return group
    
    factor_raw = factor_raw.groupby('date', group_keys=False).apply(winsorize_zscore)
    
    output = factor_raw[['date', 'stock_code', 'factor']].dropna()
    output = output.sort_values(['date', 'stock_code'])
    output.to_csv(output_path, index=False)
    
    print(f"\n✅ 保存: {output_path}")
    print(f"  行数: {len(output)}, 日期: {output['date'].nunique()}")
    print(f"  范围: {output['date'].min()} ~ {output['date'].max()}")
    
    # 同时输出隔夜IR变体
    ir_output_path = output_path.replace('.csv', '_ir.csv')
    ir_raw = df[['date', 'stock_code', 'oc_ir']].dropna().copy()
    ir_raw.columns = ['date', 'stock_code', 'factor_raw']
    ir_raw = ir_raw.merge(amt_df, on=['date', 'stock_code'], how='left')
    ir_raw = ir_raw.groupby('date', group_keys=False).apply(neutralize)
    ir_raw = ir_raw.groupby('date', group_keys=False).apply(winsorize_zscore)
    ir_out = ir_raw[['date', 'stock_code', 'factor']].dropna()
    ir_out.to_csv(ir_output_path, index=False)
    print(f"  变体(隔夜IR): {ir_output_path}, {len(ir_out)} rows")
    
    return output

if __name__ == '__main__':
    kline = sys.argv[1] if len(sys.argv) > 1 else 'data/csi1000_kline_raw.csv'
    out = sys.argv[2] if len(sys.argv) > 2 else 'data/factor_overnight_consistency_v1.csv'
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    calc_overnight_consistency(kline, out, window)
