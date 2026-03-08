#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAX Effect Factor (max_ret_v1)

复现论文: Bali, Cakici & Whitelaw (2011)
"Maxing Out: Stocks as Lotteries and the Cross-Section of Expected Returns"
Journal of Financial Economics, 99(2), 427-446.

原文发现:
- 过去一个月最大日收益率(MAX)与未来收益呈负相关
- 投资者有彩票偏好(lottery preference)，追逐极端正收益的股票
- 导致高MAX股票被高估，后续表现差
- 做空高MAX + 做多低MAX → 显著正超额收益

本土化:
- MAX = max(daily_return, past 20 trading days)
- 反向使用：做多低MAX（理性定价），做空高MAX（彩票股高估）
- 市值中性化
- 股票池：中证1000

扩展: 也计算 MIN (最小日收益率) 作为对照
"""

import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def compute_max_factor(kline_path, output_path, window=20, use_min=False):
    """
    MAX/MIN Factor
    MAX_t = max(r_{t-N+1}, ..., r_t)  过去N日最大日收益率
    MIN_t = min(r_{t-N+1}, ..., r_t)  过去N日最小日收益率
    
    反向使用: -MAX (做多低MAX股票)
    """
    label = "MIN" if use_min else "MAX"
    print(f"{'='*60}")
    print(f"{label} Effect Factor (Bali et al. 2011 JFE, window={window})")
    print(f"{'='*60}")
    
    # 1. Load
    print("\n[1] 加载数据...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    print(f"    股票: {df['stock_code'].nunique()}, 日期: {df['date'].nunique()}")
    
    # 2. Daily returns
    print("[2] 计算日收益率...")
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    
    # 3. Rolling MAX or MIN
    print(f"[3] 计算{window}日滚动{label}...")
    if use_min:
        df['raw_factor'] = df.groupby('stock_code')['ret'].transform(
            lambda x: x.rolling(window, min_periods=window//2).min()
        )
        # MIN is already negative for bad stocks; use as-is (做多极端下跌的均值回复)
    else:
        df['raw_factor'] = df.groupby('stock_code')['ret'].transform(
            lambda x: x.rolling(window, min_periods=window//2).max()
        )
        # Reverse: -MAX (做多低MAX，做空高MAX)
        df['raw_factor'] = -df['raw_factor']
    
    # 4. Market cap neutralization
    print("[4] 市值中性化...")
    df['log_amount'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    
    def neutralize(group):
        y = group['raw_factor']
        x = group['log_amount']
        mask = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            return pd.Series(np.nan, index=group.index)
        y_c, x_c = y[mask].values, x[mask].values
        X = np.column_stack([np.ones(len(x_c)), x_c])
        try:
            beta = np.linalg.lstsq(X, y_c, rcond=None)[0]
            res = y_c - X @ beta
            out = pd.Series(np.nan, index=group.index)
            out[mask] = res
            return out
        except:
            return pd.Series(np.nan, index=group.index)
    
    df['factor'] = df.groupby('date', group_keys=False).apply(neutralize)
    
    # 5. Winsorize MAD 3x
    print("[5] Winsorize (MAD 3x)...")
    def winsorize_mad(group):
        v = group['factor']; m = v.notna()
        if m.sum() < 10: return v
        c = v[m]; med = c.median(); mad = np.median(np.abs(c - med))
        if mad < 1e-10: return v
        return v.clip(med - 3*1.4826*mad, med + 3*1.4826*mad)
    
    df['factor'] = df.groupby('date', group_keys=False).apply(winsorize_mad)
    
    # 6. Z-score
    print("[6] Z-score标准化...")
    def zscore(group):
        v = group['factor']; m = v.notna()
        if m.sum() < 10: return v
        mu, s = v[m].mean(), v[m].std()
        if s < 1e-10: return v * 0
        return (v - mu) / s
    
    df['factor'] = df.groupby('date', group_keys=False).apply(zscore)
    
    # 7. Output
    output = df[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    output = output.rename(columns={'factor': 'factor_value'})
    output.to_csv(output_path, index=False)
    
    print(f"\n✅ {label} factor saved: {output_path}")
    print(f"   Records: {len(output):,}, Dates: {output['date'].nunique()}")
    return output


def compute_max_min_diff(kline_path, output_path, window=20):
    """
    MAX-MIN差异因子: -(MAX + MIN) / 2 的变体
    实际上: -(MAX - |MIN|) 衡量收益率分布的不对称性
    高值 = 下跌极端 > 上涨极端 → 后续均值回复
    """
    print(f"{'='*60}")
    print(f"MAX-MIN Asymmetry Factor (window={window})")
    print(f"{'='*60}")
    
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    
    # MAX and MIN
    df['max_ret'] = df.groupby('stock_code')['ret'].transform(
        lambda x: x.rolling(window, min_periods=window//2).max()
    )
    df['min_ret'] = df.groupby('stock_code')['ret'].transform(
        lambda x: x.rolling(window, min_periods=window//2).min()
    )
    
    # Asymmetry: -(MAX + MIN)
    # When MAX is large and MIN is small (near 0), this is very negative → sell
    # When MAX is small and MIN is very negative, this is positive → buy
    df['raw_factor'] = -(df['max_ret'] + df['min_ret'])
    
    # Neutralize, winsorize, zscore
    df['log_amount'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    
    def neutralize(group):
        y = group['raw_factor']; x = group['log_amount']
        mask = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            return pd.Series(np.nan, index=group.index)
        y_c, x_c = y[mask].values, x[mask].values
        X = np.column_stack([np.ones(len(x_c)), x_c])
        try:
            beta = np.linalg.lstsq(X, y_c, rcond=None)[0]
            out = pd.Series(np.nan, index=group.index)
            out[mask] = y_c - X @ beta
            return out
        except:
            return pd.Series(np.nan, index=group.index)
    
    df['factor'] = df.groupby('date', group_keys=False).apply(neutralize)
    
    def winsorize_mad(group):
        v = group['factor']; m = v.notna()
        if m.sum() < 10: return v
        c = v[m]; med = c.median(); mad = np.median(np.abs(c - med))
        if mad < 1e-10: return v
        return v.clip(med - 3*1.4826*mad, med + 3*1.4826*mad)
    
    df['factor'] = df.groupby('date', group_keys=False).apply(winsorize_mad)
    
    def zscore(group):
        v = group['factor']; m = v.notna()
        if m.sum() < 10: return v
        mu, s = v[m].mean(), v[m].std()
        if s < 1e-10: return v * 0
        return (v - mu) / s
    
    df['factor'] = df.groupby('date', group_keys=False).apply(zscore)
    
    output = df[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    output = output.rename(columns={'factor': 'factor_value'})
    output.to_csv(output_path, index=False)
    
    print(f"\n✅ MAX-MIN Asymmetry factor saved: {output_path}")
    print(f"   Records: {len(output):,}, Dates: {output['date'].nunique()}")
    return output


if __name__ == "__main__":
    kline = "data/csi1000_kline_raw.csv"
    
    # 1. MAX effect (反向: 做多低MAX)
    compute_max_factor(kline, "data/factor_max_ret_neg_20d.csv", window=20, use_min=False)
    
    # 2. MAX effect positive (正向: 做多高MAX - A股散户可能方向不同?)
    # Just reverse the negative 
    
    # 3. MIN effect (做多极端下跌 - 均值回复)
    compute_max_factor(kline, "data/factor_min_ret_20d.csv", window=20, use_min=True)
    
    # 4. MAX-MIN asymmetry
    compute_max_min_diff(kline, "data/factor_max_min_asym_20d.csv", window=20)
    
    print("\n✅ All MAX/MIN variants computed!")
