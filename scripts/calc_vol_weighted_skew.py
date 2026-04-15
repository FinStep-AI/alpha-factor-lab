#!/usr/bin/env python3
"""
因子：成交额加权收益偏度 (Volume-Weighted Return Skewness)
ID: vol_w_skew_v1

逻辑：
  普通偏度=每天同等权重。但不同天信息含量不同。
  成交额加权：高成交额天对偏度贡献更大。
  - 正偏度 → 放量时涨>放量时跌 → 资金推升意愿强 → 正向信号
  - 负偏度 → 放量时跌>放量时涨 → 资金出逃 → 负向信号

公式：
  对每只股票过去20天：
    w_t = amount_t / sum(amount)  (成交额权重)
    weighted_mean = sum(w * ret)
    weighted_var = sum(w * (ret - wmean)^2)
    weighted_skew = sum(w * (ret - wmean)^3) / weighted_var^1.5

  然后做成交额OLS中性化 + MAD winsorize + z-score
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

WINDOW = 20

def calc_vol_weighted_skew(df_stock):
    """对单只股票计算滚动成交额加权偏度"""
    ret = df_stock['pct_change'].values / 100.0  # 转为小数
    amt = df_stock['amount'].values
    dates = df_stock['date'].values
    n = len(ret)
    
    results = []
    for i in range(WINDOW - 1, n):
        r = ret[i - WINDOW + 1: i + 1]
        a = amt[i - WINDOW + 1: i + 1]
        
        # 过滤NaN
        mask = ~(np.isnan(r) | np.isnan(a) | (a <= 0))
        if mask.sum() < 10:  # 至少10天有效数据
            results.append((dates[i], np.nan))
            continue
        
        r_valid = r[mask]
        a_valid = a[mask]
        
        # 成交额权重
        w = a_valid / a_valid.sum()
        
        # 加权均值
        wmean = np.sum(w * r_valid)
        
        # 加权方差
        wvar = np.sum(w * (r_valid - wmean) ** 2)
        
        if wvar < 1e-16:
            results.append((dates[i], np.nan))
            continue
        
        # 加权偏度
        wskew = np.sum(w * (r_valid - wmean) ** 3) / (wvar ** 1.5)
        
        results.append((dates[i], wskew))
    
    return results


def neutralize_ols(df, factor_col, neutral_col):
    """OLS中性化"""
    from numpy.linalg import lstsq
    result = df[factor_col].copy()
    for date in df['date'].unique():
        mask = df['date'] == date
        sub = df.loc[mask, [factor_col, neutral_col]].dropna()
        if len(sub) < 30:
            continue
        y = sub[factor_col].values
        X = np.column_stack([np.ones(len(y)), sub[neutral_col].values])
        try:
            beta, _, _, _ = lstsq(X, y, rcond=None)
            residual = y - X @ beta
            result.loc[sub.index] = residual
        except:
            pass
    return result


def mad_winsorize(s, n_mad=5):
    """MAD winsorize"""
    median = s.median()
    mad = (s - median).abs().median()
    if mad < 1e-10:
        return s
    upper = median + n_mad * 1.4826 * mad
    lower = median - n_mad * 1.4826 * mad
    return s.clip(lower, upper)


def zscore(s):
    """z-score标准化"""
    mu = s.mean()
    sd = s.std()
    if sd < 1e-10:
        return s * 0
    return (s - mu) / sd


def main():
    base = Path(__file__).resolve().parent.parent
    data_dir = base / 'data'
    
    print("读取K线数据...")
    df = pd.read_csv(data_dir / 'csi1000_kline_raw.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # 过滤pct_change为NaN的行
    df = df.dropna(subset=['pct_change'])
    
    print(f"股票数: {df['stock_code'].nunique()}, 行数: {len(df)}")
    
    # 计算每只股票的滚动成交额加权偏度
    all_results = []
    stocks = df['stock_code'].unique()
    print(f"计算成交额加权偏度 (窗口={WINDOW})...")
    
    for i, code in enumerate(stocks):
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(stocks)}")
        sub = df[df['stock_code'] == code].copy()
        if len(sub) < WINDOW:
            continue
        res = calc_vol_weighted_skew(sub)
        for date, val in res:
            all_results.append({'date': date, 'stock_code': code, 'factor': val})
    
    factor_df = pd.DataFrame(all_results)
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    print(f"因子记录数: {len(factor_df)}, 非NaN: {factor_df['factor'].notna().sum()}")
    
    # 计算20日平均成交额用于中性化
    print("计算20日平均成交额...")
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    
    # 合并
    factor_df = factor_df.merge(
        df[['date', 'stock_code', 'log_amount_20d']],
        on=['date', 'stock_code'],
        how='left'
    )
    
    # OLS中性化
    print("成交额OLS中性化...")
    factor_df['factor'] = neutralize_ols(factor_df, 'factor', 'log_amount_20d')
    
    # 按日MAD winsorize + z-score
    print("MAD winsorize + z-score...")
    factor_df['factor'] = factor_df.groupby('date')['factor'].transform(
        lambda x: zscore(mad_winsorize(x))
    )
    
    # 输出
    output = factor_df[['date', 'stock_code', 'factor']].copy()
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')
    output = output.dropna(subset=['factor'])
    
    out_path = data_dir / 'factor_vol_w_skew_v1.csv'
    output.to_csv(out_path, index=False)
    print(f"因子文件保存到: {out_path}")
    print(f"记录数: {len(output)}, 日期范围: {output['date'].min()} ~ {output['date'].max()}")
    print(f"因子统计: mean={output['factor'].mean():.4f}, std={output['factor'].std():.4f}")
    
    # 检查截面覆盖率
    date_counts = output.groupby('date')['stock_code'].count()
    print(f"每日平均覆盖股票数: {date_counts.mean():.0f}")
    

if __name__ == '__main__':
    main()
