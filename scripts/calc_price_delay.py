#!/usr/bin/env python3
"""
Price Delay Factor - Hou & Moskowitz (2005)
"Market Frictions, Price Delay, and the Cross-Section of Expected Returns"
Review of Financial Studies, 18(3), 981-1020.

公式:
  Restricted:   r_i,t = alpha + beta_0 * r_m,t + epsilon
  Unrestricted: r_i,t = alpha + beta_0 * r_m,t + sum(beta_k * r_m,t-k, k=1..K) + epsilon
  
  Price Delay D1 = 1 - R²_restricted / R²_unrestricted
  
高Delay = 价格对市场信息反应延迟 = 市场摩擦大 = 被忽视
正向使用: 高Delay → 高预期收益 (信息摩擦补偿)
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def compute_market_returns(df):
    """计算等权市场收益率(中证1000成分股等权)"""
    daily_ret = df.pivot_table(index='date', columns='stock_code', values='pct_change')
    mkt_ret = daily_ret.mean(axis=1)  # 等权
    return mkt_ret

def compute_price_delay(stock_ret, mkt_ret, window=60, n_lags=4, min_obs=40):
    """
    计算滚动窗口的Price Delay D1
    
    Parameters:
        stock_ret: 个股日收益率序列
        mkt_ret: 市场收益率序列
        window: 滚动窗口(交易日)
        n_lags: 滞后阶数
        min_obs: 最少观察数
    
    Returns:
        delay: Price Delay D1 序列
    """
    aligned = pd.DataFrame({'stock': stock_ret, 'mkt': mkt_ret}).dropna()
    
    if len(aligned) < window:
        return pd.Series(dtype=float, index=stock_ret.index)
    
    results = {}
    dates = aligned.index
    
    for i in range(window, len(dates)+1):
        end_date = dates[i-1]
        window_data = aligned.iloc[max(0, i-window):i]
        
        if len(window_data) < min_obs:
            continue
        
        y = window_data['stock'].values
        x_mkt = window_data['mkt'].values
        
        # Restricted model: r_i = alpha + beta_0 * r_m
        X_r = np.column_stack([np.ones(len(y)), x_mkt])
        try:
            beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
            resid_r = y - X_r @ beta_r
            ss_res_r = np.sum(resid_r**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            if ss_tot == 0:
                continue
            r2_r = 1 - ss_res_r / ss_tot
        except:
            continue
        
        # Unrestricted model: r_i = alpha + beta_0 * r_m + sum(beta_k * r_m_lag_k)
        # 构造滞后市场收益
        mkt_series = window_data['mkt']
        X_u_cols = [np.ones(len(y)), x_mkt]
        valid_start = n_lags
        
        for lag in range(1, n_lags+1):
            lagged = mkt_series.shift(lag).values
            X_u_cols.append(lagged)
        
        X_u = np.column_stack(X_u_cols)
        # 去掉前n_lags行(有NaN)
        mask = ~np.isnan(X_u).any(axis=1)
        X_u_clean = X_u[mask]
        y_clean = y[mask]
        
        if len(y_clean) < min_obs - n_lags:
            continue
        
        try:
            beta_u = np.linalg.lstsq(X_u_clean, y_clean, rcond=None)[0]
            resid_u = y_clean - X_u_clean @ beta_u
            ss_res_u = np.sum(resid_u**2)
            ss_tot_u = np.sum((y_clean - np.mean(y_clean))**2)
            if ss_tot_u == 0:
                continue
            r2_u = 1 - ss_res_u / ss_tot_u
        except:
            continue
        
        # Restricted model on same sample
        X_r_clean = np.column_stack([np.ones(len(y_clean)), X_u_clean[:, 1]])  # intercept + concurrent mkt
        try:
            beta_r2 = np.linalg.lstsq(X_r_clean, y_clean, rcond=None)[0]
            resid_r2 = y_clean - X_r_clean @ beta_r2
            ss_res_r2 = np.sum(resid_r2**2)
            r2_r_same = 1 - ss_res_r2 / ss_tot_u
        except:
            continue
        
        # Price Delay D1
        if r2_u > 0 and r2_r_same >= 0:
            delay = 1 - r2_r_same / max(r2_u, 1e-10)
            delay = max(0, min(delay, 1))  # clip to [0, 1]
            results[end_date] = delay
    
    return pd.Series(results)

def neutralize_ols(factor_df, control_df):
    """OLS中性化: 用控制变量回归残差"""
    result = factor_df.copy()
    for date in factor_df.index:
        if date not in control_df.index:
            continue
        y = factor_df.loc[date].dropna()
        x = control_df.loc[date].reindex(y.index).dropna()
        common = y.index.intersection(x.index)
        if len(common) < 30:
            continue
        y_c = y[common].values
        x_c = x[common].values.reshape(-1, 1)
        X = np.column_stack([np.ones(len(y_c)), x_c])
        try:
            beta = np.linalg.lstsq(X, y_c, rcond=None)[0]
            resid = y_c - X @ beta
            result.loc[date, common] = resid
        except:
            pass
    return result

def mad_winsorize(s, n_mad=5):
    """MAD winsorize"""
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return s
    upper = med + n_mad * 1.4826 * mad
    lower = med - n_mad * 1.4826 * mad
    return s.clip(lower, upper)

def zscore(s):
    """截面z-score"""
    std = s.std()
    if std == 0 or np.isnan(std):
        return s * 0
    return (s - s.mean()) / std

def main():
    print("=" * 60)
    print("Price Delay Factor - Hou & Moskowitz (2005)")
    print("=" * 60)
    
    # 加载数据
    print("\n[1/5] 加载数据...")
    df = pd.read_csv('data/csi1000_kline_raw.csv', parse_dates=['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # 清洗收益率
    df['ret'] = df['pct_change'] / 100.0  # 从百分比转为小数
    
    # 过滤异常
    df = df[df['ret'].notna() & (df['ret'].abs() < 0.11)]  # A股涨跌停10%+
    
    print(f"  数据范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  股票数: {df['stock_code'].nunique()}")
    print(f"  总行数: {len(df)}")
    
    # 计算市场收益
    print("\n[2/5] 计算市场收益率...")
    ret_pivot = df.pivot_table(index='date', columns='stock_code', values='ret')
    mkt_ret = ret_pivot.mean(axis=1)
    print(f"  交易日数: {len(mkt_ret)}")
    print(f"  市场日均收益: {mkt_ret.mean()*100:.4f}%")
    
    # 计算每只股票的Price Delay
    print("\n[3/5] 计算Price Delay (60日滚动窗口, 4阶滞后)...")
    stocks = ret_pivot.columns.tolist()
    delay_dict = {}
    
    for i, stk in enumerate(stocks):
        if (i+1) % 100 == 0:
            print(f"  进度: {i+1}/{len(stocks)}")
        
        stk_ret = ret_pivot[stk].dropna()
        if len(stk_ret) < 60:
            continue
        
        delay = compute_price_delay(stk_ret, mkt_ret, window=60, n_lags=4, min_obs=40)
        if len(delay) > 0:
            delay_dict[stk] = delay
    
    delay_df = pd.DataFrame(delay_dict)
    print(f"  Delay矩阵: {delay_df.shape[0]}日 × {delay_df.shape[1]}股")
    print(f"  Delay均值: {delay_df.stack().mean():.4f}")
    print(f"  Delay中位数: {delay_df.stack().median():.4f}")
    print(f"  Delay标准差: {delay_df.stack().std():.4f}")
    
    # 中性化处理
    print("\n[4/5] 中性化处理...")
    
    # 计算20日平均成交额(对数)作为控制变量
    amt_pivot = df.pivot_table(index='date', columns='stock_code', values='amount')
    log_amt_20d = np.log(amt_pivot.rolling(20, min_periods=10).mean())
    
    # OLS中性化(对数成交额)
    delay_neutral = neutralize_ols(delay_df, log_amt_20d)
    
    # MAD winsorize + z-score
    factor_final = delay_neutral.copy()
    for date in factor_final.index:
        row = factor_final.loc[date].dropna()
        if len(row) < 30:
            continue
        row = mad_winsorize(row)
        row = zscore(row)
        factor_final.loc[date, row.index] = row
    
    # 保存因子
    print("\n[5/5] 保存因子...")
    # 转为长格式
    factor_long = factor_final.stack().reset_index()
    factor_long.columns = ['date', 'stock_code', 'factor']
    factor_long = factor_long.dropna()
    factor_long = factor_long.sort_values(['date', 'stock_code'])
    
    output_path = 'data/factor_price_delay_v1.csv'
    factor_long.to_csv(output_path, index=False)
    print(f"  已保存: {output_path}")
    print(f"  行数: {len(factor_long)}")
    print(f"  日期范围: {factor_long['date'].min()} ~ {factor_long['date'].max()}")
    
    # 因子统计
    print("\n--- 因子统计 ---")
    print(f"  截面覆盖率: {factor_long.groupby('date')['factor'].count().mean():.0f}只/日")
    print(f"  因子均值: {factor_long['factor'].mean():.4f}")
    print(f"  因子标准差: {factor_long['factor'].std():.4f}")
    
    # 每个截面的非空比例
    coverage = factor_long.groupby('date')['factor'].count()
    print(f"  最小覆盖: {coverage.min()}只")
    print(f"  最大覆盖: {coverage.max()}只")
    
    print("\n✅ Price Delay因子计算完成!")

if __name__ == '__main__':
    main()
