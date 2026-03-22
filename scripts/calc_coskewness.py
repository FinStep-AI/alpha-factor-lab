#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coskewness 协偏度因子 (向量化版)

论文: Harvey & Siddique (2000) "Conditional Skewness in Asset Pricing Tests"
      Journal of Finance, 55(3), 1263-1295
      https://doi.org/10.1111/0022-1082.00247

核心公式:
  Coskewness_i = E[(r_i - μ_i)(r_m - μ_m)^2] / [σ_i × σ_m^2]

中性化: 成交额OLS中性化 + MAD winsorize + z-score
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")


def compute_coskewness_vectorized(kline_df, window=60, min_periods=30):
    """
    向量化计算协偏度。利用滚动窗口和矩阵运算加速。
    """
    kline = kline_df.sort_values(['stock_code', 'date']).copy()
    kline['ret'] = kline.groupby('stock_code')['close'].pct_change()
    
    # Pivot成宽表
    ret_wide = kline.pivot(index='date', columns='stock_code', values='ret')
    ret_wide = ret_wide.sort_index()
    
    dates = ret_wide.index
    stocks = ret_wide.columns
    ret_arr = ret_wide.values  # (T, N)
    T, N = ret_arr.shape
    
    # 市场收益 = 等权平均(忽略NaN)
    mkt_ret = np.nanmean(ret_arr, axis=1)  # (T,)
    
    # 输出矩阵
    coskew_arr = np.full((T, N), np.nan)
    
    print(f"  矩阵大小: {T} 天 × {N} 股")
    
    for i in range(window, T):
        if (i - window) % 100 == 0:
            print(f"  进度: {i-window+1}/{T-window}", flush=True)
        
        # 窗口内市场收益
        rm = mkt_ret[i-window:i]
        rm_mean = np.mean(rm)
        rm_std = np.std(rm, ddof=1)
        if rm_std < 1e-10:
            continue
        
        rm_dm = rm - rm_mean       # (W,)
        rm_dm_sq = rm_dm ** 2      # (W,)
        
        # 窗口内个股收益 (W, N)
        ri = ret_arr[i-window:i, :]
        
        # 有效数据数量
        valid_count = np.sum(~np.isnan(ri), axis=0)  # (N,)
        
        # 均值和标准差
        ri_mean = np.nanmean(ri, axis=0)    # (N,)
        ri_std = np.nanstd(ri, axis=0, ddof=1)  # (N,)
        
        # 去均值
        ri_dm = ri - ri_mean[np.newaxis, :]  # (W, N)
        
        # (r_i - mu_i) * (r_m - mu_m)^2
        # 将NaN替换为0来求和
        ri_dm_filled = np.where(np.isnan(ri_dm), 0, ri_dm)
        cross = ri_dm_filled * rm_dm_sq[:, np.newaxis]  # (W, N)
        cross_mean = np.sum(cross, axis=0) / np.maximum(valid_count, 1)  # (N,)
        
        # coskewness = cross_mean / (sigma_i * sigma_m^2)
        denom = ri_std * rm_std**2
        
        mask = (valid_count >= min_periods) & (ri_std > 1e-10)
        coskew_arr[i, mask] = cross_mean[mask] / denom[mask]
    
    print(f"  计算完成")
    
    # 转回长表
    coskew_wide = pd.DataFrame(coskew_arr, index=dates, columns=stocks)
    coskew_long = coskew_wide.stack().reset_index()
    coskew_long.columns = ['date', 'stock_code', 'raw_coskew']
    coskew_long = coskew_long.dropna(subset=['raw_coskew'])
    
    return coskew_long


def neutralize_and_zscore(factor_df, kline_df):
    """成交额OLS中性化 + MAD winsorize + z-score"""
    kline = kline_df.sort_values(['stock_code', 'date']).copy()
    kline['log_amount_20d'] = kline.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    
    merged = factor_df.merge(
        kline[['date', 'stock_code', 'log_amount_20d']],
        on=['date', 'stock_code'],
        how='left'
    )
    
    results = []
    for date, gdf in merged.groupby('date'):
        gdf = gdf.dropna(subset=['raw_coskew', 'log_amount_20d'])
        if len(gdf) < 50:
            continue
        
        X = gdf[['log_amount_20d']].values
        y = gdf['raw_coskew'].values
        
        # MAD winsorize
        med = np.median(y)
        mad = np.median(np.abs(y - med))
        if mad > 0:
            y = np.clip(y, med - 5*1.4826*mad, med + 5*1.4826*mad)
        
        # OLS中性化
        lr = LinearRegression()
        lr.fit(X, y)
        resid = y - lr.predict(X)
        
        # MAD winsorize residual
        med_r = np.median(resid)
        mad_r = np.median(np.abs(resid - med_r))
        if mad_r > 0:
            resid = np.clip(resid, med_r - 3*1.4826*mad_r, med_r + 3*1.4826*mad_r)
        
        # z-score
        std_r = np.std(resid)
        if std_r > 0:
            z = (resid - np.mean(resid)) / std_r
        else:
            z = np.zeros_like(resid)
        
        codes = gdf['stock_code'].values
        for idx in range(len(z)):
            results.append({
                'date': date,
                'stock_code': codes[idx],
                'factor_value': z[idx]
            })
    
    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("Harvey & Siddique (2000) Coskewness 协偏度因子")
    print("=" * 60)
    
    data_dir = '/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab/data'
    
    print("\n1. 读取行情数据...")
    kline = pd.read_csv(f'{data_dir}/csi1000_kline_raw.csv')
    kline['date'] = pd.to_datetime(kline['date'])
    kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
    print(f"   行数: {len(kline)}, 股票数: {kline['stock_code'].nunique()}")
    
    print("\n2. 计算协偏度 (60日滚动)...")
    coskew_df = compute_coskewness_vectorized(kline, window=60, min_periods=30)
    print(f"   有效记录数: {len(coskew_df)}")
    print(f"   协偏度统计:")
    print(f"     mean={coskew_df['raw_coskew'].mean():.4f}")
    print(f"     median={coskew_df['raw_coskew'].median():.4f}")
    print(f"     std={coskew_df['raw_coskew'].std():.4f}")
    
    print("\n3. 成交额中性化 + MAD + z-score...")
    factor_df = neutralize_and_zscore(coskew_df, kline)
    print(f"   最终因子记录数: {len(factor_df)}")
    
    # 保存正向和反向
    factor_df.to_csv(f'{data_dir}/factor_coskew_v1.csv', index=False)
    
    factor_neg = factor_df.copy()
    factor_neg['factor_value'] = -factor_neg['factor_value']
    factor_neg.to_csv(f'{data_dir}/factor_coskew_neg_v1.csv', index=False)
    
    print(f"\n   正向(高协偏做多): factor_coskew_v1.csv")
    print(f"   反向(低协偏做多): factor_coskew_neg_v1.csv")
    print(f"   日期数: {factor_df['date'].nunique()}")
    print(f"\n✅ 协偏度因子计算完成!")


if __name__ == '__main__':
    main()
