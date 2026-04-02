#!/usr/bin/env python3
"""
ROE Acceleration v4 - Sigmoid Truncation
=========================================
基于roe_accel_v1(IC=0.023, t=2.97, mono=0.8差一点)的改进版：
- 问题：G4>G5（极端高加速度组不是收益最高）
- 解决思路：对ROE加速度值做sigmoid变换，压制极端值

构造步骤:
1. 读取基本面数据(ROE)
2. 计算ROE同比变化(delta_ROE = ROE(t) - ROE(t-4))
3. 计算ROE加速度(accel = delta_ROE(t) - delta_ROE(t-1))  
4. 映射到日频(45天延迟)
5. 对加速度做tanh变换压缩极端值
6. 市值中性化 + MAD + z-score
"""

import numpy as np
import pandas as pd
import os
import sys

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # 检查是否已有roe_accel的原始因子
    accel_path = os.path.join(data_dir, 'factor_roe_accel.csv')
    if os.path.exists(accel_path):
        print("Found existing roe_accel data, loading...")
        df = pd.read_csv(accel_path)
        print(f"  Loaded {len(df)} rows")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Date range: {df['date'].min()} ~ {df['date'].max()}")
        
        # 检测列名
        fcol = 'factor' if 'factor' in df.columns else 'factor_value'
        # 统一列名
        if fcol != 'factor':
            df = df.rename(columns={fcol: 'factor'})
        
        vals = df['factor'].dropna()
        print(f"\n  Factor stats: mean={vals.mean():.4f}, std={vals.std():.4f}")
        print(f"  min={vals.min():.4f}, max={vals.max():.4f}")
        print(f"  5%={vals.quantile(0.05):.4f}, 95%={vals.quantile(0.95):.4f}")
        
        df_v4 = df.copy()
        
        all_results = []
        for date, group in df_v4.groupby('date'):
            g = group.dropna(subset=['factor']).copy()
            if len(g) < 50:
                continue
            
            # 先做tanh
            z = g['factor'].values
            z_tanh = np.tanh(z)  # 压缩到[-1, 1]
            
            # 再re-normalize到z-score
            std = np.std(z_tanh)
            if std < 1e-10:
                continue
            z_new = (z_tanh - np.mean(z_tanh)) / std
            z_new = np.clip(z_new, -3, 3)
            
            g = g.copy()
            g['factor'] = z_new
            all_results.append(g[['date', 'stock_code', 'factor']])
        
        result_df = pd.concat(all_results, ignore_index=True)
        
        output_path = os.path.join(data_dir, 'factor_roe_accel_v4.csv')
        result_df.to_csv(output_path, index=False)
        print(f"\n  Saved tanh version: {output_path} ({len(result_df)} rows)")
        
        return
    
    # 如果没有现成的，需要从头计算
    print("No existing roe_accel data found. Computing from scratch...")
    
    # 读取基本面数据
    fund_path = os.path.join(data_dir, 'csi1000_fundamental_cache.csv')
    kline_path = os.path.join(data_dir, 'csi1000_kline_raw.csv')
    
    df_fund = pd.read_csv(fund_path)
    df_kline = pd.read_csv(kline_path)
    
    print(f"  Fundamental: {len(df_fund)} rows, columns={df_fund.columns.tolist()}")
    print(f"  Kline: {len(df_kline)} rows")
    
    df_fund['report_date'] = pd.to_datetime(df_fund['report_date'])
    df_kline['date'] = pd.to_datetime(df_kline['date'])
    
    # 计算ROE同比变化和加速度
    # 按stock排序，按report_date排序
    df_fund = df_fund.sort_values(['stock_code', 'report_date'])
    
    accel_records = []
    for code, group in df_fund.groupby('stock_code'):
        group = group.sort_values('report_date').reset_index(drop=True)
        roes = group['roe'].values
        dates = group['report_date'].values
        
        if len(group) < 6:
            continue
        
        for i in range(5, len(group)):
            # 同比变化 (t vs t-4)
            delta_roe_curr = roes[i] - roes[i-4] if i >= 4 else np.nan
            delta_roe_prev = roes[i-1] - roes[i-5] if i >= 5 else np.nan
            
            if np.isnan(delta_roe_curr) or np.isnan(delta_roe_prev):
                continue
            if np.isnan(roes[i]) or np.isnan(roes[i-1]) or np.isnan(roes[i-4]) or np.isnan(roes[i-5]):
                continue
            
            accel = delta_roe_curr - delta_roe_prev
            
            # 报告日期 + 45天延迟
            avail_date = dates[i] + pd.Timedelta(days=45)
            
            accel_records.append({
                'stock_code': code,
                'report_date': dates[i],
                'avail_date': avail_date,
                'roe_accel': accel
            })
    
    df_accel = pd.DataFrame(accel_records)
    print(f"  ROE accel records: {len(df_accel)}")
    
    if len(df_accel) == 0:
        print("ERROR: No acceleration records!")
        sys.exit(1)
    
    # 映射到日频
    # 对每个交易日，每只股票取最新的可用信号
    trade_dates = sorted(df_kline['date'].unique())
    all_stocks = df_kline['stock_code'].unique()
    
    daily_records = []
    
    # 获取每只股票的市值用于中性化
    for code in all_stocks:
        stock_accel = df_accel[df_accel['stock_code'] == code].sort_values('avail_date')
        stock_kline = df_kline[df_kline['stock_code'] == code].sort_values('date')
        
        if len(stock_accel) == 0:
            continue
        
        accel_dates = stock_accel['avail_date'].values
        accel_vals = stock_accel['roe_accel'].values
        
        for _, row in stock_kline.iterrows():
            td = row['date']
            # 找最新的可用信号
            mask = accel_dates <= td
            if not np.any(mask):
                continue
            idx = np.where(mask)[0][-1]
            
            # 只用最近1年内的信号
            days_diff = (td - pd.Timestamp(accel_dates[idx])).days
            if days_diff > 365:
                continue
            
            daily_records.append({
                'date': td,
                'stock_code': code,
                'roe_accel_raw': accel_vals[idx],
                'log_amount': np.log(row['amount'] + 1) if pd.notna(row['amount']) else np.nan
            })
    
    df_daily = pd.DataFrame(daily_records)
    print(f"  Daily mapped: {len(df_daily)} rows")
    
    # 计算20日平均成交额用于中性化
    # 简化：直接用当日的log_amount
    
    # 截面中性化 + tanh + z-score
    all_results = []
    for date, group in df_daily.groupby('date'):
        g = group.dropna(subset=['roe_accel_raw', 'log_amount']).copy()
        if len(g) < 50:
            continue
        
        y = g['roe_accel_raw'].values
        x = g['log_amount'].values
        x_with_const = np.column_stack([np.ones(len(x)), x])
        
        try:
            beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
            residual = y - x_with_const @ beta
        except:
            continue
        
        # MAD winsorize
        med = np.median(residual)
        mad = np.median(np.abs(residual - med))
        if mad < 1e-10:
            continue
        upper = med + 5 * 1.4826 * mad
        lower = med - 5 * 1.4826 * mad
        residual = np.clip(residual, lower, upper)
        
        # z-score
        std = np.std(residual)
        if std < 1e-10:
            continue
        z = (residual - np.mean(residual)) / std
        
        # tanh变换压缩极端值
        z_tanh = np.tanh(z)
        std2 = np.std(z_tanh)
        if std2 < 1e-10:
            continue
        z_final = (z_tanh - np.mean(z_tanh)) / std2
        z_final = np.clip(z_final, -3, 3)
        
        g = g.copy()
        g['factor'] = z_final
        all_results.append(g[['date', 'stock_code', 'factor']])
    
    if not all_results:
        print("ERROR: No factor values!")
        sys.exit(1)
    
    result_df = pd.concat(all_results, ignore_index=True)
    output_path = os.path.join(data_dir, 'factor_roe_accel_v4.csv')
    result_df.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path} ({len(result_df)} rows)")
    print(f"  Date range: {result_df['date'].min()} ~ {result_df['date'].max()}")

if __name__ == '__main__':
    main()
