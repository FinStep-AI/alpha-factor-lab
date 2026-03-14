#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下行Beta因子 (Downside Beta) — Ang, Chen & Xing (2006) JFE

论文：
  Ang, Chen & Xing (2006) "Downside Risk" Review of Financial Studies
  
核心思想：
  传统CAPM Beta假设收益率对称分布，但实际中投资者更关注下行风险。
  Downside Beta只在市场下跌日计算beta，衡量股票在市场下跌时的敏感度。
  
  高Downside Beta = 市场下跌时损失更大 = 更高的下行风险暴露
  
  资产定价理论：高下行风险应获得风险补偿(正向因子)
  但A股可能存在反转：高下行beta的股票反弹更强(风险补偿+反转)
  
构造：
  1. 每天计算等权市场收益率 r_mkt
  2. 只取 r_mkt < median(r_mkt) 的交易日
  3. 在这些日子上做回归：r_i = alpha + beta_down * r_mkt + epsilon
  4. beta_down 就是下行Beta
  5. 使用60日滚动窗口

中性化：对 log(20日均成交额) OLS 回归取残差
"""

import numpy as np
import pandas as pd
import sys

def calc_downside_beta(kline_path, output_path, window=60):
    print(f"📥 读取: {kline_path}")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # 日收益率
    if 'pct_change' in df.columns:
        df['ret'] = df['pct_change'] / 100.0
    else:
        df['ret'] = df.groupby('stock_code')['close'].pct_change()
    
    # 等权市场收益率
    mkt_ret = df.groupby('date')['ret'].mean().reset_index()
    mkt_ret.columns = ['date', 'mkt_ret']
    df = df.merge(mkt_ret, on='date', how='left')
    
    # 标记市场下跌日（收益低于滚动中位数）
    dates_sorted = df[['date', 'mkt_ret']].drop_duplicates().sort_values('date')
    # 用全局中位数或0，简单直接
    mkt_median = 0  # 市场收益<0即为下跌日
    
    print(f"📊 市场收益<0天数占比: {(dates_sorted['mkt_ret'] < 0).mean():.3f}")
    
    results = []
    n_stocks = df['stock_code'].nunique()
    
    for idx, (code, grp) in enumerate(df.groupby('stock_code')):
        if idx % 200 == 0:
            print(f"  处理: {idx}/{n_stocks}...")
        
        grp = grp.sort_values('date').reset_index(drop=True)
        n = len(grp)
        ret_arr = grp['ret'].values.astype(float)
        mkt_arr = grp['mkt_ret'].values.astype(float)
        dates = grp['date'].values
        
        for i in range(window - 1, n):
            w_ret = ret_arr[i - window + 1:i + 1]
            w_mkt = mkt_arr[i - window + 1:i + 1]
            
            # 只取市场下跌日
            valid = ~(np.isnan(w_ret) | np.isnan(w_mkt))
            down_mask = valid & (w_mkt < mkt_median)
            
            n_down = down_mask.sum()
            if n_down < 10:  # 需要至少10个下跌日
                continue
            
            y_down = w_ret[down_mask]
            x_down = w_mkt[down_mask]
            
            # OLS: y = alpha + beta * x
            X = np.column_stack([np.ones(n_down), x_down])
            try:
                params = np.linalg.lstsq(X, y_down, rcond=None)[0]
                beta_down = params[1]
            except:
                continue
            
            # 也计算上行Beta做比较
            up_mask = valid & (w_mkt >= mkt_median)
            n_up = up_mask.sum()
            if n_up >= 10:
                X_up = np.column_stack([np.ones(n_up), w_mkt[up_mask]])
                try:
                    params_up = np.linalg.lstsq(X_up, w_ret[up_mask], rcond=None)[0]
                    beta_up = params_up[1]
                except:
                    beta_up = np.nan
            else:
                beta_up = np.nan
            
            # 下行Beta - 上行Beta = 不对称性
            # 也可以直接用下行Beta
            results.append({
                'date': dates[i],
                'stock_code': code,
                'beta_down': beta_down,
                'beta_up': beta_up if not np.isnan(beta_up) else np.nan,
                'beta_asym': beta_down - beta_up if not np.isnan(beta_up) else np.nan
            })
    
    factor_df = pd.DataFrame(results)
    print(f"\n✅ 原始计算完成: {len(factor_df)} 行")
    print(f"  Beta Down: mean={factor_df['beta_down'].mean():.4f}, std={factor_df['beta_down'].std():.4f}")
    print(f"  Beta Up:   mean={factor_df['beta_up'].mean():.4f}, std={factor_df['beta_up'].std():.4f}")
    print(f"  Beta Asym: mean={factor_df['beta_asym'].mean():.4f}, std={factor_df['beta_asym'].std():.4f}")
    
    # 使用下行Beta作为主因子
    factor_raw = factor_df[['date', 'stock_code', 'beta_down']].copy()
    factor_raw.columns = ['date', 'stock_code', 'factor_raw']
    
    # 同时输出不对称性变体
    asym_raw = factor_df[['date', 'stock_code', 'beta_asym']].dropna().copy()
    asym_raw.columns = ['date', 'stock_code', 'factor_raw']
    
    # 成交额中性化
    print("⚙️  成交额中性化...")
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    amt_df = df[['date', 'stock_code', 'log_amount_20d']].drop_duplicates()
    
    def process_factor(raw_df, out_path):
        raw_df = raw_df.merge(amt_df, on=['date', 'stock_code'], how='left')
        
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
        
        raw_df = raw_df.groupby('date', group_keys=False).apply(neutralize)
        
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
        
        raw_df = raw_df.groupby('date', group_keys=False).apply(winsorize_zscore)
        
        output = raw_df[['date', 'stock_code', 'factor']].dropna()
        output = output.sort_values(['date', 'stock_code'])
        output.to_csv(out_path, index=False)
        print(f"  ✅ {out_path}: {len(output)} rows, {output['date'].nunique()} dates")
        return output
    
    # 主因子: 下行Beta
    process_factor(factor_raw, output_path)
    
    # 变体: Beta不对称性
    asym_path = output_path.replace('.csv', '_asym.csv')
    process_factor(asym_raw, asym_path)

if __name__ == '__main__':
    kline = sys.argv[1] if len(sys.argv) > 1 else 'data/csi1000_kline_raw.csv'
    out = sys.argv[2] if len(sys.argv) > 2 else 'data/factor_downside_beta_v1.csv'
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    
    calc_downside_beta(kline, out, window)
