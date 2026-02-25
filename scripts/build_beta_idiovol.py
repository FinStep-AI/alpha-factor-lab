#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建两个新因子：
1. Beta弹性因子 (beta_elasticity) — 滚动60日Beta的近期变化速度
   - 计算个股对中证1000指数的滚动Beta
   - Beta弹性 = Beta_20d - Beta_60d（短期Beta相对长期Beta的偏离）
   - 高Beta弹性 = 近期变得更激进 → 在动量市场中正向，在反转市场中负向
   
2. 特质波动率因子 (idio_vol) — Fama-French残差波动率
   - 个股收益对市场收益回归后的残差标准差
   - 学术界广泛验证的负向因子（低特质波动率 → 高收益）
   - 在中证1000小盘股中效果通常很好

两个因子都是纯量价，不需要基本面数据。
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

DATA_DIR = "data"

def load_data():
    """加载K线和指数数据"""
    codes = set(pd.read_csv(f"{DATA_DIR}/csi1000_codes.csv")['stock_code'].astype(str).str.zfill(6))
    
    kline = pd.read_csv(f"{DATA_DIR}/csi1000_kline_raw.csv")
    kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
    kline = kline[kline['stock_code'].isin(codes)]
    
    index = pd.read_csv(f"{DATA_DIR}/csi1000_index_daily.csv")
    index['mkt_return'] = index['pct_change'].astype(float) / 100.0
    
    # 个股日收益率
    kline['return'] = kline['pct_change'].astype(float) / 100.0
    
    return kline, index, codes


def build_beta_elasticity(kline, index):
    """
    Beta弹性因子：短期Beta - 长期Beta
    
    Beta_short = 20日滚动Beta
    Beta_long = 60日滚动Beta
    Beta弹性 = Beta_short - Beta_long
    
    正值 = 近期变得更敏感/激进
    负值 = 近期变得更防御
    """
    print("\n[因子1] Beta弹性因子")
    t0 = time.time()
    
    # 准备市场收益
    mkt = index[['date', 'mkt_return']].copy()
    mkt = mkt.rename(columns={'mkt_return': 'rm'})
    
    # 合并
    df = kline[['date', 'stock_code', 'return', 'amount']].merge(mkt, on='date', how='inner')
    df = df.sort_values(['stock_code', 'date'])
    
    results = []
    stocks = df['stock_code'].unique()
    
    for i, code in enumerate(stocks):
        if i % 200 == 0:
            print(f"  处理: {i}/{len(stocks)}", flush=True)
        
        stk = df[df['stock_code'] == code].copy()
        if len(stk) < 60:
            continue
        
        ret = stk['return'].values
        rm = stk['rm'].values
        dates = stk['date'].values
        amounts = stk['amount'].values
        
        # 滚动Beta (20日和60日)
        for j in range(60, len(ret)):
            # 60日Beta
            r60 = ret[j-60:j]
            m60 = rm[j-60:j]
            cov60 = np.cov(r60, m60)
            var_m60 = cov60[1, 1]
            beta60 = cov60[0, 1] / var_m60 if var_m60 > 1e-10 else np.nan
            
            # 20日Beta
            r20 = ret[j-20:j]
            m20 = rm[j-20:j]
            cov20 = np.cov(r20, m20)
            var_m20 = cov20[1, 1]
            beta20 = cov20[0, 1] / var_m20 if var_m20 > 1e-10 else np.nan
            
            if np.isnan(beta60) or np.isnan(beta20):
                continue
            
            elasticity = beta20 - beta60
            results.append((dates[j], code, elasticity, amounts[j]))
    
    result_df = pd.DataFrame(results, columns=['date', 'stock_code', 'raw_factor', 'amount'])
    
    # 截面标准化 + 市值中性化
    factor_values = []
    for dt, group in result_df.groupby('date'):
        g = group.copy()
        if len(g) < 30:
            continue
        
        # Winsorize (MAD)
        med = g['raw_factor'].median()
        mad = (g['raw_factor'] - med).abs().median()
        bound = 5 * 1.4826 * mad
        g['raw_factor'] = g['raw_factor'].clip(med - bound, med + bound)
        
        # 市值中性化
        g['ln_amt'] = np.log(g['amount'].clip(lower=1))
        mask = g['ln_amt'].notna() & g['raw_factor'].notna()
        if mask.sum() > 30:
            try:
                sl, ic, _, _, _ = sp_stats.linregress(g.loc[mask, 'ln_amt'].values, g.loc[mask, 'raw_factor'].values)
                g.loc[mask, 'neutralized'] = g.loc[mask, 'raw_factor'] - (ic + sl * g.loc[mask, 'ln_amt'])
            except:
                g['neutralized'] = g['raw_factor']
        else:
            g['neutralized'] = g['raw_factor']
        
        # Z-score
        m, s = g['neutralized'].mean(), g['neutralized'].std()
        if s > 0:
            g['factor_value'] = (g['neutralized'] - m) / s
            for _, row in g.iterrows():
                if pd.notna(row['factor_value']):
                    factor_values.append((dt, row['stock_code'], round(row['factor_value'], 6)))
    
    out = pd.DataFrame(factor_values, columns=['date', 'stock_code', 'factor_value'])
    out.to_csv(f"{DATA_DIR}/csi1000_beta_elasticity.csv", index=False)
    print(f"  完成: {len(out)}行, {out['date'].nunique()}天, {time.time()-t0:.1f}秒")
    return out


def build_idio_vol(kline, index):
    """
    特质波动率因子：对市场回归后的残差波动率（20日滚动）
    
    r_i = alpha + beta * r_m + epsilon
    idio_vol = std(epsilon) over 20 days
    
    低特质波动 → 高收益（经典异象）
    """
    print("\n[因子2] 特质波动率因子")
    t0 = time.time()
    
    mkt = index[['date', 'mkt_return']].copy().rename(columns={'mkt_return': 'rm'})
    
    df = kline[['date', 'stock_code', 'return', 'amount']].merge(mkt, on='date', how='inner')
    df = df.sort_values(['stock_code', 'date'])
    
    results = []
    stocks = df['stock_code'].unique()
    window = 20
    
    for i, code in enumerate(stocks):
        if i % 200 == 0:
            print(f"  处理: {i}/{len(stocks)}", flush=True)
        
        stk = df[df['stock_code'] == code].copy()
        if len(stk) < window:
            continue
        
        ret = stk['return'].values
        rm = stk['rm'].values
        dates = stk['date'].values
        amounts = stk['amount'].values
        
        for j in range(window, len(ret)):
            r_w = ret[j-window:j]
            m_w = rm[j-window:j]
            
            # 回归
            var_m = np.var(m_w)
            if var_m < 1e-10:
                continue
            beta = np.cov(r_w, m_w)[0, 1] / var_m
            alpha = np.mean(r_w) - beta * np.mean(m_w)
            residuals = r_w - (alpha + beta * m_w)
            ivol = np.std(residuals)
            
            if np.isnan(ivol):
                continue
            
            results.append((dates[j], code, ivol, amounts[j]))
    
    result_df = pd.DataFrame(results, columns=['date', 'stock_code', 'raw_factor', 'amount'])
    
    # 截面处理
    factor_values = []
    for dt, group in result_df.groupby('date'):
        g = group.copy()
        if len(g) < 30:
            continue
        
        # Winsorize
        med = g['raw_factor'].median()
        mad = (g['raw_factor'] - med).abs().median()
        bound = 5 * 1.4826 * mad
        g['raw_factor'] = g['raw_factor'].clip(med - bound, med + bound)
        
        # 市值中性化
        g['ln_amt'] = np.log(g['amount'].clip(lower=1))
        mask = g['ln_amt'].notna() & g['raw_factor'].notna()
        if mask.sum() > 30:
            try:
                sl, ic, _, _, _ = sp_stats.linregress(g.loc[mask, 'ln_amt'].values, g.loc[mask, 'raw_factor'].values)
                g.loc[mask, 'neutralized'] = g.loc[mask, 'raw_factor'] - (ic + sl * g.loc[mask, 'ln_amt'])
            except:
                g['neutralized'] = g['raw_factor']
        else:
            g['neutralized'] = g['raw_factor']
        
        m, s = g['neutralized'].mean(), g['neutralized'].std()
        if s > 0:
            g['factor_value'] = (g['neutralized'] - m) / s
            for _, row in g.iterrows():
                if pd.notna(row['factor_value']):
                    factor_values.append((dt, row['stock_code'], round(row['factor_value'], 6)))
    
    out = pd.DataFrame(factor_values, columns=['date', 'stock_code', 'factor_value'])
    out.to_csv(f"{DATA_DIR}/csi1000_idio_vol.csv", index=False)
    print(f"  完成: {len(out)}行, {out['date'].nunique()}天, {time.time()-t0:.1f}秒")
    return out


def main():
    print("=" * 60)
    print("构建 Beta弹性 + 特质波动率 因子")
    print("=" * 60)
    
    kline, index, codes = load_data()
    print(f"K线: {len(kline)}行, 指数: {len(index)}天")
    
    build_beta_elasticity(kline, index)
    build_idio_vol(kline, index)
    
    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
