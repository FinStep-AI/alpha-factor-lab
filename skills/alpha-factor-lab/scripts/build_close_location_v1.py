#!/usr/bin/env python3
"""
因子: 尾盘异动 (Close-to-Range Position) v1
factor_id: close_location_v1

逻辑:
  尾盘异动衡量收盘价在日内价格区间中的位置(Close Location Value, CLV)。
  CLV = (2×Close - High - Low) / (High - Low)
  取值 [-1, +1]：
    +1: 收盘在最高价 (尾盘强势)
    -1: 收盘在最低价 (尾盘弱势)
    
  20日CLV均值衡量持续的尾盘行为模式:
  - 持续尾盘拉升 (CLV持续正): 可能有知情交易者/机构在尾盘买入
  - 持续尾盘回落 (CLV持续负): 尾盘抛压重，可能散户主导
  
  A股特色: 
  - 14:57集合竞价制度，机构常在尾盘执行大单
  - ETF/指数基金尾盘调仓效应
  - 游资尾盘打板/炸板模式
  
  因子方向: 反向使用 (高CLV → 低后续收益)
  - 尾盘持续强势的股票可能已被过度追涨
  - 尾盘持续弱势的股票卖压释放后更容易反弹

参考:
  - Arms (1989) "Volume Cycles in the Stock Market"
  - 海通证券《收盘价位置因子研究》
  - Berkman et al. (2012) "Paying Attention: Overnight Returns and the Hidden Cost of Buying at the Open"
"""

import numpy as np
import pandas as pd
from pathlib import Path


def main():
    data_dir = Path("data")
    
    print("Loading kline data...")
    df = pd.read_csv(data_dir / "csi1000_kline_raw.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date'])
    
    n_stocks = df['stock_code'].nunique()
    print(f"Stocks: {n_stocks}, Dates: {df['date'].min()} ~ {df['date'].max()}")
    
    # ---- Close Location Value (CLV) ----
    df['range'] = df['high'] - df['low']
    # 避免0幅区间(一字涨跌停)
    df['range'] = df['range'].replace(0, np.nan)
    df['clv'] = (2 * df['close'] - df['high'] - df['low']) / df['range']
    # CLV在[-1,1]范围，clip处理极端值
    df['clv'] = df['clv'].clip(-1, 1)
    
    window = 20
    min_per = 15
    
    # ---- 核心成分1: 20日CLV均值 ----
    print("Computing 20d CLV mean...")
    df['clv_mean'] = df.groupby('stock_code')['clv'].transform(
        lambda x: x.rolling(window, min_periods=min_per).mean()
    )
    
    # ---- 核心成分2: CLV趋势(近5日vs远15日) ----
    print("Computing CLV trend...")
    df['clv_5d'] = df.groupby('stock_code')['clv'].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )
    df['clv_20d'] = df['clv_mean']  # already computed
    df['clv_trend'] = df['clv_5d'] - df['clv_20d']
    
    # ---- 核心成分3: CLV与成交量的交互 ----
    # 高量日的CLV更有信息量(机构大单)
    print("Computing volume-weighted CLV...")
    df['log_amount'] = np.log(df['amount'].replace(0, np.nan))
    df['log_amount_ma'] = df.groupby('stock_code')['log_amount'].transform(
        lambda x: x.rolling(window, min_periods=min_per).mean()
    )
    # 量能相对强度
    df['vol_rel'] = df['log_amount'] - df['log_amount_ma']
    df['vol_rel'] = df['vol_rel'].clip(-2, 2)
    # 放量日CLV权重更大
    df['clv_vol_weighted'] = df['clv'] * (1 + df['vol_rel'].clip(0, 2))
    df['clv_vol_mean'] = df.groupby('stock_code')['clv_vol_weighted'].transform(
        lambda x: x.rolling(window, min_periods=min_per).mean()
    )
    
    # ---- 截面标准化 ----
    print("Cross-sectional normalization...")
    for col in ['clv_mean', 'clv_trend', 'clv_vol_mean']:
        df[f'{col}_z'] = df.groupby('date')[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
    
    # ---- 复合因子 ----
    # 反向使用：高CLV（尾盘强势）→ 低后续收益
    # 核心CLV占主导，趋势和量加权辅助
    df['raw_factor'] = -(
        0.5 * df['clv_mean_z'] + 
        0.25 * df['clv_trend_z'] + 
        0.25 * df['clv_vol_mean_z']
    )
    
    # ---- 市值代理 ----
    df['mktcap_proxy'] = df['amount'] / df['turnover'].replace(0, np.nan)
    df['log_mktcap'] = np.log(df['mktcap_proxy'].replace(0, np.nan))
    
    result = df[['date', 'stock_code', 'raw_factor', 'log_mktcap']].dropna().copy()
    
    # ---- 5% 缩尾 ----
    print("5% winsorization...")
    def winsorize(s, lower=0.025, upper=0.975):
        lo, hi = s.quantile(lower), s.quantile(upper)
        return s.clip(lo, hi)
    
    result['raw_factor'] = result.groupby('date')['raw_factor'].transform(winsorize)
    
    # ---- 截面Z-score ----
    print("Cross-sectional standardization...")
    result['factor_zscore'] = result.groupby('date')['raw_factor'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    result['factor_zscore'] = result['factor_zscore'].clip(-3, 3)
    
    # ---- 市值中性化 (OLS) ----
    print("Market cap neutralization (OLS)...")
    def neutralize(group):
        g = group.dropna(subset=['factor_zscore', 'log_mktcap'])
        if len(g) < 10:
            g = g.copy()
            g['factor_neutral'] = np.nan
            return g[['factor_neutral']]
        x = g['log_mktcap'].values
        y = g['factor_zscore'].values
        x_mean = np.nanmean(x)
        y_mean = np.nanmean(y)
        b = np.nansum((x - x_mean) * (y - y_mean)) / (np.nansum((x - x_mean)**2) + 1e-10)
        a = y_mean - b * x_mean
        residuals = y - (a + b * x)
        g = g.copy()
        g['factor_neutral'] = residuals
        return g[['factor_neutral']]
    
    neutralized = result.groupby('date', group_keys=False).apply(neutralize)
    result['factor_neutral'] = neutralized['factor_neutral'].values
    
    # ---- 最终Z-score ----
    result['factor_value'] = result.groupby('date')['factor_neutral'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    result['factor_value'] = result['factor_value'].clip(-3, 3)
    
    # ---- 输出 ----
    output = result[['date', 'stock_code', 'factor_value']].dropna()
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')
    
    out_path = data_dir / "factor_close_location_v1.csv"
    output.to_csv(out_path, index=False)
    
    print(f"\nSaved to {out_path}")
    print(f"Shape: {output.shape}")
    print(f"Date range: {output['date'].min()} ~ {output['date'].max()}")
    print(f"Factor stats:")
    print(output['factor_value'].describe())


if __name__ == "__main__":
    main()
