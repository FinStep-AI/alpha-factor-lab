#!/usr/bin/env python3
"""
因子: 量价背离 (Volume-Price Divergence) v1
factor_id: vol_price_diverge_v1

逻辑:
  量价背离衡量价格走势和成交量走势的不一致性。
  - 计算20日窗口内，每日收益率与成交量变化率的滚动相关系数
  - 正常情况下量价同向(价涨量增/价跌量减)，相关系数为正
  - 量价背离(相关系数为负)暗示趋势不可持续：
    * 价涨量缩：上涨动力不足，后续回落
    * 价跌量增：恐慌性抛售，后续反弹
  - 反向使用：低相关(量价背离大) → 做多(均值回复)

假设:
  量价背离严重的股票后续倾向均值回复。
  在中证1000小盘股上，散户主导的量价背离信号更清晰。

复合信号设计:
  1. 核心成分: 20日滚动 corr(return, volume_change)
  2. 辅助成分: 近5日累计收益方向 × 近5日量能变化方向的交叉信号
  3. 强度成分: 近5日绝对收益 / 20日波动率 (趋势强度)
  
  最终因子 = -(核心×0.6 + 辅助×0.2 + 强度×0.2) (反向，低值做多)

参考:
  - Gervais, Kaniel & Mingelgrin (2001) "The High-Volume Return Premium"
  - 中信证券《量价背离的Alpha因子》
  - Llorente et al. (2002) "Dynamic volume-return relation"
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
    
    # ---- 基础变量 ----
    # 日收益率
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    
    # 成交量变化率 (使用成交额，比成交量更稳定)
    df['amount_chg'] = df.groupby('stock_code')['amount'].pct_change()
    # Clip extreme values
    df['amount_chg'] = df['amount_chg'].clip(-2, 10)  # 成交额变化可以很大
    
    # 成交量对数变化 (更稳健)
    df['log_amount'] = np.log(df['amount'].replace(0, np.nan))
    df['log_amount_chg'] = df.groupby('stock_code')['log_amount'].diff()
    
    window = 20
    min_per = 15
    
    # ---- 核心成分: 量价滚动相关 ----
    print("Computing rolling volume-price correlation (20d)...")
    
    def rolling_corr(group):
        """计算单只股票的滚动相关系数"""
        ret = group['ret']
        vol_chg = group['log_amount_chg']
        corr = ret.rolling(window=window, min_periods=min_per).corr(vol_chg)
        return corr
    
    df['vp_corr'] = df.groupby('stock_code', group_keys=False).apply(
        lambda g: rolling_corr(g)
    ).values
    
    # ---- 辅助成分: 短期量价交叉信号 ----
    print("Computing short-term VP cross signal...")
    # 近5日累计收益方向
    df['ret_5d'] = df.groupby('stock_code')['ret'].transform(
        lambda x: x.rolling(5, min_periods=3).sum()
    )
    # 近5日量能变化方向
    df['vol_5d'] = df.groupby('stock_code')['log_amount_chg'].transform(
        lambda x: x.rolling(5, min_periods=3).sum()
    )
    # 交叉信号：收益和量能方向不一致时为负（背离）
    # 标准化后相乘：正=同向，负=背离
    df['vp_cross'] = np.sign(df['ret_5d']) * np.sign(df['vol_5d'])
    # 取20日滚动均值平滑
    df['vp_cross_ma'] = df.groupby('stock_code')['vp_cross'].transform(
        lambda x: x.rolling(window, min_periods=min_per).mean()
    )
    
    # ---- 强度成分: 近期趋势强度 ----
    print("Computing trend strength...")
    df['vol_20d'] = df.groupby('stock_code')['ret'].transform(
        lambda x: x.rolling(window, min_periods=min_per).std()
    )
    # 近5日绝对累计收益 / 20日波动率 (取绝对值，方向已在交叉信号中)
    df['trend_strength'] = df['ret_5d'].abs() / (df['vol_20d'] * np.sqrt(5) + 1e-8)
    df['trend_strength'] = df['trend_strength'].clip(0, 5)
    
    # ---- 截面标准化各成分 ----
    print("Cross-sectional normalization...")
    for col in ['vp_corr', 'vp_cross_ma', 'trend_strength']:
        df[f'{col}_z'] = df.groupby('date')[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
    
    # ---- 复合因子 ----
    # 量价背离 = -(量价相关×0.6 + 交叉信号×0.2 + 趋势强度×量价背离方向×0.2)
    # 反向：低相关(背离) → 做多
    # 趋势强度只在背离时有意义，所以乘以-sign(vp_corr)
    df['raw_factor'] = -(
        0.6 * df['vp_corr_z'] + 
        0.2 * df['vp_cross_ma_z'] + 
        0.2 * df['trend_strength_z'] * (-np.sign(df['vp_corr_z']))
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
    
    out_path = data_dir / "factor_vol_price_diverge_v1.csv"
    output.to_csv(out_path, index=False)
    
    print(f"\nSaved to {out_path}")
    print(f"Shape: {output.shape}")
    print(f"Date range: {output['date'].min()} ~ {output['date'].max()}")
    print(f"Factor stats:")
    print(output['factor_value'].describe())


if __name__ == "__main__":
    main()
