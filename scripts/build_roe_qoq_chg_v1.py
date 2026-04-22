#!/usr/bin/env python3
"""
因子: ROE季度变化率 (ROE QoQ Change) v1
factor_id: roe_qoq_chg_v1

逻辑:
  - 最新报告期ROE - 上期ROE = 季度ROE变化
  - 代表公司盈利能力的边际改善/恶化
  - 正向: ROE提升的股票后续表现更好
  - 经成交额中性化(proxy for market cap)

假设:
  - ROE提升 = 基本面改善/效率提高 → 价值发现 → 持续上涨
  - 属于Quality中的Growth维度：不只看盈利水平高低，看改善趋势
  - Novy-Marx (2013): "Is Momentum Really Momentum?" —  profitability changes predict returns

中性化:
  - OLS回归对log(amount_20d_avg)做中性化
  - 消除市值暴露

输出:
  - data/factor_roe_qoq_chg_v1.csv (wide format: date rows, stock_code columns)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def main():
    data_dir = Path("data")
    output_dir = data_dir
    output_dir.mkdir(exist_ok=True)
    
    # ============================================================
    # 1. 加载数据
    # ============================================================
    print("Loading fundamental data...")
    fund = pd.read_csv(data_dir / "csi1000_fundamental_cache.csv")
    fund['report_date'] = pd.to_datetime(fund['report_date'])
    fund = fund.sort_values(['stock_code', 'report_date'])
    
    print("Loading kline data...")
    kline = pd.read_csv(data_dir / "csi1000_kline_raw.csv")
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date'])
    
    print(f"Fundamental: {fund['stock_code'].nunique()} stocks, {fund['report_date'].nunique()} dates")
    print(f"Kline: {kline['stock_code'].nunique()} stocks, {kline['date'].min()} ~ {kline['date'].max()}")
    
    # ============================================================
    # 2. 计算ROE QoQ变化
    # ============================================================
    print("Computing ROE QoQ change...")
    
    fund['roe_prev'] = fund.groupby('stock_code')['roe'].shift(1)
    fund['roe_qoq'] = fund['roe'] - fund['roe_prev']  # 绝对变化 (%)
    # 也计算百分比变化: (ROE_t - ROE_t-1) / |ROE_t-1|
    fund['roe_qoq_pct'] = fund['roe_qoq'] / fund['roe_prev'].abs().replace(0, np.nan)
    
    # 关健: 将quarterly信号映射到每个kline日期
    # 方法: 用report_date作为信号发布日期，与kline的date做merge(merge_asof)
    # 之前有fundamental数据的日期: report_date
    # kline可用该信号的日期: >= report_date
    # 用forward-fill within stock
    
    # 简化: 取每个stock的fundamental, 按date升序, forward-fill
    # 先创建一个所有kline日期的"版本": merge_asof将fundamental的signal映射到最近>=该日期的report_date
    
    # 更清晰的逻辑:
    # 对于每个stock, 对每个kline date, 找 <= date 的最新report_date的roe_qoq
    
    # 构建signal dates
    fund_signal = fund[['stock_code', 'report_date', 'roe_qoq', 'roe_qoq_pct']].dropna(subset=['roe_qoq']).copy()
    fund_signal.rename(columns={'report_date': 'signal_date'}, inplace=True)
    
    # merge_asof: for each (stock, kline_date), find latest fund signal_date <= kline_date
    kline_for_merge = kline[['date', 'stock_code']].copy()
    kline_for_merge = kline_for_merge.sort_values(['stock_code', 'date'])
    fund_signal = fund_signal.sort_values(['stock_code', 'signal_date'])
    
    merged = pd.merge_asof(
        kline_for_merge,
        fund_signal,
        left_on='date',
        right_on='signal_date',
        by='stock_code',
        direction='backward'
    )
    
    # ============================================================
    # 3. 市值中性化准备
    # ============================================================
    print("Computing market cap proxy...")
    # market_cap_proxy = amount / turnover = avg_price * shares
    # turnover is in %, so amount / (turnover/100) = amount * 100 / turnover
    kline['turnover_clean'] = kline['turnover'].replace(0, np.nan)
    kline['mktcap_proxy'] = kline['amount'] / (kline['turnover_clean'] / 100.0)
    kline['log_mktcap'] = np.log(kline['mktcap_proxy'].replace(0, np.nan))
    
    # also compute 20d average amount for neutralization
    kline['amount_ma20'] = kline.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    kline['log_amount_20d'] = np.log(kline['amount_ma20'].replace(0, np.nan))
    
    # ============================================================
    # 4. 合并因子值
    # ============================================================
    print("Merging and neutralization...")
    
    # 回merged加log_mktcap
    merged = merged.merge(
        kline[['date', 'stock_code', 'log_mktcap', 'log_amount_20d', 'roe_qoq', 'roe_qoq_pct']],
        on=['date', 'stock_code'],
        how='left',
        suffixes=('', '_dup')
    )
    
    # 优先用merge_asof来的(更及时), merge直接来的如果更new就覆盖
    # 其实两个一样, 但保持一致性
    merged['factor_raw'] = merged['roe_qoq_pct']
    
    # 去掉无穷大值
    merged['factor_raw'] = merged['factor_raw'].clip(-1, 1)  # ROE变化率合理范围
    
    # ============================================================
    # 5. 横截面中性化 (OLS: factor ~ log(amount_20d))
    # ============================================================
    print("Neutralizing...")
    
    def neutralize_group(group):
        """OLS neutralize factor_raw ~ log_amount_20d"""
        y = group['factor_raw'].values
        x_col = 'log_amount_20d'
        x = group[x_col].values
        
        valid = np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 30:
            group['factor_value'] = np.nan
            return group
        
        # OLS
        x_valid = x[valid]
        y_valid = y[valid]
        x_dm = x_valid - x_valid.mean()
        beta = np.sum(x_dm * (y_valid - y_valid.mean())) / (np.sum(x_dm**2) + 1e-10)
        alpha = y_valid.mean() - beta * x_valid.mean()
        residual = y_valid - (alpha + beta * x_valid)
        
        group.loc[group.index[valid], 'factor_value'] = residual
        group.loc[group.index[~valid], 'factor_value'] = np.nan
        return group
    
    merged = merged.groupby('date', group_keys=False).apply(neutralize_group)
    
    # ============================================================
    # 6. MAD winsorize + cross-sectional rank/z-score
    # ============================================================
    print("Winsorizing and standardizing...")
    
    def standardize_group(group):
        vals = group['factor_value'].values
        valid = np.isfinite(vals)
        if valid.sum() < 30:
            group['factor_final'] = np.nan
            return group
        
        v = vals[valid]
        # MAD winsorize: clip at 3 MADs
        median = np.median(v)
        mad = np.median(np.abs(v - median))
        upper = median + 5.2 * mad  # ~3 std equivalent
        lower = median - 5.2 * mad
        v = np.clip(v, lower, upper)
        
        # z-score
        mean_v = v.mean()
        std_v = v.std()
        if std_v > 0:
            v = (v - mean_v) / std_v
        else:
            v = np.zeros_like(v)
        
        group.loc[group.index[valid], 'factor_final'] = v
        group.loc[group.index[~valid], 'factor_final'] = np.nan
        return group
    
    merged = merged.groupby('date', group_keys=False).apply(standardize_group)
    
    # ============================================================
    # 7. 输出wide format CSV
    # ============================================================
    print("Writing output...")
    
    df_out = merged[['date', 'stock_code', 'factor_final']].dropna(subset=['factor_final'])
    df_out = df_out.pivot(index='date', columns='stock_code', values='factor_final')
    df_out.index = pd.to_datetime(df_out.index)
    df_out = df_out.sort_index()
    
    # Drop dates with too few stocks (<100)
    df_out = df_out.loc[df_out.count(axis=1) >= 100]
    
    output_path = output_dir / "factor_roe_qoq_chg_v1.csv"
    df_out.to_csv(output_path)
    
    print(f"\nDone! Output: {output_path}")
    print(f"Shape: {df_out.shape}")
    print(f"Date range: {df_out.index.min()} ~ {df_out.index.max()}")
    print(f"Mean stocks per date: {df_out.count(axis=1).mean():.0f}")
    print(f"\nFactor stats (cross-sectional, last date):")
    last_vals = df_out.iloc[-1].dropna()
    print(f"  N={len(last_vals)}, mean={last_vals.mean():.4f}, std={last_vals.std():.4f}")
    
    # Save a sample for inspection
    sample_path = output_dir / "factor_roe_qoq_chg_v1_sample.txt"
    with open(sample_path, 'w') as f:
        f.write(f"Shape: {df_out.shape}\n")
        f.write(f"Date range: {df_out.index.min()} ~ {df_out.index.max()}\n")
        f.write(f"\nLast 5 dates factor distribution:\n")
        for i in range(-5, 0):
            vals = df_out.iloc[i].dropna()
            f.write(f"  {df_out.index[i].date()}: N={len(vals)}, mean={vals.mean():.4f}, std={vals.std():.4f}\n")
    print(f"Sample: {sample_path}")


if __name__ == "__main__":
    main()
