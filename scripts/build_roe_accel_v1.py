#!/usr/bin/env python3
"""
因子: ROE加速度 v1 (ROE Acceleration v1)
factor_id: roe_accel_v1

逻辑:
  - 最新报告期ROE - 上期ROE = 季度ROE变化率
  - 衡量盈利能力的边际改善速度
  - 正向: RoE加速改善的股票后续表现更好
  - 经成交额OLS中性化 + MAD缩尾 + z-score

假设:
  - ROE加速上升 = 盈利能力边际改善 + 经营拐点
  - Novy-Marx (2013): profitability changes predict cross-section returns
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def main():
    data_dir = Path("data")
    
    print("Loading data...")
    fund = pd.read_csv(data_dir / "csi1000_fundamental_cache.csv")
    fund['report_date'] = pd.to_datetime(fund['report_date'])
    fund = fund.sort_values(['stock_code', 'report_date'])
    
    kline = pd.read_csv(data_dir / "csi1000_kline_raw.csv")
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date'])
    
    print(f"Fund: {fund['stock_code'].nunique()} stocks, {fund['report_date'].min()} ~ {fund['report_date'].max()}")
    
    # ============================================================
    # 2. ROE QoQ change rate
    # ============================================================
    print("Computing ROE QoQ change...")
    fund['roe_prev'] = fund.groupby('stock_code')['roe'].shift(1)
    fund['roe_qoq_raw'] = fund['roe'] - fund['roe_prev']
    # Normalize by |prev|, clip extreme
    fund['roe_qoq_pct'] = fund['roe_qoq_raw'] / fund['roe_prev'].abs().replace(0, np.nan)
    fund['roe_qoq_pct'] = fund['roe_qoq_pct'].fillna(0).clip(-0.5, 0.5)
    
    valid_count = fund['roe_qoq_pct'].abs().gt(0.001).sum()
    print(f"Valid RoE QoQ records: {valid_count}")
    
    # ============================================================
    # 3. 构造wide-format factor matrix
    # ============================================================
    print("Building factor matrix (wide format)...")
    
    fund['fund_dt'] = fund['report_date']
    # For each stock: create time series indexed by report_date
    fund_pivot = fund.pivot_table(
        index='report_date', columns='stock_code', values='roe_qoq_pct', aggfunc='first'
    )
    fund_pivot.index = pd.to_datetime(fund_pivot.index)
    fund_pivot = fund_pivot.sort_index()
    
    # All kline dates
    kline_dates = pd.to_datetime(kline['date'].unique())
    kline_dates = np.sort(kline_dates)
    
    # Reindex fund to kline dates + ffill
    # This creates a lag: the latest report's value is available from next trading day
    fund_aligned = fund_pivot.reindex(kline_dates, method='ffill')
    print(f"Factor matrix shape after ffill: {fund_aligned.shape}")
    print(f"Non-zero days: {(fund_aligned.abs() > 0.001).sum(axis=1).sum()}")
    
    # ============================================================
    # 4. Market cap proxy for neutralization
    # ============================================================
    print("Computing market cap proxy...")
    kline['turnover_clean'] = kline['turnover'].replace(0, np.nan)
    kline['mktcap_proxy'] = kline['amount'] / (kline['turnover_clean'] / 100.0)
    kline['log_amount_20d'] = kline.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    kline['log_amount_20d'] = np.log(kline['log_amount_20d'].replace(0, np.nan))
    
    # Merge kline + factor
    print("Merging factor with kline...")
    merged = kline[['date', 'stock_code', 'log_amount_20d']].copy()
    merged['factor_raw'] = merged.apply(
        lambda row: fund_aligned.loc[row['date'], row['stock_code']]
        if row['date'] in fund_aligned.index and row['stock_code'] in fund_aligned.columns
        else np.nan,
        axis=1
    )
    merged = merged.dropna(subset=['factor_raw', 'log_amount_20d'])
    
    print(f"Merged records: {len(merged)}, unique dates: {merged['date'].nunique()}")
    print(f"Date range: {merged['date'].min()} ~ {merged['date'].max()}")
    
    # ============================================================
    # 5. OLS neutralize by log(amount_20d) per date
    # ============================================================
    print("OLS neutralization...")
    
    def neutralize(group):
        y = group['factor_raw'].values
        x = group['log_amount_20d'].values
        valid = np.isfinite(y) & np.isfinite(x)
        n_valid = valid.sum()
        
        if n_valid < 30:
            group['factor_value'] = np.nan
            return group
        
        y_v = y[valid]
        x_v = x[valid]
        x_dm = x_v - x_v.mean()
        y_dm = y_v - y_v.mean()
        
        denom = np.sum(x_dm**2) + 1e-10
        beta = np.sum(x_dm * y_dm) / denom
        alpha = y_v.mean() - beta * x_v.mean()
        residual = np.full(len(y), np.nan)
        residual[valid] = y_v - (alpha + beta * x_v)
        
        group['factor_value'] = residual
        return group
    
    merged = merged.groupby('date', group_keys=False).apply(neutralize)
    
    # ============================================================
    # 6. Cross-sectional standardization (MAD + z-score)
    # ============================================================
    print("Standardizing...")
    
    def standardize(group):
        vals = group['factor_value'].values
        valid = np.isfinite(vals)
        if valid.sum() < 30:
            group['factor_final'] = np.nan
            return group
        
        v = vals[valid].copy()
        median = np.median(v)
        mad = np.median(np.abs(v - median))
        
        if mad > 0:
            u = median + 5.2 * mad
            l = median - 5.2 * mad
            v = np.clip(v, l, u)
        
        m, s = v.mean(), v.std()
        if s > 0:
            v = (v - m) / s
        else:
            v = np.zeros_like(v)
        
        result = np.full(len(vals), np.nan)
        result[valid] = v
        group['factor_final'] = result
        return group
    
    merged = merged.groupby('date', group_keys=False).apply(standardize)
    
    # ============================================================
    # 7. Output wide CSV
    # ============================================================
    print("Writing output...")
    df_out = merged[['date', 'stock_code', 'factor_final']].dropna(subset=['factor_final'])
    df_out = df_out.pivot(index='date', columns='stock_code', values='factor_final')
    df_out.index = pd.to_datetime(df_out.index)
    df_out = df_out.sort_index()
    df_out = df_out.loc[df_out.count(axis=1) >= 100]
    
    output_path = data_dir / "factor_roe_accel_v1.csv"
    df_out.to_csv(output_path)
    
    print(f"\nDone! {output_path}")
    print(f"Shape: {df_out.shape}")
    print(f"Dates: {df_out.index.min()} ~ {df_out.index.max()}")
    print(f"Mean stocks/date: {df_out.count(axis=1).mean():.0f}")
    
    sample_path = data_dir / "factor_roe_accel_v1_sample.txt"
    with open(sample_path, 'w') as f:
        f.write(f"Shape: {df_out.shape}\nDates: {df_out.index.min()} ~ {df_out.index.max()}\n")
        for i in range(-3, 0):
            vals = df_out.iloc[i].dropna()
            f.write(f"  {df_out.index[i].date()}: N={len(vals)}, mean={vals.mean():.4f}, std={vals.std():.4f}\n")
    print(f"Sample: {sample_path}")


if __name__ == "__main__":
    main()
