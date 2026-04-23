"""
因子：振幅-换手效率 (Range-Turnover Efficiency)
ID: range_turnover_efficiency_v1

逻辑：
  amplitude = (high - low) / close  # 已有字段
  turnover = 换手率
  
  range_eff = amplitude / turnover
  
  值越低 = 每个单位换手率造成的价格波动越小 = 流动性深度好 = 市场质量高
  
  对比Amihud：Amihud用|returns|/volume衡量绝对流动性冲击；
  range_eff用相对价格波动/换手率衡量边际交易冲击。
  两者负相关但有区别，本因子更多反映盘口深度和流动性质量。

Barra风格: Quality (流动性质量维度)
中性化：对数成交额（市值代理）OLS + MAD + z-score
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def neutralize(values, control):
    """OLS neutralize + MAD winsorize + z-score."""
    combined = pd.DataFrame({'v': values, 'c': control})
    combined = combined.dropna()
    if len(combined) < 30:
        return np.full_like(values, np.nan)
    
    v = combined['v'].values
    x = combined['c'].values
    
    X = np.column_stack([np.ones(len(x)), x])
    try:
        beta = np.linalg.lstsq(X, v, rcond=None)[0]
        r = v - X @ beta
    except Exception:
        return np.full_like(values, np.nan)
    
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    if mad < 1e-10:
        return np.full_like(values, np.nan)
    r = np.clip(r, med - 5.2 * mad, med + 5.2 * mad)
    std = r.std()
    if std < 1e-10:
        return np.full_like(values, np.nan)
    
    result = np.full_like(values, np.nan)
    idx = combined.index.values
    z = (r - np.median(r)) / std
    result[idx] = z
    return result


def main():
    print("Loading data...")
    kline = pd.read_csv('data/csi1000_kline_raw.csv')
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
    kline['date'] = pd.to_datetime(kline['date'].astype(str))
    
    # amplitude and turnover are already in the data
    # Clean data
    print("Cleaning data...")
    kline['amplitude'] = pd.to_numeric(kline['amplitude'], errors='coerce')
    kline['turnover'] = pd.to_numeric(kline['turnover'], errors='coerce')
    
    # Filter invalid values
    kline = kline[kline['turnover'] > 0].copy()
    kline = kline[kline['amplitude'] > 0].copy()
    kline = kline[kline['amplitude'] < 30].copy()  # remove extreme spikes (likely data errors)
    
    # Compute range efficiency: amplitude / turnover
    # Lower = more efficient (less price movement per unit volume)
    # Also use log version if needed
    print("Computing range efficiency = amplitude / turnover...")
    kline['factor_raw'] = kline['amplitude'] / kline['turnover']
    
    # Neutralize cross-sectionally per date
    print("Neutralizing by market cap proxy (log_amount)...")
    
    all_out = []
    date_count = 0
    total_dates = kline['date'].nunique()
    
    for date, grp in kline.groupby('date'):
        date_count += 1
        if date_count % 200 == 0:
            print(f"  Date {date_count}/{total_dates}...")
        
        grp = grp.dropna(subset=['factor_raw']).copy()
        if len(grp) < 50:
            continue
        
        log_amount = np.log(grp['amount'].clip(lower=1))
        neutralized = neutralize(grp['factor_raw'].values, log_amount.values)
        
        grp = grp.copy()
        grp['factor_value'] = neutralized
        grp = grp.dropna(subset=['factor_value'])
        all_out.append(grp[['stock_code', 'date', 'factor_value']])
    
    result = pd.concat(all_out, ignore_index=True)
    print(f"\nFinal factor shape: {result.shape}")
    print(f"  Non-null: {result['factor_value'].notna().sum()}")
    print(f"  Mean: {result['factor_value'].mean():.4f}")
    print(f"  Std: {result['factor_value'].std():.4f}")
    print(f"  Dates: {result['date'].nunique()}")
    
    output_path = 'data/factor_range_efficiency_v1.csv'
    result.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print("Done!")


if __name__ == '__main__':
    main()
