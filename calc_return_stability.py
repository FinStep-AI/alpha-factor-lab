"""
因子：收益率稳定性 (Return Stability)
ID: return_stability_v1

逻辑：
  计算过去20个交易日的日收益率标准差（负向）。
  收益率波动越低 = 股价走势越稳定 = 信息效率越高 = 质量特征。
  反直觉：低波动不等于低风险，而是高确定性 / 高信息质量。

  计算方式：
    factor_raw = - std(daily_returns, 20d rolling)
    越高越好表示稳定度高

Barra风格: Quality
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
    
    # Compute daily returns
    print("Computing daily returns...")
    kline['ret'] = kline.groupby('stock_code')['close'].pct_change()
    
    # Rolling 20-day std of returns (negative = higher is more stable)
    print("Computing 20-day rolling std of returns (inverted)...")
    kline['ret_std_20d'] = kline.groupby('stock_code')['ret'].transform(
        lambda x: x.rolling(20, min_periods=15).std()
    )
    kline['factor_raw'] = -kline['ret_std_20d']  # higher = more stable
    
    # Neutralize cross-sectionally per date
    print("Neutralizing by market cap proxy (log_amount)...")
    
    all_out = []
    date_count = 0
    for date, grp in kline.groupby('date'):
        date_count += 1
        if date_count % 200 == 0:
            print(f"  Date {date_count}/{kline['date'].nunique()}...")
        
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
    
    output_path = 'data/factor_return_stability_v1.csv'
    result.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print("Done!")


if __name__ == '__main__':
    main()
