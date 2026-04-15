"""
因子: Market Sync Ratio (SR) - Systematic Risk Proportion
ID: msr_v1

逻辑:
  每只股票相对于市场的同步程度。
  
  每日截面: 股票 i 的收益率 ~ 截面均值收益率
  R² = correlation(stock_i_ret, cross_section_mean_ret)²
  
  R² 高 = 股票高度跟随市场 = 低特质 alpha = 低信息独特性
  R² 低 = 股票特立独行 = 高特质 alpha 潜力 = 高预期收益
  
  20日滚动 R²:
  继续做: 20日均值 (减少噪音)
  经市值中性化处理。

方向: 反向（低 R² = 高预期收益）
Barra风格: 特异性风险（非传统）
"""

import pandas as pd
import numpy as np

def calc_msr_factor(kline_path, output_path, window=20):
    """计算 Market Sync Ratio (低同步 = 高分)"""
    print("Loading kline data...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df['pct_change'] = pd.to_numeric(df['pct_change'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    print("Building return matrix...")
    returns = df.pivot(index='date', columns='stock_code', values='pct_change')
    n_dates, n_stocks = returns.shape
    print(f"  Shape: {returns.shape}")
    
    # Compute cross-sectional mean return each day
    cs_mean = returns.mean(axis=1)
    
    print("Computing rolling R² (stock vs cross-section)...")
    results = []
    
    for date_idx in range(window - 1, n_dates):
        date = returns.index[date_idx]
        date_pos = df[df['date'] == date]
        stock_list = date_pos['stock_code'].values
        
        # Window slice
        w_start = date_idx - window + 1
        w_end = date_idx + 1
        ret_window = returns.iloc[w_start:w_end]  # window × n_stocks
        
        cs_mean_window = cs_mean.iloc[w_start:w_end].values
        if np.isnan(cs_mean_window).sum() > window * 0.2:
            continue
        
        for stock in stock_list:
            stock_ret = ret_window[stock].values
            
            # Check validity
            valid = np.isfinite(stock_ret) & np.isfinite(cs_mean_window)
            if valid.sum() < int(window * 0.6):
                continue
            
            # Correlation and R²
            cc = np.corrcoef(stock_ret[valid], cs_mean_window[valid])
            r = cc[0, 1]
            if np.isnan(r) or not np.isfinite(r):
                continue
            
            r2 = r ** 2
            # Low synchronicity score = 1 - R² (high = idiosyncratic)
            results.append({
                'date': date,
                'stock_code': stock,
                'r2': r2,
                'idiosyncratic': 1 - r2  # low sync = high score
            })
    
    factor_df = pd.DataFrame(results)
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    factor_df = factor_df.rename(columns={'idiosyncratic': 'raw_factor'})
    
    print(f"Raw factor rows: {len(factor_df)}")
    
    # Neutralize by log(amount)
    print("Neutralizing by log(amount)...")
    amt_df = df[['date', 'stock_code', 'amount']].copy()
    amt_df['log_amount_20d'] = amt_df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean().clip(lower=1))
    )
    amt_df['date'] = pd.to_datetime(amt_df['date'])
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    
    factor_df = factor_df.merge(
        amt_df[['date', 'stock_code', 'log_amount_20d']],
        on=['date', 'stock_code'], how='left'
    )
    
    def neutralize_cross_section(group):
        y = group['raw_factor'].values.astype(float)
        x = group['log_amount_20d'].values.astype(float)
        
        valid = np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 30:
            group['factor'] = np.nan
            return group
        
        y_valid = y[valid]
        x_valid = x[valid]
        
        X = np.column_stack([np.ones(len(x_valid)), x_valid])
        try:
            beta = np.linalg.lstsq(X, y_valid, rcond=None)[0]
            residuals = np.full(len(y), np.nan)
            X_full = np.column_stack([np.ones(valid.sum()), x_valid])
            residuals[valid] = y_valid - X_full @ beta
        except:
            group['factor'] = np.nan
            return group
        
        group['factor'] = residuals
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(neutralize_cross_section)
    
    # MAD winsorize + z-score
    def mad_zscore(group):
        vals = group['factor'].values
        valid = np.isfinite(vals)
        if valid.sum() < 30:
            group['factor'] = np.nan
            return group
        
        med = np.nanmedian(vals[valid])
        mad = np.nanmedian(np.abs(vals[valid] - med))
        if mad < 1e-8:
            group['factor'] = 0.0
            return group
        
        upper = med + 5 * 1.4826 * mad
        lower = med - 5 * 1.4826 * mad
        vals_clipped = np.clip(vals, lower, upper)
        
        mean = np.nanmean(vals_clipped[valid])
        std = np.nanstd(vals_clipped[valid])
        if std < 1e-8:
            group['factor'] = 0.0
            return group
        
        group['factor'] = (vals_clipped - mean) / std
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(mad_zscore)
    
    # Output
    output = factor_df[['date', 'stock_code', 'factor']].dropna()
    output = output.rename(columns={'factor': 'factor_value'})
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')
    output.to_csv(output_path, index=False)
    
    print(f"\nFactor saved to {output_path}")
    print(f"  Rows: {len(output)}")
    print(f"  Date range: {output['date'].min()} ~ {output['date'].max()}")
    print(f"  Stocks per date (mean): {output.groupby('date')['stock_code'].count().mean():.0f}")
    print(f"\nFactor distribution:")
    print(f"  Mean: {output['factor_value'].mean():.4f}")
    print(f"  Std:  {output['factor_value'].std():.4f}")
    
    return output

if __name__ == '__main__':
    calc_msr_factor('data/csi1000_kline_raw.csv', 'data/factor_msr_v1.csv')
