"""
因子: Cross-Sectional Return Spread (CSR Spread)
ID: cs_spread_v1

逻辑:
  每日计算全市场收益率截面标准差（剔除极端值后的 MAD）
  这里 capitalizing on 一个市场微观结构现象:
  - 高截面离散度日 = 市场波动/分化大 = 不确定性高
  - 但 CS spread 的低值（低离散度）可能表明 informed consensus
  
  实际上我们换一个方向：
  截面标准差 Rolling Mean - 标准差的 ratio
  即: 近期截面标准差相对历史水平的比值
  > 1 = 当前分化 > 历史平均 = 市场不确定性高
  < 1 = 当前分化 < 历史平均 = consensus forming
  
  高比值（分化大）= 市场在 price discovery（不确定性高）
  低比值（分化小）= 共识形成（可能过度一致？）
  
  20日均值: 持续高 cs_spread 可能意味着个股 alpha 波动大

方向: 正向（高因子值 = 高不确定性 = 高收益？待检验）
Barra风格: 风险/系统
"""

import pandas as pd
import numpy as np

def calc_cs_spread_factor(kline_path, output_path, short_win=20, long_win=60):
    """计算 Cross-Sectional Return Spread 因子"""
    print("Loading kline data...")
    df = pd.read_csv(kline_path)
    df['date'] = pd.to_datetime(df['date'])
    df['pct_change'] = pd.to_numeric(df['pct_change'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # 构建收益率矩阵（日期×股票）
    print("Building return matrix...")
    returns = df.pivot(index='date', columns='stock_code', values='pct_change')
    print(f"  Shape: {returns.shape}")
    
    # 每日截面标准差 (用MAD estimate避免极端值影响)
    print("Calculating daily cross-sectional dispersion...")
    
    def cs_dispersion(arr):
        """截面标准差估计"""
        arr = np.asarray(arr, dtype=float)
        valid = np.isfinite(arr)
        if valid.sum() < 30:
            return np.nan
        vals = arr[valid]
        # MAD-based std estimate
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        std_est = 1.4826 * mad if mad > 0 else np.std(vals)
        return std_est
    
    daily_disp = returns.apply(cs_dispersion, axis=1)
    print(f"  Daily dispersion computed: {daily_disp.notna().sum()} days")
    
    # Rolling mean ratio: short_win / long_win
    print(f"Computing {short_win}/{long_win} dispersion ratio...")
    disp_ratio = daily_disp.rolling(short_win, min_periods=int(short_win*0.6)).mean() / \
                 daily_disp.rolling(long_win, min_periods=int(long_win*0.6)).mean().clip(lower=1e-6)
    
    # 展开回长格式
    disp_ratio_df = disp_ratio.reset_index()
    disp_ratio_df.columns = ['date', 'disp_ratio']
    disp_ratio_df = disp_ratio_df.dropna()
    
    # 与原始数据合并
    df_key = df[['date', 'stock_code']].copy()
    df_key['date'] = pd.to_datetime(df_key['date'])
    disp_ratio_df['date'] = pd.to_datetime(disp_ratio_df['date'])
    
    factor_df = df_key.merge(disp_ratio_df, on='date', how='inner')
    factor_df = factor_df.rename(columns={'disp_ratio': 'raw_factor'})
    
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
    calc_cs_spread_factor('data/csi1000_kline_raw.csv', 'data/factor_cs_spread_v1.csv')
