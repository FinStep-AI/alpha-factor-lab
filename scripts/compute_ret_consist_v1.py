"""
涨跌持续性因子 (Return Consistency / Hurst-like) v1
====================================================
逻辑：
- 计算过去20天收益的"持续性"
- 用 sign consistency: sum(sign(ret_t) == sign(ret_{t-1})) / (N-1)
  即相邻两天收益同号的比例
- 高一致性 = 趋势型股票(连涨或连跌)
- 低一致性 = 震荡型股票(涨跌交替)
- 在中证1000中，预期趋势型股票(高一致性)后续有动量延续

方法2(改进)：
- 计算 cumulative return / sum(|daily return|)
  = 价格效率(net movement / total movement)
- 高效率 = 单方向移动(趋势)
- 低效率 = 来回波动(噪声)

组合：同时计算两个版本，取效果更好的

Barra: Momentum (但角度不同于传统动量)
"""

import pandas as pd
import numpy as np

def compute_factor():
    print("Loading data...")
    df = pd.read_csv('data/csi1000_kline_raw.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # Daily return
    df['daily_ret'] = df.groupby('stock_code')['close'].pct_change()
    
    window = 20
    
    # Method 1: Sign consistency
    # For each stock, rolling: fraction of days where sign(ret_t) == sign(ret_{t-1})
    def sign_consistency(rets):
        valid = rets[~np.isnan(rets)]
        if len(valid) < 5:
            return np.nan
        signs = np.sign(valid)
        same_sign = np.sum(signs[1:] == signs[:-1])
        return same_sign / (len(valid) - 1)
    
    # Method 2: Return consistency ratio = |cum_ret| / sum(|daily_ret|)
    def return_consistency(rets):
        valid = rets[~np.isnan(rets)]
        if len(valid) < 5:
            return np.nan
        cum = np.abs(np.sum(valid))
        total = np.sum(np.abs(valid))
        if total < 1e-10:
            return 0.0
        return cum / total
    
    # Use vectorized rolling for speed
    results = []
    for stock_code, gdf in df.groupby('stock_code'):
        gdf = gdf.sort_values('date').reset_index(drop=True)
        rets = gdf['daily_ret'].values
        n = len(rets)
        
        sc_vals = []
        rc_vals = []
        dates = []
        
        for i in range(window - 1, n):
            w_rets = rets[i - window + 1: i + 1]
            sc = sign_consistency(w_rets)
            rc = return_consistency(w_rets)
            sc_vals.append(sc)
            rc_vals.append(rc)
            dates.append(gdf.iloc[i]['date'])
        
        stock_result = pd.DataFrame({
            'date': dates,
            'stock_code': stock_code,
            'sign_consist': sc_vals,
            'ret_consist': rc_vals
        })
        results.append(stock_result)
    
    factor_df = pd.concat(results, ignore_index=True)
    print(f"Factor computed: {factor_df.shape}")
    print(f"Sign consist stats: mean={factor_df['sign_consist'].mean():.4f}, std={factor_df['sign_consist'].std():.4f}")
    print(f"Ret consist stats: mean={factor_df['ret_consist'].mean():.4f}, std={factor_df['ret_consist'].std():.4f}")
    
    # Try combining: use ret_consist as main factor (more informative)
    factor_df['factor_raw'] = factor_df['ret_consist']
    
    # Market cap neutralization
    amt_df = df[['date', 'stock_code', 'amount']].copy()
    amt_df = amt_df.sort_values(['stock_code', 'date'])
    amt_df['avg_amount_20d'] = amt_df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    amt_df['log_amount'] = np.log(amt_df['avg_amount_20d'].clip(lower=1))
    
    factor_df = factor_df.merge(amt_df[['date', 'stock_code', 'log_amount']], 
                                 on=['date', 'stock_code'], how='left')
    
    # Cross-sectional neutralization
    def neutralize(group):
        y = group['factor_raw'].values
        x = group['log_amount'].values
        valid = ~(np.isnan(y) | np.isnan(x))
        if valid.sum() < 30:
            group['factor'] = np.nan
            return group
        y_v = y[valid]
        x_v = x[valid]
        X = np.column_stack([np.ones(len(x_v)), x_v])
        try:
            beta = np.linalg.lstsq(X, y_v, rcond=None)[0]
            residuals = np.full(len(y), np.nan)
            residuals[valid] = y_v - X @ beta
            group['factor'] = residuals
        except:
            group['factor'] = np.nan
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(neutralize)
    
    # Winsorize MAD 3x
    def winsorize_mad(group):
        vals = group['factor'].values
        valid = ~np.isnan(vals)
        if valid.sum() < 10:
            return group
        median = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals[valid] - median))
        if mad < 1e-10:
            return group
        upper = median + 3 * 1.4826 * mad
        lower = median - 3 * 1.4826 * mad
        group['factor'] = np.clip(vals, lower, upper)
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_mad)
    
    # Z-score
    def zscore(group):
        vals = group['factor'].values
        valid = ~np.isnan(vals)
        if valid.sum() < 10:
            return group
        mean = np.nanmean(vals[valid])
        std = np.nanstd(vals[valid])
        if std < 1e-10:
            group['factor'] = 0.0
        else:
            group['factor'] = (vals - mean) / std
        return group
    
    factor_df = factor_df.groupby('date', group_keys=False).apply(zscore)
    
    # Also save sign_consist version
    factor_df['factor_raw_sc'] = factor_df['sign_consist']
    factor_df2 = factor_df.copy()
    factor_df2['factor_raw'] = factor_df2['factor_raw_sc']
    factor_df2 = factor_df2.groupby('date', group_keys=False).apply(
        lambda g: neutralize(g.assign(factor_raw=g['factor_raw_sc']))
    )
    factor_df2 = factor_df2.groupby('date', group_keys=False).apply(winsorize_mad)
    factor_df2 = factor_df2.groupby('date', group_keys=False).apply(zscore)
    
    # Save both
    output1 = factor_df[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    output1.to_csv('data/factor_ret_consist_v1.csv', index=False)
    print(f"\nRet consist saved: {output1.shape}")
    
    output2 = factor_df2[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    output2.to_csv('data/factor_sign_consist_v1.csv', index=False)
    print(f"Sign consist saved: {output2.shape}")

if __name__ == '__main__':
    compute_factor()
