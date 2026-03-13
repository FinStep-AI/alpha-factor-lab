"""
价格冲击因子 (Kyle Lambda / Price Impact) v1
=============================================
逻辑：
- Kyle's Lambda = |daily_return| / log(1 + volume) 的斜率
- 高价格冲击 = 少量成交就能大幅推动价格 = 流动性差 = 信息不对称高
- 在中证1000中，类似Amihud但用不同的构造方式
- 可能与Amihud高相关，但值得一试

改进构造：
- 不用简单比值，用20日rolling回归斜率
  daily_return = alpha + lambda × signed_volume + epsilon
  其中 signed_volume = sign(daily_ret) × log(1+amount)
- lambda就是价格冲击系数

也测试：
- v2: |daily_ret| / (turnover^0.5) 的20日均值(非线性冲击)
- v3: 只看下跌日的价格冲击(下行流动性)

Barra: Liquidity
"""

import pandas as pd
import numpy as np
from numpy.linalg import lstsq

def compute_factor():
    print("Loading data...")
    df = pd.read_csv('data/csi1000_kline_raw.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    df['daily_ret'] = df.groupby('stock_code')['close'].pct_change()
    df['log_amount'] = np.log(df['amount'].clip(lower=1))
    df['signed_vol'] = np.sign(df['daily_ret']) * df['log_amount']
    
    window = 20
    
    results = []
    for stock_code, gdf in df.groupby('stock_code'):
        gdf = gdf.sort_values('date').reset_index(drop=True)
        n = len(gdf)
        
        lambda_vals = []
        v2_vals = []
        v3_vals = []
        dates = []
        
        for i in range(window - 1, n):
            w = gdf.iloc[i - window + 1: i + 1]
            rets = w['daily_ret'].values
            svol = w['signed_vol'].values
            turnover = w['turnover'].values
            
            valid = ~(np.isnan(rets) | np.isnan(svol))
            
            # v1: Kyle lambda (regression slope)
            if valid.sum() >= 10:
                r = rets[valid]
                sv = svol[valid]
                X = np.column_stack([np.ones(len(sv)), sv])
                try:
                    beta = lstsq(X, r, rcond=None)[0]
                    lambda_vals.append(beta[1])
                except:
                    lambda_vals.append(np.nan)
            else:
                lambda_vals.append(np.nan)
            
            # v2: |ret| / sqrt(turnover)
            valid2 = ~(np.isnan(rets) | np.isnan(turnover)) & (turnover > 0)
            if valid2.sum() >= 10:
                impact = np.abs(rets[valid2]) / np.sqrt(turnover[valid2].clip(min=0.01))
                v2_vals.append(np.mean(impact))
            else:
                v2_vals.append(np.nan)
            
            # v3: downside price impact only
            valid3 = valid2 & (rets < 0)
            if valid3.sum() >= 3:
                down_impact = np.abs(rets[valid3]) / np.sqrt(turnover[valid3].clip(min=0.01))
                v3_vals.append(np.mean(down_impact))
            else:
                v3_vals.append(np.nan)
            
            dates.append(gdf.iloc[i]['date'])
        
        stock_result = pd.DataFrame({
            'date': dates,
            'stock_code': stock_code,
            'kyle_lambda': lambda_vals,
            'impact_v2': v2_vals,
            'down_impact': v3_vals
        })
        results.append(stock_result)
    
    factor_df = pd.concat(results, ignore_index=True)
    
    # Amount for neutralization
    amt_df = df[['date', 'stock_code', 'amount']].copy()
    amt_df['avg_amount_20d'] = amt_df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    amt_df['log_amt'] = np.log(amt_df['avg_amount_20d'].clip(lower=1))
    
    factor_df = factor_df.merge(amt_df[['date', 'stock_code', 'log_amt']], on=['date', 'stock_code'], how='left')
    
    for version, col in [('v1', 'kyle_lambda'), ('v2', 'impact_v2'), ('v3', 'down_impact')]:
        fdf = factor_df[['date', 'stock_code', col, 'log_amt']].copy()
        fdf = fdf.rename(columns={col: 'factor_raw'})
        fdf = fdf.dropna(subset=['factor_raw', 'log_amt'])
        
        # Log transform for v2/v3 (positive values, skewed)
        if version in ['v2', 'v3']:
            fdf['factor_raw'] = np.log(fdf['factor_raw'].clip(lower=1e-10))
        
        # Neutralize
        def neutralize(group):
            y = group['factor_raw'].values
            x = group['log_amt'].values
            valid = ~(np.isnan(y) | np.isnan(x))
            if valid.sum() < 30:
                group['factor'] = np.nan
                return group
            y_v = y[valid]; x_v = x[valid]
            X = np.column_stack([np.ones(len(x_v)), x_v])
            try:
                beta = lstsq(X, y_v, rcond=None)[0]
                res = np.full(len(y), np.nan)
                res[valid] = y_v - X @ beta
                group['factor'] = res
            except:
                group['factor'] = np.nan
            return group
        
        fdf = fdf.groupby('date', group_keys=False).apply(neutralize)
        
        # Winsorize
        def winsorize_mad(group):
            vals = group['factor'].values
            valid = ~np.isnan(vals)
            if valid.sum() < 10: return group
            med = np.nanmedian(vals)
            mad = np.nanmedian(np.abs(vals[valid] - med))
            if mad < 1e-10: return group
            group['factor'] = np.clip(vals, med - 3*1.4826*mad, med + 3*1.4826*mad)
            return group
        
        fdf = fdf.groupby('date', group_keys=False).apply(winsorize_mad)
        
        # Z-score
        def zscore(group):
            vals = group['factor'].values
            valid = ~np.isnan(vals)
            if valid.sum() < 10: return group
            m = np.nanmean(vals[valid]); s = np.nanstd(vals[valid])
            if s < 1e-10: group['factor'] = 0.0
            else: group['factor'] = (vals - m) / s
            return group
        
        fdf = fdf.groupby('date', group_keys=False).apply(zscore)
        
        out = fdf[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
        out.to_csv(f'data/factor_price_impact_{version}.csv', index=False)
        print(f"Price impact {version} saved: {out.shape}")

if __name__ == '__main__':
    compute_factor()
