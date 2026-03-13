"""
异常换手率因子 (Abnormal Turnover Level) v1
============================================
逻辑：
- 近期换手率异常高的股票，未来收益较低(过度关注/overreaction)
- 近期换手率异常低的股票，未来收益较高(被忽视/neglect premium)
- 这是A股著名的"换手率效应"，在小盘股上尤为显著

构造：
- 因子 = -log(MA_5d_turnover / MA_60d_turnover)
  即近期换手率相对历史的异常程度，取反
  高值 = 近期缩量(被忽视) → 做多
  低值 = 近期放量(被追捧) → 做空

也测试：
- v2: 纯绝对换手率 = -log(MA_20d_turnover)
- v3: 换手率与收益交互 = -|cum_ret_5d| × abn_turnover (过度反应幅度)

Barra: Liquidity / 反转
"""

import pandas as pd
import numpy as np

def compute_factor():
    print("Loading data...")
    df = pd.read_csv('data/csi1000_kline_raw.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # Rolling turnover stats
    df['turnover_5d'] = df.groupby('stock_code')['turnover'].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )
    df['turnover_20d'] = df.groupby('stock_code')['turnover'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    df['turnover_60d'] = df.groupby('stock_code')['turnover'].transform(
        lambda x: x.rolling(60, min_periods=30).mean()
    )
    
    # v1: abnormal turnover (5d vs 60d), reversed
    df['factor_v1'] = -np.log((df['turnover_5d'] / df['turnover_60d']).clip(lower=0.01, upper=100))
    
    # v2: absolute turnover level, reversed
    df['factor_v2'] = -np.log(df['turnover_20d'].clip(lower=0.01))
    
    # v3: 5d vs 20d (shorter-term comparison)
    df['factor_v3'] = -np.log((df['turnover_5d'] / df['turnover_20d']).clip(lower=0.01, upper=100))
    
    # Market cap neutralization using log_amount
    df['avg_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    df['log_amount'] = np.log(df['avg_amount_20d'].clip(lower=1))
    
    for version, col in [('v1', 'factor_v1'), ('v2', 'factor_v2'), ('v3', 'factor_v3')]:
        factor_df = df[['date', 'stock_code', col, 'log_amount']].copy()
        factor_df = factor_df.rename(columns={col: 'factor_raw'})
        factor_df = factor_df.dropna(subset=['factor_raw', 'log_amount'])
        
        # Neutralize
        def neutralize(group):
            y = group['factor_raw'].values
            x = group['log_amount'].values
            valid = ~(np.isnan(y) | np.isnan(x))
            if valid.sum() < 30:
                group['factor'] = np.nan
                return group
            y_v = y[valid]; x_v = x[valid]
            X = np.column_stack([np.ones(len(x_v)), x_v])
            try:
                beta = np.linalg.lstsq(X, y_v, rcond=None)[0]
                res = np.full(len(y), np.nan)
                res[valid] = y_v - X @ beta
                group['factor'] = res
            except:
                group['factor'] = np.nan
            return group
        
        factor_df = factor_df.groupby('date', group_keys=False).apply(neutralize)
        
        # Winsorize MAD 3x
        def winsorize_mad(group):
            vals = group['factor'].values
            valid = ~np.isnan(vals)
            if valid.sum() < 10: return group
            med = np.nanmedian(vals)
            mad = np.nanmedian(np.abs(vals[valid] - med))
            if mad < 1e-10: return group
            group['factor'] = np.clip(vals, med - 3*1.4826*mad, med + 3*1.4826*mad)
            return group
        
        factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_mad)
        
        # Z-score
        def zscore(group):
            vals = group['factor'].values
            valid = ~np.isnan(vals)
            if valid.sum() < 10: return group
            m = np.nanmean(vals[valid]); s = np.nanstd(vals[valid])
            if s < 1e-10: group['factor'] = 0.0
            else: group['factor'] = (vals - m) / s
            return group
        
        factor_df = factor_df.groupby('date', group_keys=False).apply(zscore)
        
        out = factor_df[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
        out.to_csv(f'data/factor_abn_turn_{version}.csv', index=False)
        print(f"{version} saved: {out.shape}")

if __name__ == '__main__':
    compute_factor()
