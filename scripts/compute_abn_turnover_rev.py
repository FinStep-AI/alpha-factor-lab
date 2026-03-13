"""
异常换手反转因子 (Abnormal Turnover Reversal) v1
================================================
逻辑：
- 近期换手率相对历史异常高的股票，通常经历了过度关注/overreaction
- 高异常换手+负收益 → 恐慌抛售已过度 → 后续反弹
- 高异常换手+正收益 → 追涨已过度 → 后续回落
- 构造：-sign(cum_ret_5d) × log(avg_turnover_5d / avg_turnover_60d)
  = 惩罚近期放量追涨、奖励近期缩量或放量下跌后

改进版(v2思路)：
- 直接用 -cum_ret_5d × abnormal_turnover
  = 放量下跌取反做多(过度恐慌反弹)，放量上涨取反做空(追涨回落)

Barra: 反转/Liquidity
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
    
    # Rolling calculations per stock
    df = df.sort_values(['stock_code', 'date'])
    
    # 5-day cumulative return
    df['cum_ret_5d'] = df.groupby('stock_code')['daily_ret'].transform(
        lambda x: x.rolling(5, min_periods=3).sum()
    )
    
    # Abnormal turnover: 5d avg / 60d avg
    df['turnover_5d'] = df.groupby('stock_code')['turnover'].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )
    df['turnover_60d'] = df.groupby('stock_code')['turnover'].transform(
        lambda x: x.rolling(60, min_periods=30).mean()
    )
    df['abn_turnover'] = np.log((df['turnover_5d'] / df['turnover_60d']).clip(lower=0.01))
    
    # Factor: -cum_ret_5d × (1 + abn_turnover)
    # High abn_turnover amplifies the reversal signal
    # Negative return + high volume = stronger buy signal (panic selling)
    df['factor_raw'] = -df['cum_ret_5d'] * (1 + df['abn_turnover'].clip(lower=0))
    
    # Also try simpler version: just -cum_ret_5d weighted by volume spike
    df['factor_raw_v2'] = -df['cum_ret_5d'] * df['abn_turnover']
    
    # And even simpler: pure short-term reversal weighted by abnormal volume
    df['factor_raw_v3'] = -df['cum_ret_5d'] * np.where(df['abn_turnover'] > 0, df['abn_turnover'], 0)
    
    print(f"Factor raw stats: mean={df['factor_raw'].mean():.6f}, std={df['factor_raw'].std():.6f}")
    
    # Filter out early dates (need 60d warmup)
    df = df.dropna(subset=['factor_raw'])
    
    # Market cap neutralization using log_amount
    df['avg_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    df['log_amount'] = np.log(df['avg_amount_20d'].clip(lower=1))
    
    # Process each factor version
    for version, col in [('v1', 'factor_raw'), ('v2', 'factor_raw_v2'), ('v3', 'factor_raw_v3')]:
        factor_df = df[['date', 'stock_code', col, 'log_amount']].copy()
        factor_df = factor_df.rename(columns={col: 'factor_raw'})
        factor_df = factor_df.dropna(subset=['factor_raw'])
        
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
        out.to_csv(f'data/factor_abn_turnover_rev_{version}.csv', index=False)
        print(f"\n{version} saved: {out.shape}, range: {out['date'].min()} ~ {out['date'].max()}")

if __name__ == '__main__':
    compute_factor()
