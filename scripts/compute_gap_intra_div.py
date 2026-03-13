"""
缺口-日内背离因子 (Gap-Intraday Divergence) v1
================================================
逻辑：
- 缺口方向与日内方向背离 = 机构行为
- 跳空高开但日内下跌 = 机构派发(利空)
- 跳空低开但日内上涨 = 机构吸筹(利好)  
- 连续多日的背离积累 = 更强信号

构造：
  gap_ret = (open_t - close_{t-1}) / close_{t-1}
  intra_ret = (close_t - open_t) / open_t
  divergence = gap_ret × intra_ret  (负值=背离)
  
  factor = -mean(divergence, 20d)  
  高值 = 持续背离 = 机构暗中操作
  
  对背离加权：|gap_ret| 越大，背离信号越强
  weighted_divergence = -gap_ret × intra_ret × turnover_t
  (高换手日的背离更可靠)

Barra: 微观结构
"""

import pandas as pd
import numpy as np

def compute_factor():
    print("Loading data...")
    df = pd.read_csv('data/csi1000_kline_raw.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # Previous close
    df['prev_close'] = df.groupby('stock_code')['close'].shift(1)
    
    # Gap return: open vs prev close
    df['gap_ret'] = (df['open'] - df['prev_close']) / df['prev_close']
    
    # Intraday return: close vs open
    df['intra_ret'] = (df['close'] - df['open']) / df['open']
    
    # Divergence: gap_ret × intra_ret (negative = diverging)
    df['divergence'] = df['gap_ret'] * df['intra_ret']
    
    # Volume-weighted divergence
    df['vol_divergence'] = df['divergence'] * df['turnover']
    
    # Version 1: simple 20d mean of -divergence
    df['factor_v1'] = df.groupby('stock_code')['divergence'].transform(
        lambda x: -x.rolling(20, min_periods=10).mean()
    )
    
    # Version 2: volume-weighted 20d mean
    df['factor_v2'] = df.groupby('stock_code')['vol_divergence'].transform(
        lambda x: -x.rolling(20, min_periods=10).mean()
    )
    
    # Version 3: only count "significant" divergences (|gap| > median)
    # First, compute conditional divergence
    gap_median = df.groupby('stock_code')['gap_ret'].transform(
        lambda x: x.abs().rolling(60, min_periods=20).median()
    )
    df['sig_divergence'] = np.where(
        df['gap_ret'].abs() > gap_median,
        df['divergence'],
        0
    )
    df['factor_v3'] = df.groupby('stock_code')['sig_divergence'].transform(
        lambda x: -x.rolling(20, min_periods=10).mean()
    )
    
    # Market cap neutralization
    df['avg_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    df['log_amount'] = np.log(df['avg_amount_20d'].clip(lower=1))
    
    for version, col in [('v1', 'factor_v1'), ('v2', 'factor_v2'), ('v3', 'factor_v3')]:
        fdf = df[['date', 'stock_code', col, 'log_amount']].copy()
        fdf = fdf.rename(columns={col: 'factor_raw'})
        fdf = fdf.dropna(subset=['factor_raw', 'log_amount'])
        
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
        
        fdf = fdf.groupby('date', group_keys=False).apply(neutralize)
        
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
        out.to_csv(f'data/factor_gap_intra_div_{version}.csv', index=False)
        print(f"Gap-Intra Divergence {version}: {out.shape}")

if __name__ == '__main__':
    compute_factor()
