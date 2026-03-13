"""
换手率衰减因子改进版 v2
========================
原版 turnover_decay_v1: 
  IC=0.021(t=1.95), Sharpe=0.72, 单调性0.9 — 差一点达标
  
改进思路：
1. 原版用turnover_5d/turnover_20d，可能窗口不够极端
2. 改用更短近端(3d)和更长远端(40d)，放大衰减信号
3. 添加换手率水平：只在绝对换手率不太低时有效（否则缩量是无流动性而非主力控盘）
4. 用EWM替代简单均值，近期权重更高

v2a: MA3d / MA40d (更极端的窗口)
v2b: MA3d / MA20d (缩短近端)  
v2c: MA5d / MA40d (拉长远端)
"""

import pandas as pd
import numpy as np

def compute_factor():
    print("Loading data...")
    df = pd.read_csv('data/csi1000_kline_raw.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # Various turnover windows
    for w in [3, 5, 10, 20, 40, 60]:
        df[f'turn_{w}d'] = df.groupby('stock_code')['turnover'].transform(
            lambda x: x.rolling(w, min_periods=max(w//2, 2)).mean()
        )
    
    # Factor versions
    versions = {
        'v2a': ('turn_3d', 'turn_40d'),   # more extreme
        'v2b': ('turn_3d', 'turn_20d'),   # short near end
        'v2c': ('turn_5d', 'turn_40d'),   # long far end
        'v2d': ('turn_3d', 'turn_60d'),   # most extreme
        'v2e': ('turn_10d', 'turn_60d'),  # medium
    }
    
    # Market cap neutralization proxy
    df['avg_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    df['log_amount'] = np.log(df['avg_amount_20d'].clip(lower=1))
    
    for version, (near, far) in versions.items():
        # Turnover decay ratio: negative ratio = recent turnover lower than historical = 缩量
        df['factor_raw'] = -np.log((df[near] / df[far]).clip(lower=0.01, upper=100))
        # Negative log: when near < far (缩量), log ratio is negative, -log is positive
        # So high factor = 缩量(量能枯竭)
        
        fdf = df[['date', 'stock_code', 'factor_raw', 'log_amount']].copy()
        fdf = fdf.dropna(subset=['factor_raw', 'log_amount'])
        
        def neutralize(g):
            y, x = g['factor_raw'].values, g['log_amount'].values
            v = ~(np.isnan(y)|np.isnan(x))
            if v.sum()<30: g['factor']=np.nan; return g
            X = np.column_stack([np.ones(v.sum()), x[v]])
            try:
                b = np.linalg.lstsq(X, y[v], rcond=None)[0]
                r = np.full(len(y), np.nan); r[v] = y[v] - X@b; g['factor'] = r
            except: g['factor'] = np.nan
            return g
        
        fdf = fdf.groupby('date', group_keys=False).apply(neutralize)
        
        def winsorize_mad(g):
            v = g['factor'].values; m = ~np.isnan(v)
            if m.sum()<10: return g
            med = np.nanmedian(v); mad = np.nanmedian(np.abs(v[m]-med))
            if mad<1e-10: return g
            g['factor'] = np.clip(v, med-3*1.4826*mad, med+3*1.4826*mad)
            return g
        
        fdf = fdf.groupby('date', group_keys=False).apply(winsorize_mad)
        
        def zscore(g):
            v = g['factor'].values; m = ~np.isnan(v)
            if m.sum()<10: return g
            mn, s = np.nanmean(v[m]), np.nanstd(v[m])
            g['factor'] = (v-mn)/s if s>1e-10 else 0.0
            return g
        
        fdf = fdf.groupby('date', group_keys=False).apply(zscore)
        out = fdf[['date','stock_code','factor']].dropna(subset=['factor'])
        out.to_csv(f'data/factor_turnover_decay_{version}.csv', index=False)
        print(f"Turnover Decay {version} ({near}/{far}): {out.shape}")

if __name__ == '__main__':
    compute_factor()
