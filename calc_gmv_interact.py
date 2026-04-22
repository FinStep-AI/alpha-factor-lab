"""
因子：隔夜缺口VWAP交互因子 (Gap-Momentum-VWAP Composite Interaction, GMV_I)
ID: gmv_interact_v1

逻辑：三个已验证alpha源的交互：
  - overnight_momentum_v1 (IC=0.032)
  - gap_momentum_v1
  - vwap_dev_20 (翻转后有效)

交互 = sign × √|rank_X × rank_Y × rank_Z|
同向时绝对值最大 = 强确认；拮抗时接近0 = 噪声过滤

Barra风格: MICRO (多源信息融合)
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def rolling_neutralize(values, controls):
    """Vectorized rolling neutralization per date."""
    results = []
    for date in values['date'].unique():
        mask = values['date'] == date
        vals = values.loc[mask, 'factor_raw'].values
        ctrl = controls.loc[mask, 'log_amount'].values
        
        valid = np.isfinite(vals) & np.isfinite(ctrl)
        if valid.sum() < 30:
            continue
        
        v, x = vals[valid], ctrl[valid]
        X = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(X, v, rcond=None)[0]
            r = v - X @ beta
        except:
            continue
        
        med = np.median(r)
        mad = np.median(np.abs(r - med))
        if mad < 1e-10:
            continue
        r = np.clip(r, med - 5.2*mad, med + 5.2*mad)
        std = r.std()
        if std < 1e-10:
            continue
        z = (r - np.median(r)) / std
        
        idx = np.where(mask)[0][valid]
        results.append(pd.DataFrame({
            'date': values.loc[idx, 'date'].values,
            'stock_code': values.loc[idx, 'stock_code'].values,
            'factor_value': z
        }))
    
    return pd.concat(results, ignore_index=True)


def main():
    print("Loading factor inputs...")
    f_ovn = pd.read_csv('data/factor_overnight_momentum_v1.csv')
    f_gap = pd.read_csv('data/factor_gap_momentum_v1.csv')
    f_vw  = pd.read_csv('data/factor_vwap_dev_20.csv')
    
    print("Merging factors...")
    merged = f_ovn.rename(columns={'factor_value': 'ovn'})\
                  .merge(f_gap.rename(columns={'factor_value': 'gap'}), on=['date','stock_code'])\
                  .merge(f_vw .rename(columns={'factor_value': 'vw'}),  on=['date','stock_code'])\
                  .dropna()
    print(f"After dropna: {merged.shape}, dates: {merged['date'].nunique()}")
    
    # Rank-transform each factor (0-1) then center at 0
    print("Rank-transform within cross-section...")
    for col in ['ovn', 'gap', 'vw']:
        merged[f'r_{col}'] = merged.groupby('date')[col].transform(lambda x: x.rank(pct=True) - 0.5)
    
    # Triple product interaction
    merged['interact_raw'] = merged['r_ovn'] * merged['r_gap'] * merged['r_vw']
    merged['factor_raw'] = np.sign(merged['interact_raw']) * np.sqrt(np.abs(merged['interact_raw']))
    
    # Neutralize
    print("Neutralizing...")
    amounts = pd.read_csv('data/csi1000_kline_raw.csv')[['date','stock_code','amount']].copy()
    amounts['log_amount'] = np.log(amounts['amount'].clip(lower=1))
    merged = merged.merge(amounts[['date','stock_code','log_amount']], on=['date','stock_code'], how='left')
    merged['date'] = merged['date'].astype(str)
    
    result = rolling_neutralize(merged[['date','stock_code','factor_raw']], merged[['date','stock_code','log_amount']])
    
    print(f"Final: {result.shape}, mean={result['factor_value'].mean():.4f}, std={result['factor_value'].std():.4f}")
    out = 'data/factor_gmv_interact_v1.csv'
    result.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
