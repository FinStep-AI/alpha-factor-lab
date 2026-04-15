"""
量价背离因子 (vol_price_diverge_v1) - 向量化快速版
---------------------------------------------------
逻辑：过去20天内，日收益率与成交量变化率的滚动Pearson相关性取负。
量价背离（负相关）→ 因子值为正 → 预示趋势不可持续/反转信号。

优化：用pandas rolling + corr 向量化计算，避免逐窗口scipy
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def main():
    kline = pd.read_csv('data/csi1000_kline_raw.csv')
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # Daily return
    kline['ret'] = kline['pct_change'] / 100.0
    
    # Volume relative to 5-day MA
    kline['vol_ma5'] = kline.groupby('stock_code')['volume'].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )
    kline['vol_ratio'] = kline['volume'] / kline['vol_ma5'] - 1
    
    # Rolling 20-day Pearson correlation using pandas
    # Need to pivot to compute per-stock rolling corr efficiently
    print("Computing rolling correlations (vectorized)...")
    
    window = 20
    
    def compute_corr(group):
        r = group['ret']
        v = group['vol_ratio']
        corr = r.rolling(window, min_periods=12).corr(v)
        group['vp_corr'] = -corr  # negate: divergence -> positive
        return group
    
    kline = kline.groupby('stock_code', group_keys=False).apply(compute_corr)
    
    # Filter valid
    factor_df = kline[['date', 'stock_code', 'vp_corr', 'amount', 'turnover']].dropna(subset=['vp_corr']).copy()
    factor_df = factor_df.rename(columns={'vp_corr': 'factor_raw'})
    
    print(f"Raw factor: {len(factor_df)} rows, {factor_df['date'].nunique()} dates")
    
    # Market-cap neutralization
    factor_df['mktcap_proxy'] = np.where(
        factor_df['turnover'] > 0, 
        factor_df['amount'] / factor_df['turnover'], 
        np.nan
    )
    factor_df['log_mktcap'] = np.log(factor_df['mktcap_proxy'].clip(lower=1))
    
    def neutralize(group):
        valid = group.dropna(subset=['factor_raw', 'log_mktcap'])
        if len(valid) < 30:
            group['factor'] = np.nan
            return group
        x = valid['log_mktcap'].values
        y = valid['factor_raw'].values
        
        # Winsorize
        y_low, y_high = np.percentile(y, [1, 99])
        y = np.clip(y, y_low, y_high)
        
        # OLS residual
        x_dm = x - x.mean()
        denom = np.sum(x_dm ** 2)
        beta = np.sum(x_dm * y) / denom if denom > 0 else 0
        residual = y - beta * x_dm - y.mean()
        
        std = residual.std()
        if std > 0:
            residual = residual / std
        
        group.loc[valid.index, 'factor'] = residual
        return group
    
    print("Neutralizing...")
    factor_df = factor_df.groupby('date', group_keys=False).apply(neutralize)
    
    output = factor_df[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    output = output.sort_values(['date', 'stock_code']).reset_index(drop=True)
    
    output.to_csv('data/factor_vol_price_diverge_v1.csv', index=False)
    print(f"\nOutput: data/factor_vol_price_diverge_v1.csv")
    print(f"Rows: {len(output)}, Dates: {output['date'].nunique()}")
    print(f"Date range: {output['date'].min()} to {output['date'].max()}")
    print(output['factor'].describe())

if __name__ == '__main__':
    main()
