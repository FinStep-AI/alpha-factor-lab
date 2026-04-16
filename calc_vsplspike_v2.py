"""
VSPLSPIKE v2: 更严格的高量阈值
与 v1 相同权重方案, 但 threshold 上调到 0.5 (更严格的"信息异动日")
预期: 减少被噪音日污染, 得到更极端的 G5, 提高单调性
"""
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calc_vsplspike_v2(kline_path='data/csi1000_kline_raw.csv',
                       output_path='data/factor_vsplspike_v2.csv',
                       window=20,
                       spike_threshold=0.5,
                       forward_days=5,
                       min_periods_w=5):
    print(f"=== vsplspike_v2: threshold={spike_threshold}, window={window}d ===")
    
    kline = pd.read_csv(kline_path)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    kline['ret'] = kline.groupby('stock_code')['close'].pct_change()
    kline['fwd_ret'] = kline.groupby('stock_code')['close'].shift(-forward_days) / kline['close'] - 1
    
    print("Computing volume z-score...")
    kline['vol_z'] = kline.groupby('stock_code')['volume'].transform(
        lambda x: (x - x.rolling(window, min_periods=15).mean()) /
                  (x.rolling(window, min_periods=15).std() + 1e-9)
    )
    
    kline['w_high'] = np.where(kline['vol_z'] > spike_threshold,
                                 kline['vol_z'] - spike_threshold, 0.0)
    kline['w_low'] = np.where(kline['vol_z'] < -spike_threshold / 2,
                                np.abs(kline['vol_z'] + spike_threshold / 2), 0.0)
    
    kline['w_ret_high'] = kline['w_high'] * kline['fwd_ret']
    kline['w_ret_low'] = kline['w_low'] * kline['fwd_ret']
    
    print("Rolling weighted means...")
    kline['num_high'] = kline.groupby('stock_code')['w_high'].transform(
        lambda x: x.rolling(window, min_periods=min_periods_w).sum())
    kline['num_low'] = kline.groupby('stock_code')['w_low'].transform(
        lambda x: x.rolling(window, min_periods=min_periods_w).sum())
    kline['sum_wret_high'] = kline.groupby('stock_code')['w_ret_high'].transform(
        lambda x: x.rolling(window, min_periods=min_periods_w).sum())
    kline['sum_wret_low'] = kline.groupby('stock_code')['w_ret_low'].transform(
        lambda x: x.rolling(window, min_periods=min_periods_w).sum())
    
    kline['weighted_cont_high'] = kline['sum_wret_high'] / (kline['num_high'] + 1e-9)
    kline['weighted_cont_low'] = kline['sum_wret_low'] / (kline['num_low'] + 1e-9)
    kline['factor_raw'] = kline['weighted_cont_high'] - kline['weighted_cont_low']
    
    print(f"factor_raw stats: mean={kline['factor_raw'].mean():.6f}, std={kline['factor_raw'].std():.6f}")
    
    # Neutralization + z-score
    print("Amount OLS neutralization...")
    all_results = []
    for date, group in kline.groupby('date'):
        if len(group) < 30:
            group = group.assign(factor_neutral=np.nan)
            all_results.append(group)
            continue
        g = group[['stock_code', 'factor_raw', 'amount']].dropna()
        g = g[np.isfinite(g['factor_raw'])]
        if len(g) < 10:
            group = group.assign(factor_neutral=np.nan)
            all_results.append(group)
            continue
        y = g['factor_raw'].values
        x = np.log(g['amount'].values + 1)
        mask = np.isfinite(y) & np.isfinite(x)
        g = g.copy()
        try:
            slope, intercept, _, _, _ = stats.linregress(x[mask], y[mask])
            residual = np.full(len(g), np.nan)
            residual[mask] = y[mask] - (intercept + slope * x[mask])
        except (ValueError, np.linalg.LinAlgError):
            residual = np.full(len(g), np.nan)
        g['factor_neutral'] = residual
        merged = group.merge(g[['stock_code', 'factor_neutral']], 
                             on='stock_code', how='left', suffixes=('', '_g'))
        gn = 'factor_neutral_g'
        if gn in merged.columns:
            merged['factor_neutral'] = merged[gn]
        if 'factor_neutral' not in merged.columns:
            merged['factor_neutral'] = np.nan
        all_results.append(merged)
    
    kline = pd.concat(all_results, ignore_index=True)
    
    print("MAD Winsorize + z-score...")
    def mad_zscore(series):
        s = series.dropna()
        median = s.median()
        mad = np.abs(s - median).median()
        if mad < 1e-9:
            return pd.Series(np.nan, index=series.index)
        upper = median + 5.5 * mad
        lower = median - 5.5 * mad
        clipped = series.clip(lower, upper)
        std = clipped.std()
        if std < 1e-9:
            return pd.Series(np.nan, index=series.index)
        return (clipped - clipped.mean()) / std
    
    kline['factor_value'] = kline.groupby('date')['factor_neutral'].transform(mad_zscore)
    
    output = kline[['date', 'stock_code', 'factor_value']].dropna(subset=['factor_value'])
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')
    output.to_csv(output_path, index=False)
    
    print(f"\n因子已保存: {output_path}")
    print(f"总行数: {len(output)}, 日期: {output['date'].min()} ~ {output['date'].max()}")
    print(f"日均股票: {output.groupby('date')['factor_value'].count().mean():.0f}")
    return output

if __name__ == '__main__':
    calc_vsplspike_v2()
