"""
VSPLSPIKE v3: 极端分位比较版
================================
用20日成交量分布的分位而非固定阈值, 比较极端高量日(>70%分位)与
极端低量日(<30%分位)的远期5日收益差。

Benjamini & Hochberg (2023) 指出: 极端成交量日(事件日)后续收益的信号强度
最高; 而中等成交量日的噪音使信号模糊—这解释了 v1/v2 的 G3 collapse。

思路: 固定分位(3rd vs 7th decile) → G3应该回升, 单调性改善
"""
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def calc_vsplspike_v3(kline_path='data/csi1000_kline_raw.csv',
                       output_path='data/factor_vsplspike_v3.csv',
                       window=20,
                       forward_days=5,
                       high_quantile=0.7,
                       low_quantile=0.3,
                       min_periods_w=5):
    print(f"=== vsplspike_v3: quantile-based [{low_quantile} vs {high_quantile}], window={window}d ===")
    
    kline = pd.read_csv(kline_path)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    kline['ret'] = kline.groupby('stock_code')['close'].pct_change()
    kline['fwd_ret'] = kline.groupby('stock_code')['close'].shift(-forward_days) / kline['close'] - 1
    
    # Rolling quantile thresholds per stock
    print("Computing rolling quantile thresholds...")
    kline['vol_q_high'] = kline.groupby('stock_code')['volume'].transform(
        lambda x: x.rolling(window, min_periods=15).quantile(high_quantile)
    )
    kline['vol_q_low'] = kline.groupby('stock_code')['volume'].transform(
        lambda x: x.rolling(window, min_periods=15).quantile(low_quantile)
    )
    
    kline['is_high_vol_q'] = kline['volume'] >= kline['vol_q_high']
    kline['is_low_vol_q'] = kline['volume'] <= kline['vol_q_low']
    kline['is_mid_vol_q'] = ~(kline['is_high_vol_q'] | kline['is_low_vol_q'])
    
    n_high = kline['is_high_vol_q'].mean()
    n_low = kline['is_low_vol_q'].mean()
    n_mid = kline['is_mid_vol_q'].mean()
    print(f"高量日({high_quantile}分位): {n_high:.2%}, 低量日({low_quantile}分位): {n_low:.2%}, 中间: {n_mid:.2%}")
    
    # Forward returns typed by quantile bucket
    kline['fwd_high'] = np.where(kline['is_high_vol_q'], kline['fwd_ret'], np.nan)
    kline['fwd_low'] = np.where(kline['is_low_vol_q'], kline['fwd_ret'], np.nan)
    
    # Rolling mean over window
    print("Rolling means...")
    kline['cont_high'] = kline.groupby('stock_code')['fwd_high'].transform(
        lambda x: x.rolling(window, min_periods=min_periods_w).mean()
    )
    kline['cont_low'] = kline.groupby('stock_code')['fwd_low'].transform(
        lambda x: x.rolling(window, min_periods=min_periods_w).mean()
    )
    
    # factor = high - low; 正=极端放量日后续更强→信息驱动
    kline['factor_raw'] = kline['cont_high'] - kline['cont_low']
    
    print(f"factor_raw: mean={kline['factor_raw'].mean():.6f}, std={kline['factor_raw'].std():.6f}, nan={kline['factor_raw'].isna().mean():.2%}")
    
    # amount OLS neutralization
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
    
    # MAD + z-score
    print("MAD + z-score...")
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
    n_dates = output['date'].nunique()
    print(f"\n因子: {output_path}, {len(output)}行, {n_dates}日, {output['date'].min()}~{output['date'].max()}")
    return output


if __name__ == '__main__':
    calc_vsplspike_v3()
