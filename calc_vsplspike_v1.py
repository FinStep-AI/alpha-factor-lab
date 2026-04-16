"""
Volume Spike-Weighted Momentum (vsplspike_v1)
===========================================
计算高低成交量日远期收益差异因子

核心思路：区分"信息驱动的放量"与"情绪驱动的放量"。
- 用20日volume z-score作连续权重，而非二元分类
- 高量日收益×高权重，低量日收益×低权重
- 若放量日后期走势更好 → 高因子值 → 信息认同

简化版(避免min_periods不足): 
直接用 forward_return 与 volume_z 做滚动20日加权平均,
比较高权重(>0)和低权重(>0)的远期回报差异。

Barra风格: MICRO
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def calc_vsplspike_v1(kline_path='data/csi1000_kline_raw.csv',
                       output_path='data/factor_vsplspike_v1.csv',
                       window=20,
                       spike_threshold=0.4,
                       forward_days=5,
                       min_periods_cont=8):
    """
    计算Volume Spike-Weighted Return Contrast因子
    
    公式:
    factor = mean(fwd_ret * vol_z_clipped, window), where vol_z_clipped > 0
              - mean(fwd_ret * |vol_z|_neg_clipped, window), where vol_z < 0
    
    即: 高量日远期收益'(正值) - 超低量日远期收益'(正值=远离均值)
    去掉绝对量只看相对方向。
    """
    print(f"=== vsplspike_v1: Volume Spike-Weighted Momentum ===")
    print(f"窗口={window}d, 前瞻={forward_days}d, z-threshold={spike_threshold}")
    
    kline = pd.read_csv(kline_path)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # daily return
    kline['ret'] = kline.groupby('stock_code')['close'].pct_change()
    
    # forward return
    kline['fwd_ret'] = kline.groupby('stock_code')['close'].shift(-forward_days) / kline['close'] - 1
    
    # rolling volume stats
    print("Computing volume z-score...")
    kline['vol_z'] = kline.groupby('stock_code')['volume'].transform(
        lambda x: (x - x.rolling(window, min_periods=15).mean()) /
                  (x.rolling(window, min_periods=15).std() + 1e-9)
    )
    
    # 权重: 只保留正值信号(高量vs低量), 用 |vol_z| 脉冲幅度做权重
    # 分开计算: 高量日(vol_z>spike)和低量日(vol_z<-spike/2)
    kline['w_high'] = np.where(kline['vol_z'] > spike_threshold, 
                                 kline['vol_z'] - spike_threshold, 0.0)
    kline['w_low'] = np.where(kline['vol_z'] < -spike_threshold / 2,
                                np.abs(kline['vol_z'] + spike_threshold / 2), 0.0)
    
    # weighted fwd_ret
    kline['w_ret_high'] = kline['w_high'] * kline['fwd_ret']
    kline['w_ret_low'] = kline['w_low'] * kline['fwd_ret']
    
    # rolling weighted mean
    print("Rolling weighted means...")
    # Use raw weighted sums / total weight
    kline['num_high'] = kline.groupby('stock_code')['w_high'].transform(
        lambda x: x.rolling(window, min_periods=5).sum()
    )
    kline['num_low'] = kline.groupby('stock_code')['w_low'].transform(
        lambda x: x.rolling(window, min_periods=5).sum()
    )
    kline['sum_wret_high'] = kline.groupby('stock_code')['w_ret_high'].transform(
        lambda x: x.rolling(window, min_periods=5).sum()
    )
    kline['sum_wret_low'] = kline.groupby('stock_code')['w_ret_low'].transform(
        lambda x: x.rolling(window, min_periods=5).sum()
    )
    
    # weighted average: sum(w*ret) / sum(w)
    kline['weighted_cont_high'] = kline['sum_wret_high'] / (kline['num_high'] + 1e-9)
    kline['weighted_cont_low'] = kline['sum_wret_low'] / (kline['num_low'] + 1e-9)
    
    # Final factor: weighted high - weighted low
    # 高量日远期加权收益 - 低量日远期加权收益
    # 正 = 高量/信息日行情确认; 负 = 放量后反转(噪音)
    kline['factor_raw'] = kline['weighted_cont_high'] - kline['weighted_cont_low']
    
    # Take abs(factor) for scale-conservation
    # But keep sign!
    
    print(f"factor_raw stats: mean={kline['factor_raw'].mean():.6f}, "
          f"std={kline['factor_raw'].std():.6f}, "
          f"nan_rate={kline['factor_raw'].isna().mean():.2%}")
    
    # Neutralization: amount OLS
    print("Amount OLS neutralization...")
    results = []
    for date, group in kline.groupby('date'):
        if len(group) < 30:
            group = group.assign(factor_neutral=np.nan)
            results.append(group)
            continue
        
        g = group[['stock_code', 'factor_raw', 'amount']].dropna()
        g = g[np.isfinite(g['factor_raw'])]
        if len(g) < 10:
            group = group.assign(factor_neutral=np.nan)
            results.append(group)
            continue
        
        y = g['factor_raw'].values
        x = np.log(g['amount'].values + 1)
        
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 10:
            group = group.assign(factor_neutral=np.nan)
            results.append(group)
            continue
        
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
        fn_col = [c for c in merged.columns if c.endswith('_g')]
        if fn_col:
            merged['factor_neutral'] = merged[fn_col[0]]
            merged = merged.drop(columns=[fn_col[0], 'factor_neutral_g'], errors='ignore')
        if 'factor_neutral' not in merged.columns:
            merged['factor_neutral'] = np.nan
        results.append(merged)
    
    kline = pd.concat(results, ignore_index=True)
    
    # MAD Winsorize + z-score
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
    
    # Output
    output = kline[['date', 'stock_code', 'factor_value']].dropna(subset=['factor_value'])
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')
    output.to_csv(output_path, index=False)
    
    print(f"\n因子已保存: {output_path}")
    print(f"总行数: {len(output)}, 日期范围: {output['date'].min()} ~ {output['date'].max()}")
    print(f"因子均值: {output['factor_value'].mean():.4f}, 标准差: {output['factor_value'].std():.4f}")
    
    # diagnostics
    per_date = output.groupby('date')['factor_value'].count()
    print(f"日均股票数: {per_date.mean():.0f}")
    print(f"日期数: {output['date'].nunique()}")
    
    return output


if __name__ == '__main__':
    calc_vsplspike_v1()
