"""
信息量归集因子 (Information Aggregation Factor)
=================================================
Factor: vol_spike_continuation_v1

思路: 区分成交量异动是信息扩散(后续趋势延续)还是噪音交易(后续反转)。
     高量日后续5日累计收益 - 低量日后续5日累计收益。
     正=信息确认(放量后继续涨/跌)，负=噪音回吐(放量后反转)。

计算步骤:
1. 20日均量, 计算volume_z = (volume - MA20) / std(volume, 20d)
2. 高量日: volume_z > 0.5; 低量日: volume_z < 0
3. 每个T日定义:往后5日累计收益
4. 滚动20日计算: avg(cont_return_high_vol) - avg(cont_return_low_vol)
5. 创业板中性化 + MAD Winsorize + z-score

Hypothesis: 高成交量伴随趋势延续的状况反映知情交易者破冰,
            对应黄俊和韩立岩(2013)的市场情绪假说;
A股小盘股放量上涨后往往继续涨(散户共振),但收益模式的持续性与信息质量相关。

Barra Style: MICRO (微观结构)
"""

import numpy as np
import pandas as pd
import sys
import os

def calc_vol_spike_continuation(kline_path='data/csi1000_kline_raw.csv',
                                 output_path='data/factor_vol_spike_continuation_v1.csv',
                                 ret_path='data/csi1000_returns.csv',
                                 window=20,
                                 forward_days=5,
                                 spike_threshold=0.5,
                                 zero_vol_threshold=0.0):
    """
    计算信息量归集因子
    
    Parameters:
    -----------
    window: 滚动计算窗口 (默认20日)
    forward_days: 前瞻天数 (默认5日)
    spike_threshold: 高量日volume_z阈值 (默认0.5 = z-score高于均值0.5个标准差)
    zero_vol_threshold: 低量日边界 (默认0 = z-score < 0)
    """
    print(f"=== 信息量归集因子计算 ===")
    print(f"窗口={window}日, 前瞻={forward_days}日, spike_z>={spike_threshold}")
    
    # Load data
    print("加载数据...")
    kline = pd.read_csv(kline_path)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
    
    # Calculate daily return
    kline['ret'] = kline.groupby('stock_code')['close'].pct_change()
    
    # Forward return (用于因子前瞻)
    kline['fwd_ret'] = kline.groupby('stock_code')['close'].shift(-forward_days) / kline['close'] - 1
    
    # Rolling mean and std of volume (20d)
    print("计算成交量z-score...")
    kline['vol_ma'] = kline.groupby('stock_code')['volume'].transform(
        lambda x: x.rolling(window, min_periods=15).mean()
    )
    kline['vol_std'] = kline.groupby('stock_code')['volume'].transform(
        lambda x: x.rolling(window, min_periods=15).std()
    )
    kline['vol_z'] = (kline['volume'] - kline['vol_ma']) / (kline['vol_std'] + 1e-9)
    
    # Classify volume days
    kline['is_high_vol'] = kline['vol_z'] > spike_threshold
    kline['is_low_vol'] = kline['vol_z'] < zero_vol_threshold
    kline['is_normal_vol'] = ~(kline['is_high_vol'] | kline['is_low_vol'])
    
    print(f"高量日比例: {kline['is_high_vol'].mean():.2%}, 低量日比例: {kline['is_low_vol'].mean():.2%}")
    
    # Calculate forward return for each volume class
    print("计算各类成交量日的前瞻收益...")
    kline['fwd_ret_high'] = np.where(kline['is_high_vol'], kline['fwd_ret'], np.nan)
    kline['fwd_ret_low'] = np.where(kline['is_low_vol'], kline['fwd_ret'], np.nan)
    
    # Rolling 20d mean of continuation returns
    print("滚动计算因子值...")
    kline['cont_high'] = kline.groupby('stock_code')['fwd_ret_high'].transform(
        lambda x: x.rolling(window, min_periods=10).mean()
    )
    kline['cont_low'] = kline.groupby('stock_code')['fwd_ret_low'].transform(
        lambda x: x.rolling(window, min_periods=10).mean()
    )
    
    # Raw factor = high_vol_continuation - low_vol_continuation
    # 正值 = 高量日后续更好 = 信息确认
    kline['factor_raw'] = kline['cont_high'] - kline['cont_low']
    
    # Neutralization: Amount (成交额) OLS
    print("成交额中性化...")
    from scipy import stats
    
    results = []
    for date, group in kline.groupby('date'):
        if len(group) < 30:
            results.append(group)
            continue
        
        g = group[['stock_code', 'factor_raw', 'amount']].dropna()
        if len(g) < 10:
            results.append(group.assign(factor_neutral=np.nan))
            continue
        
        # OLS on log amounts
        y = g['factor_raw'].values
        x = np.log(g['amount'].values + 1)
        
        # Robust regression
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 10:
            results.append(group.assign(factor_neutral=np.nan))
            continue
        
        slope, intercept, _, _, _ = stats.linregress(x[mask], y[mask])
        residual = np.full(len(g), np.nan)
        residual[mask] = y[mask] - (intercept + slope * x[mask])
        g = g.copy()
        g['factor_neutral'] = residual
        group = group.merge(g[['stock_code', 'factor_neutral']], on='stock_code', how='left')
        results.append(group)
    
    kline = pd.concat(results, ignore_index=True)
    
    # MAD Winsorize + z-score
    print("MAD Winsorize + z-score...")
    def mad_zscore(series):
        median = series.median()
        mad = np.abs(series - median).median()
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
    
    print(f"因子已保存: {output_path}")
    print(f"总行数: {len(output)}, 日期范围: {output['date'].min()} ~ {output['date'].max()}")
    print(f"因子均值: {output['factor_value'].mean():.4f}, 标准差: {output['factor_value'].std():.4f}")
    
    return output


if __name__ == '__main__':
    calc_vol_spike_continuation()
