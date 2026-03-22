#!/usr/bin/env python3
"""
Volume-Weighted Close Location Value (VW-CLV) 因子

原理：
CLV = (2*close - high - low) / (high - low)  ∈ [-1, 1]
衡量收盘价在日内价格区间中的位置。

VW-CLV = 20日内，以成交量为权重的CLV加权平均。
高VW-CLV → 放量日倾向收在高位 → 买方主导/主力吸筹
低VW-CLV → 放量日倾向收在低位 → 卖方主导/主力出货

成交额OLS中性化 + MAD winsorize + z-score

灵感来源:
- Chaikin (Accumulation/Distribution Line) 使用 CLV × volume 但做的是时序累积
- 本因子将其截面化：做横截面排序(cross-sectional factor)
- 与 pv_corr 不同: pv_corr 看量变化与价变化的相关性，VW-CLV看放量时收盘位置偏好

参考文献:
- Chaikin, M. (1986) "Volume and Price Action"
- Datar et al. (1998) "Liquidity and stock returns" JFE
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compute_vw_clv(kline_path: str, output_path: str, window: int = 20):
    """计算VW-CLV因子"""
    print(f"Loading kline data from {kline_path}...")
    df = pd.read_csv(kline_path, parse_dates=['date'])
    
    # 排除异常数据(2026-03-09后数据可能有异常)
    df = df[df['date'] <= '2026-03-07'].copy()
    
    # 计算CLV: (2*close - high - low) / (high - low)
    price_range = df['high'] - df['low']
    # 避免除零（涨跌停板high==low）
    price_range = price_range.replace(0, np.nan)
    df['clv'] = (2 * df['close'] - df['high'] - df['low']) / price_range
    
    # 排除CLV为NaN的行（涨跌停日）
    print(f"CLV NaN count (flat days): {df['clv'].isna().sum()}")
    
    # 按股票分组计算滚动VW-CLV
    print(f"Computing {window}d volume-weighted CLV...")
    
    results = []
    for code, grp in df.groupby('stock_code'):
        grp = grp.sort_values('date').copy()
        
        # 成交量加权CLV: sum(CLV * volume) / sum(volume)
        clv_x_vol = grp['clv'] * grp['volume']
        
        # 滚动窗口
        sum_clv_vol = clv_x_vol.rolling(window, min_periods=window).sum()
        sum_vol = grp['volume'].rolling(window, min_periods=window).sum()
        
        grp['vw_clv_raw'] = sum_clv_vol / sum_vol
        
        results.append(grp[['date', 'stock_code', 'vw_clv_raw']].dropna(subset=['vw_clv_raw']))
    
    factor_df = pd.concat(results, ignore_index=True)
    print(f"Raw factor shape: {factor_df.shape}")
    
    # === 截面处理：每日截面内标准化 ===
    # 1. 需要成交额数据做中性化
    # 计算20日均成交额
    amt_results = []
    for code, grp in df.groupby('stock_code'):
        grp = grp.sort_values('date').copy()
        grp['log_amount_20d'] = np.log(grp['amount'].rolling(20, min_periods=20).mean() + 1)
        amt_results.append(grp[['date', 'stock_code', 'log_amount_20d']].dropna())
    
    amt_df = pd.concat(amt_results, ignore_index=True)
    factor_df = factor_df.merge(amt_df, on=['date', 'stock_code'], how='inner')
    
    # 2. 截面OLS中性化（去成交额影响）
    print("OLS neutralization by log_amount_20d...")
    neutralized = []
    for dt, cross in factor_df.groupby('date'):
        if len(cross) < 50:
            continue
        x = cross['log_amount_20d'].values
        y = cross['vw_clv_raw'].values
        
        # 过滤NaN/Inf
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 50:
            continue
        
        x_v, y_v = x[valid], y[valid]
        
        # OLS: y = a + b*x + residual
        X = np.column_stack([np.ones(len(x_v)), x_v])
        try:
            beta = np.linalg.lstsq(X, y_v, rcond=None)[0]
        except:
            continue
        
        residual = np.full(len(x), np.nan)
        residual[valid] = y_v - X @ beta
        
        tmp = cross[['date', 'stock_code']].copy()
        tmp['factor_raw'] = residual
        neutralized.append(tmp)
    
    factor_df = pd.concat(neutralized, ignore_index=True).dropna(subset=['factor_raw'])
    
    # 3. MAD winsorize
    print("MAD winsorize + z-score...")
    final = []
    for dt, cross in factor_df.groupby('date'):
        vals = cross['factor_raw'].values
        median = np.median(vals)
        mad = np.median(np.abs(vals - median))
        if mad < 1e-10:
            continue
        
        # MAD winsorize at ±5 MAD
        upper = median + 5 * 1.4826 * mad
        lower = median - 5 * 1.4826 * mad
        clipped = np.clip(vals, lower, upper)
        
        # z-score
        mean = np.mean(clipped)
        std = np.std(clipped)
        if std < 1e-10:
            continue
        
        z = (clipped - mean) / std
        
        tmp = cross[['date', 'stock_code']].copy()
        tmp['factor_value'] = z
        final.append(tmp)
    
    result = pd.concat(final, ignore_index=True)
    
    # 输出
    out = result.pivot(index='date', columns='stock_code', values='factor_value')
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    
    print(f"Output shape: {out.shape}")
    print(f"Date range: {out.index.min()} ~ {out.index.max()}")
    print(f"Non-NaN ratio: {out.notna().sum().sum() / out.size:.2%}")
    
    out.to_csv(output_path)
    print(f"Saved to {output_path}")
    
    return out


if __name__ == '__main__':
    base = Path(__file__).resolve().parent.parent
    compute_vw_clv(
        kline_path=str(base / 'data' / 'csi1000_kline_raw.csv'),
        output_path=str(base / 'data' / 'factor_vw_clv_v1.csv'),
        window=20
    )
