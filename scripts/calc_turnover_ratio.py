#!/usr/bin/env python3
"""
Turnover Ratio (换手率短长比) 因子

原理：
TR = log(MA5(turnover) / MA60(turnover))
衡量近期换手率相对长期的相对水平。

高TR → 近期换手率快速上升 → 新资金涌入/关注度骤增
低TR → 换手率萎缩 → 资金退潮

灵感来源:
- Gervais, Kaniel & Mingelgrin (2001) "The High-Volume Return Premium" JF
- Lee & Swaminathan (2000) "Price Momentum and Trading Volume" JF

正向: 高TR→高收益 (关注度提升动量)
反向: 低TR→高收益 (换手率衰减反转)

也测试换手率加速度:
  turnover_accel = MA5(turnover) - 2*MA20(turnover) + MA60(turnover)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compute_turnover_ratio(kline_path: str, output_prefix: str):
    """计算多个换手率动态因子"""
    print(f"Loading kline data from {kline_path}...")
    df = pd.read_csv(kline_path, parse_dates=['date'])
    df = df[df['date'] <= '2026-03-07'].copy()
    
    factors = {}
    
    for code, grp in df.groupby('stock_code'):
        grp = grp.sort_values('date').copy()
        to = grp['turnover'].values
        amt = grp['amount'].values
        
        # 滚动均值
        ma5 = pd.Series(to).rolling(5, min_periods=5).mean()
        ma20 = pd.Series(to).rolling(20, min_periods=20).mean()
        ma60 = pd.Series(to).rolling(60, min_periods=60).mean()
        
        # 因子1: TR = log(MA5 / MA60)
        tr = np.log(ma5 / ma60)
        
        # 因子2: 换手率z-score = (MA5 - MA60) / std(turnover, 60)
        std60 = pd.Series(to).rolling(60, min_periods=60).std()
        tr_z = (ma5 - ma60) / (std60 + 1e-10)
        
        # 用于中性化的log_amount_20d
        log_amt = np.log(pd.Series(amt).rolling(20, min_periods=20).mean() + 1)
        
        dates = grp['date'].values
        codes = np.full(len(grp), code)
        
        for name, vals in [('tr_v1', tr), ('tr_z_v1', tr_z)]:
            if name not in factors:
                factors[name] = []
            tmp = pd.DataFrame({
                'date': dates, 
                'stock_code': codes,
                'factor_raw': vals,
                'log_amount_20d': log_amt
            })
            factors[name].append(tmp)
    
    for name, dfs in factors.items():
        print(f"\nProcessing factor: {name}")
        fdf = pd.concat(dfs, ignore_index=True)
        fdf = fdf.replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  Raw shape: {fdf.shape}")
        
        # OLS中性化
        neutralized = []
        for dt, cross in fdf.groupby('date'):
            if len(cross) < 50:
                continue
            x = cross['log_amount_20d'].values
            y = cross['factor_raw'].values
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 50:
                continue
            x_v, y_v = x[valid], y[valid]
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
        
        fdf = pd.concat(neutralized, ignore_index=True).dropna()
        
        # MAD + z-score
        final = []
        for dt, cross in fdf.groupby('date'):
            vals = cross['factor_raw'].values
            med = np.median(vals)
            mad = np.median(np.abs(vals - med))
            if mad < 1e-10:
                continue
            upper = med + 5 * 1.4826 * mad
            lower = med - 5 * 1.4826 * mad
            clipped = np.clip(vals, lower, upper)
            m = np.mean(clipped)
            s = np.std(clipped)
            if s < 1e-10:
                continue
            z = (clipped - m) / s
            tmp = cross[['date', 'stock_code']].copy()
            tmp['factor_value'] = z
            final.append(tmp)
        
        result = pd.concat(final, ignore_index=True)
        out_path = f"{output_prefix}_{name}_long.csv"
        result.to_csv(out_path, index=False)
        print(f"  Output: {result.shape}, saved to {out_path}")
        print(f"  Date range: {result['date'].min()} ~ {result['date'].max()}")


if __name__ == '__main__':
    base = Path(__file__).resolve().parent.parent
    compute_turnover_ratio(
        kline_path=str(base / 'data' / 'csi1000_kline_raw.csv'),
        output_prefix=str(base / 'data' / 'factor')
    )
