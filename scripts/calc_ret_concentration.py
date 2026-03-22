#!/usr/bin/env python3
"""
Return Concentration (收益集中度) 因子

构造多个收益集中度/分散度因子：

1. ret_hhi: 日收益绝对值的HHI指数（收益集中度）
   HHI = sum((|r_i|/sum(|r|))^2), 20日窗口
   高HHI = 跳跃式变动 → 信息冲击驱动
   低HHI = 渐进式变动 → 平稳信息扩散

2. ret_streak: 最长连涨/连跌天数比
   衡量收益率的序列相关性/趋势持续性

3. up_vol_ratio: 上涨日波动占比
   = std(positive_returns) / std(all_returns)
   高比值 = 上行波动大（看涨不确定性高）
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compute_ret_concentration(kline_path: str, output_prefix: str, window: int = 20):
    print(f"Loading kline data from {kline_path}...")
    df = pd.read_csv(kline_path, parse_dates=['date'])
    df = df[df['date'] <= '2026-03-07'].copy()
    
    # 计算日收益率
    df['ret'] = df['pct_change'] / 100.0  # pct_change是百分比
    
    factors = {}
    
    for code, grp in df.groupby('stock_code'):
        grp = grp.sort_values('date').copy()
        n = len(grp)
        ret = grp['ret'].values
        amt = grp['amount'].values
        dates = grp['date'].values
        
        # 用于中性化
        log_amt = np.log(pd.Series(amt).rolling(20, min_periods=20).mean() + 1)
        
        hhi_vals = np.full(n, np.nan)
        up_vol_vals = np.full(n, np.nan)
        
        for i in range(window - 1, n):
            r_win = ret[i - window + 1:i + 1]
            r_valid = r_win[np.isfinite(r_win)]
            if len(r_valid) < window * 0.8:
                continue
            
            # Factor 1: Return HHI
            abs_r = np.abs(r_valid)
            total = abs_r.sum()
            if total > 1e-10:
                shares = abs_r / total
                hhi_vals[i] = np.sum(shares ** 2)
            
            # Factor 3: Up-vol ratio
            up_r = r_valid[r_valid > 0]
            if len(up_r) >= 3:
                std_all = np.std(r_valid)
                std_up = np.std(up_r)
                if std_all > 1e-10:
                    up_vol_vals[i] = std_up / std_all
        
        for name, vals in [('ret_hhi_v1', hhi_vals), ('up_vol_ratio_v1', up_vol_vals)]:
            if name not in factors:
                factors[name] = []
            tmp = pd.DataFrame({
                'date': dates,
                'stock_code': code,
                'factor_raw': vals,
                'log_amount_20d': log_amt.values
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

        # 保存正向和反向
        out_path = f"{output_prefix}_{name}_long.csv"
        result.to_csv(out_path, index=False)
        print(f"  Output: {result.shape}, saved to {out_path}")
        
        neg = result.copy()
        neg['factor_value'] = -neg['factor_value']
        out_neg = f"{output_prefix}_{name}_neg_long.csv"
        neg.to_csv(out_neg, index=False)
        print(f"  Neg output saved to {out_neg}")


if __name__ == '__main__':
    base = Path(__file__).resolve().parent.parent
    compute_ret_concentration(
        kline_path=str(base / 'data' / 'csi1000_kline_raw.csv'),
        output_prefix=str(base / 'data' / 'factor'),
        window=20
    )
