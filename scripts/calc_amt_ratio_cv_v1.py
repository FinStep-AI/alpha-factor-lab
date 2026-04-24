#!/usr/bin/env python3
"""
因子: 成交额相对波动率CV (amount_ratio_cv_v1)

构造:
  amt_ratio = amount / MA20(amount)          # 日成交额相对基准的比值
  factor_raw = std(amt_ratio, 20d) / mean(amt_ratio, 20d)   # CV

物理意义:
  CV大 = 相对成交额百分比波动大 = 资金流入/流出不稳定
  CV小 = 相对成交额百分比稳定 = 资金流向规律性强

中性化: log_amount OLS + MAD + z-score
"""

import numpy as np
import pandas as pd
import sys, os

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(base)
    input_path = os.path.join(proj, 'data', 'csi1000_kline_raw.csv')
    output_path = os.path.join(proj, 'data', 'factor_amount_ratio_cv_v1.csv')

    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)

    # 20日平均成交额
    df['amt_ma20'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    # 相对成交额比值
    df['amt_ratio'] = df['amount'] / (df['amt_ma20'] + 1)

    # 20日滚动CV = std/mean
    df['factor_raw'] = df.groupby('stock_code')['amt_ratio'].transform(
        lambda x: x.rolling(20, min_periods=10).std() / (x.rolling(20, min_periods=10).mean() + 1e-8)
    )

    # log_amount_20d for neutralization
    df['log_amount_20d'] = np.log(df['amt_ma20'].clip(lower=1))

    # ---- 中性化 : log_amount OLS + MAD + z-score ----
    from numpy.linalg import lstsq

    def neutralize_group(g):
        y = g['factor_raw'].values.astype(float)
        x = g['log_amount_20d'].values.astype(float)
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 5:
            g['factor_neutral'] = np.nan
            return g
        A = np.column_stack([np.ones(mask.sum()), x[mask]])
        b, _, _, _ = lstsq(A, y[mask], rcond=None)
        resid = np.full(len(y), np.nan)
        resid[mask] = y[mask] - A @ b
        g['factor_neutral'] = resid
        return g

    df = df.groupby('date', group_keys=False).apply(neutralize_group)

    # MAD winsorize
    def mad_winsorize(series, n_mad=3.0):
        med = series.median()
        mad = (series - med).abs().median()
        if mad == 0 or np.isnan(mad):
            return series
        scaled = 1.4826 * mad
        return series.clip(med - n_mad * scaled, med + n_mad * scaled)

    df['factor_neutral'] = df.groupby('date')['factor_neutral'].transform(
        lambda x: mad_winsorize(x, 3.0)
    )

    # z-score
    def zscore_group(g):
        v = g['factor_neutral']
        valid = v.notna()
        if valid.sum() < 3:
            g['factor'] = np.nan
            return g
        m, s = v[valid].mean(), v[valid].std()
        if s == 0:
            g['factor'] = 0.0
            return g
        g['factor'] = (v - m) / s
        return g

    df = df.groupby('date', group_keys=False).apply(zscore_group)

    out = df[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    out.to_csv(output_path, index=False)

    fv = out['factor']
    print(f"✅ Saved to {output_path}")
    print(f"   rows={len(out)}, stocks={out['stock_code'].nunique()}, dates={out['date'].nunique()}")
    print(f"   mean={fv.mean():.4f}, std={fv.std():.4f}, range=[{fv.min():.4f},{fv.max():.4f}]")

if __name__ == '__main__':
    main()
