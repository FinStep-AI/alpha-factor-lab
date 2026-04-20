#!/usr/bin/env python3
"""Realized Skewness Factor

20日滚动已实现偏度：负偏 = 波动集中在下行 = 下行风险高。
正向使用：低偏度（波动对称或正偏）= 高预期收益。

理论：低下行风险的股票更受机构欢迎（Ang, Chen & Xing 2006; Harvey & Siddique 2000）。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import lstsq

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def roll_skew(x, window):
    """Rolling skewness."""
    vals = np.full(len(x), np.nan)
    for i in range(window - 1, len(x)):
        seg = x[i - window + 1: i + 1]
        if len(seg) < window // 2:
            continue
        s = np.std(seg)
        if s < 1e-8:
            vals[i] = 0.0
        else:
            vals[i] = np.mean(((seg - np.mean(seg)) / s) ** 3)
    return vals


def main():
    df = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    df = df.dropna(subset=['ret']).reset_index(drop=True)

    window = 20
    all_rows = []

    for stock, g in df[['date', 'stock_code', 'ret']].groupby('stock_code'):
        g = g.sort_values('date').reset_index(drop=True)
        rets = g['ret'].values
        sk = roll_skew(rets, window)
        g_out = g.copy()
        g_out['factor_raw'] = sk
        g_out = g_out.dropna(subset=['factor_raw'])
        if len(g_out) > 0:
            all_rows.append(g_out[['date', 'stock_code', 'factor_raw']])

    df_factor = pd.concat(all_rows, ignore_index=True)

    # Negate: low skew (negative raw) -> high factor value -> positive IC expected
    # Actually let's test: raw = negative_skew (stock has negative skewness = more downside)
    # We want positive IC = low skew -> better returns
    # So factor should be -skewness (negate: negative skew becomes positive raw)
    df_factor['factor_raw'] = -df_factor['factor_raw']

    # Log transform (skewness can be negative, so use tanh to bound)
    df_factor['factor_log'] = np.tanh(df_factor['factor_raw'])

    # Market cap neutralization
    df_amt = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv")
    df_amt['date'] = pd.to_datetime(df_amt['date'])
    df_amt = df_amt.sort_values(['stock_code', 'date'])
    df_amt['log_amount_20d'] = df_amt.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1))
    df_factor = df_factor.merge(df_amt[['date', 'stock_code', 'log_amount_20d']],
                                 on=['date', 'stock_code'], how='left')
    df_factor = df_factor.dropna(subset=['factor_log', 'log_amount_20d'])

    output_rows = []
    for date, d in df_factor.groupby('date'):
        d = d.copy()
        if len(d) < 100: continue
        X = np.column_stack([np.ones(len(d)), d['log_amount_20d'].values])
        y = d['factor_log'].values
        try:
            beta, _, _, _ = lstsq(X, y, rcond=None)
            d['factor_neut'] = y - X @ beta
        except: continue
        med = d['factor_neut'].median()
        mad = (d['factor_neut'] - med).abs().median() * 1.4826
        if mad < 1e-6: continue
        d['factor_neut'] = d['factor_neut'].clip(med - 5.2*mad, med + 5.2*mad)
        std = d['factor_neut'].std()
        if std < 1e-6: continue
        d['factor_value'] = (d['factor_neut'] - d['factor_neut'].mean()) / std
        output_rows.append(d[['date', 'stock_code', 'factor_value']])

    if not output_rows:
        print("ERROR: no output"); return
    result = pd.concat(output_rows, ignore_index=True)
    outpath = DATA_DIR / "factor_skewness_v1.csv"
    result.to_csv(outpath, index=False)
    print(f"✅ Saved: {outpath}")
    print(f"   Shape: {result.shape}")


if __name__ == "__main__":
    main()
