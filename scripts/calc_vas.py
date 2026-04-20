#!/usr/bin/env python3
"""Volume-Adjusted Signal Factor (VAS)

Idea: Combine return direction (sign(ret)) with volume excess.
  vol_ratio = volume / MA20(volume)  -- excess volume indicator
  signal = sign(ret) * log(1 + vol_ratio)
  factor = mean(signal, 20d)  -- 20d rolling average

Interpretation:
  High (+) = 放量上涨 = buying conviction strong = momentum continuation expected
  High (-) = 放量下跌 = selling pressure strong = negative momentum
  Low   = 缩量 = noise, low information content

Neutralizaton: log_amount_20d OLS, MAD winsorize, z-score
"""

import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import lstsq

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    df = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    df = df.dropna(subset=['ret']).reset_index(drop=True)

    window = 20
    results = []

    for stock, g in df[['date', 'stock_code', 'volume', 'ret']].groupby('stock_code'):
        g = g.sort_values('date').reset_index(drop=True)
        n = len(g)
        if n < window + 10: continue

        vols = g['volume'].values
        rets = g['ret'].values

        # rolling MA20 of volume
        ma20_vol = np.full(n, np.nan)
        for i in range(window - 1, n):
            seg = vols[max(0, i - window + 1):i + 1]
            if len(seg) >= window // 2:
                ma20_vol[i] = np.mean(seg[~np.isnan(seg)])

        # daily signal = sign(ret) * log(1 + vol/MA20_vol)
        good = ~np.isnan(ma20_vol)
        daily_signal = np.full(n, np.nan)
        for i in range(n):
            if good[i] and ma20_vol[i] > 1 and not np.isnan(rets[i]):
                vr = vols[i] / ma20_vol[i]
                daily_signal[i] = np.sign(rets[i]) * np.log1p(vr)

        # rolling 20d mean of daily signal
        factor_raw = np.full(n, np.nan)
        for i in range(window - 1, n):
            seg = daily_signal[max(0, i - window + 1):i + 1]
            seg = seg[~np.isnan(seg)]
            if len(seg) >= window // 2:
                factor_raw[i] = np.mean(seg)

        g_out = g.copy()
        g_out['factor_raw'] = factor_raw
        g_out = g_out.dropna(subset=['factor_raw'])
        if len(g_out) > 0:
            results.append(g_out[['date', 'stock_code', 'factor_raw']])

    if not results:
        print("ERROR: no data"); return
    df_factor = pd.concat(results, ignore_index=True)

    # Log transform
    df_factor['factor_log'] = np.log(df_factor['factor_raw'].clip(lower=0.01) + 0.01)

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
    outpath = DATA_DIR / "factor_vas_v1.csv"
    result.to_csv(outpath, index=False)
    print(f"✅ Saved: {outpath}")
    print(f"   Shape: {result.shape}")


if __name__ == "__main__":
    main()
