#!/usr/bin/env python3
"""Price-Trend Acceleration Factor (PTA)

Alpha#026 inspired: 20日滚动 close 与 volume 的偏相关 (控制总量后看动量一致性)。

更简单的实现：20日 close 与 volume 的相关系数。
- 正相关 = 量价齐升/齐跌 = 趋势明确 = 惯性延续
- 负相关 = 放量下跌或缩量上涨 = 分歧大 = 可能反转

But wait — existing pv_corr already in factors.json... Let me try a VARIANT:
Instead of direct correlation, use HIGHEST_CLOSE with VOLUME correlation.
High close + high volume = bullish momentum confirmed.
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

    # Rolling correlation between close and volume
    results = []
    for stock, g in df[['date', 'stock_code', 'volume', 'close']].groupby('stock_code'):
        g = g.sort_values('date').reset_index(drop=True)
        n = len(g)
        if n < window + 5: continue

        closes = g['close'].values
        volumes = g['volume'].values

        corrs = np.full(n, np.nan)
        for i in range(window - 1, n):
            c_seg = closes[i - window + 1:i + 1]
            v_seg = volumes[i - window + 1:i + 1]
            mask = (~np.isnan(c_seg)) & (~np.isnan(v_seg)) & (v_seg > 0)
            if mask.sum() < 10: continue
            c_g, v_g = c_seg[mask], v_seg[mask]
            if np.std(c_g) < 1e-8 or np.std(v_g) < 1e-8: continue
            try:
                # Pearson correlation
                std_c, std_v = np.std(c_g), np.std(v_g)
                if std_c < 1e-10 or std_v < 1e-10: continue
                cc = np.mean((c_g - c_g.mean()) * (v_g - v_g.mean())) / (std_c * std_v)
                corrs[i] = cc
            except: continue

        g_out = g.copy()
        g_out['factor_raw'] = corrs
        g_out = g_out.dropna(subset=['factor_raw']).dropna(subset=['close'])
        if len(g_out) > 0:
            results.append(g_out[['date', 'stock_code', 'factor_raw']])

    if not results:
        print("ERROR: no data"); return
    df_factor = pd.concat(results, ignore_index=True)

    # No log transform needed (already [-1,1])
    df_factor['factor_log'] = df_factor['factor_raw']

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
    outpath = DATA_DIR / "factor_ptr_v1.csv"
    result.to_csv(outpath, index=False)
    print(f"✅ Saved: {outpath}")
    print(f"   Shape: {result.shape}")


if __name__ == "__main__":
    main()
