#!/usr/bin/env python3
"""Volume-Price Efficiency Factor — sign-flipped v2

放量日平均收益 - 缩量日平均收益（波动率标准化），方向翻转后正向使用。
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
        if n < window + 5: continue
        volumes = g['volume'].values; rets = g['ret'].values
        vals = []
        for i in range(window - 1, n):
            v_win = volumes[i - window + 1 : i + 1]
            r_win = rets[i - window + 1 : i + 1]
            if np.sum(~np.isnan(v_win)) < window * 0.9: vals.append(np.nan); continue
            vv = v_win[~np.isnan(v_win)]; rr = r_win[~np.isnan(v_win)]
            if len(vv) < window // 2: vals.append(np.nan); continue
            try: q1, q3 = np.percentile(vv, [33.33, 66.66])
            except: vals.append(np.nan); continue
            hm = vv >= q3; lm = vv <= q1
            if hm.sum() < 2 or lm.sum() < 2: vals.append(np.nan); continue
            r_high = np.mean(rr[hm]); r_low = np.mean(rr[lm])
            sigma = np.std(rr)
            if sigma < 1e-8: vals.append(np.nan); continue
            # FLIP SIGN: original was (high-low)/sigma, flip to (low-high)/sigma
            # so that positive IC = high factor = higher returns
            vals.append((r_low - r_high) / sigma)
        if vals:
            go = g.iloc[window - 1:].copy()
            go['factor_raw'] = vals[:len(go)]
            go = go.dropna(subset=['factor_raw'])
            if len(go) > 0:
                results.append(go[['date', 'stock_code', 'factor_raw']])

    if not results:
        print("ERROR: no data"); return
    df_factor = pd.concat(results, ignore_index=True)

    # Log transform
    df_factor['factor_log'] = np.log(df_factor['factor_raw'].clip(lower=0.001) + 0.02)

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
    outpath = DATA_DIR / "factor_pv_efficiency_v1.csv"
    result.to_csv(outpath, index=False)
    print(f"✅ Saved: {outpath}")
    print(f"   Shape: {result.shape}")


if __name__ == "__main__":
    main()
