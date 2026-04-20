#!/usr/bin/env python3
"""Order Flow Intensity (OFI) Factor — advanced CBI variant

Alternative to CBI: Instead of weighted candle body ratio,
compute pure ORDER FLOW direction via volume-weighted return sign.

SIGNAL = sign(ret) * log(1 + vol_shares)
where vol_shares = volume / MA60(volume)  (longer-term baseline)
Rolling 20d mean.

This directly measures: on informative days, how strongly does
direction predict returns? (Higher volume should amplify valid signals)
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

    # Daily returns
    df['ret'] = df.groupby('stock_code')['close'].pct_change()
    df = df.dropna(subset=['ret']).reset_index(drop=True)

    # Volume ratio vs 60-day MA (more stable baseline than 20d)
    df['ma60_vol'] = df.groupby('stock_code')['volume'].transform(
        lambda x: x.rolling(60, min_periods=30).mean() + 1e-8)
    df['vol_ratio'] = np.log1p(df['volume'] / df['ma60_vol'])

    # Signal: sign(ret) * vol_ratio → directional conviction on volume-heavy days
    # Only signal when |ret| is meaningful (filter micro-moves)
    df['signal'] = np.nan
    mask = (df['ret'].notna()) & (df['vol_ratio'].notna())
    df.loc[mask, 'signal'] = np.sign(df.loc[mask, 'ret']) * df.loc[mask, 'vol_ratio']

    # Rolling 20d mean
    df['factor_raw'] = df.groupby('stock_code')['signal'].transform(
        lambda x: x.rolling(20, min_periods=15).mean())

    df = df.dropna(subset=['factor_raw'])

    # Log transform
    df['factor_log'] = np.log(df['factor_raw'].clip(lower=0.01) + 0.01)

    # Market cap neutralization
    df_amt = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv")
    df_amt['date'] = pd.to_datetime(df_amt['date'])
    df_amt = df_amt.sort_values(['stock_code', 'date'])
    df_amt['log_amount_20d'] = df_amt.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1))
    df_factor = df[['date', 'stock_code', 'factor_log']].merge(
        df_amt[['date', 'stock_code', 'log_amount_20d']],
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
    outpath = DATA_DIR / "factor_ofi_v1.csv"
    result.to_csv(outpath, index=False)
    print(f"✅ Saved: {outpath}")
    print(f"   Shape: {result.shape}")


if __name__ == "__main__":
    main()
