#!/usr/bin/env python3
"""Candle Body Intensity (CBI) Factor

Core idea: On high-volume days, body size = directional conviction.
   body_ratio = |close - open| / (high - low + epsilon)
   vol_weight = log(1 + volume / MA20(volume))
   signal = vol_weight * body_ratio (already directional, abs removes direction info)
   rolling 20d average:  cumulative directional conviction
Alternatively use sign(close-open) * body_ratio to preserve direction.

Direction test: use raw weighted body (with sign). If positive IC = bullish conviction on up-body days.

Measures: intensity of directional moves on informative (high volume) days.
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

    # Daily candle body ratio with sign
    rng = df['high'] - df['low']
    df['body'] = (df['close'] - df['open']) / (rng + 1e-10)  # signed, range [-1, 1]

    # Volume ratio
    df['ma20_vol'] = df.groupby('stock_code')['volume'].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    df['vol_ratio'] = np.log1p(df['volume'] / (df['ma20_vol'] + 1))

    # Signal: sign(body) * body_abs * vol_ratio
    # = volume-weighted directional conviction
    df['signal'] = np.sign(df['body']) * np.abs(df['body']) * df['vol_ratio']
    df.loc[df['body'].isna() | (rng < 1e-10), 'signal'] = np.nan

    # Rolling 20d mean
    window = 20
    df['factor_raw'] = df.groupby('stock_code')['signal'].transform(
        lambda x: x.rolling(window, min_periods=window).mean())

    df = df.dropna(subset=['factor_raw'])

    # Log transform
    df['factor_log'] = np.log(df['factor_raw'].clip(lower=0.01) + 0.01)

    # Market cap neutralization
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1))
    df = df.dropna(subset=['factor_log', 'log_amount_20d'])

    output_rows = []
    for date, d in df.groupby('date'):
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
    outpath = DATA_DIR / "factor_cbi_v1.csv"
    result.to_csv(outpath, index=False)
    print(f"✅ Saved: {outpath}")
    print(f"   Shape: {result.shape}")


if __name__ == "__main__":
    main()
