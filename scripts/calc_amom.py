#!/usr/bin/env python3
"""
Amount Momentum Factor (AMOM)

Core idea: Cross-sectional momentum in traded amount (not price).
If a stock's trading amount is accelerating, it signals growing attention/information flow.
Compare short-term (5d) vs medium-term (20d) average amount.

signal = MA5(amount) / MA20(amount)  → amount acceleration

Economic logic:
- Amount ↑ = more capital changing hands = more attention/information = informative
- Amount ↓ = liquidity drying up = disinterest

Direction: 
- Positive IC means amount acceleration → positive forward returns
  (= growing attention stocks outperform)

This is DIFFERENT from turnover:
- Turnover = volume / shares outstanding (depends on float)
- Amount = volume * price (absolute trading value, unit: CNY)
- Amount captures absolute capital flow, not percentage-of-float
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

    # MA5/MA20 amount ratio = amount acceleration
    df['ma5_amt'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(5, min_periods=5).mean())
    df['ma20_amt'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(20, min_periods=10).mean())

    # Log ratio (natural measure of relative change)
    good = (df['ma5_amt'] > 0) & (df['ma20_amt'] > 0)
    df.loc[good, 'factor_raw'] = np.log(df.loc[good, 'ma5_amt'] / df.loc[good, 'ma20_amt'])

    # Market cap neutralization
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1))

    # Filter
    df = df.dropna(subset=['factor_raw', 'log_amount_20d'])
    df = df[df['factor_raw'].abs() < 5]   # outlier filter

    output_rows = []
    for date, d in df.groupby('date'):
        d = d.copy()
        if len(d) < 100: continue
        X = np.column_stack([np.ones(len(d)), d['log_amount_20d'].values])
        y = d['factor_raw'].values
        try:
            beta, _, _, _ = lstsq(X, y, rcond=None)
            d['factor_neut'] = y - X @ beta
        except: continue

        # MAD winsorize
        med = d['factor_neut'].median()
        mad = (d['factor_neut'] - med).abs().median() * 1.4826
        if mad < 1e-6 or np.isnan(mad): continue
        d['factor_neut'] = d['factor_neut'].clip(med - 5.2*mad, med + 5.2*mad)

        std = d['factor_neut'].std()
        if std < 1e-6: continue
        d['factor_value'] = (d['factor_neut'] - d['factor_neut'].mean()) / std
        output_rows.append(d[['date', 'stock_code', 'factor_value']])

    result = pd.concat(output_rows, ignore_index=True)
    outpath = DATA_DIR / "factor_amom_v1.csv"
    result.to_csv(outpath, index=False)
    print(f"✅ Saved: {outpath} | Shape: {result.shape}")


if __name__ == "__main__":
    main()
