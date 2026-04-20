#!/usr/bin/env python3
"""Turnover Acceleration Factor (TO_ACC)

MA5(turnover) / MA20(turnover) - 衡量短期换手率相对中长期的加速程度。
高加速 = 近期关注度急剧增加 = 可能是信息驱动或动量启动 → 正向alpha（反转角度：换手加速后回吐? 需要验证）

两个版本用回测决定方向。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import lstsq

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main(flip=False):
    df = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)

    # raw ratio
    df['ma5_to'] = df.groupby('stock_code')['turnover'].transform(
        lambda x: x.rolling(5, min_periods=5).mean())
    df['ma20_to'] = df.groupby('stock_code')['turnover'].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    df['factor_raw'] = np.log(df['ma5_to'] / (df['ma20_to'] + 1e-8))

    if flip:
        df['factor_raw'] = -df['factor_raw']

    # log_amount neutralization
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1))

    df = df.dropna(subset=['factor_raw', 'log_amount_20d'])

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
        med = d['factor_neut'].median()
        mad = (d['factor_neut'] - med).abs().median() * 1.4826
        if mad < 1e-6: continue
        d['factor_neut'] = d['factor_neut'].clip(med - 5.2*mad, med + 5.2*mad)
        std = d['factor_neut'].std()
        if std < 1e-6: continue
        d['factor_value'] = (d['factor_neut'] - d['factor_neut'].mean()) / std
        output_rows.append(d[['date', 'stock_code', 'factor_value']])

    result = pd.concat(output_rows, ignore_index=True)
    suffix = "_neg" if flip else ""
    outpath = DATA_DIR / f"factor_to_acc{suffix}.csv"
    result.to_csv(outpath, index=False)
    print(f"✅ Saved: {outpath}")
    print(f"   Shape: {result.shape}")


if __name__ == "__main__":
    main(flip=False)
