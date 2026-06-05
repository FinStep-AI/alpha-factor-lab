#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: 隔夜跳空波动率 v1 (Gap Volatility v1) — 优化版
变体: gap_std (40d std), gap_absm (40d |mean|), gap_cv (std/|mean|)
都取值后中性化 + 中性化zscore → 正面用
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

BASE    = Path(__file__).resolve().parent.parent
RAW_CSV = BASE / 'data' / 'csi1000_kline_raw.csv'

W   = 40
WP  = 30
NW  = 20
NP  = 10
MAD_K = 5.2


def main():
    df = pd.read_csv(RAW_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    print(f'data: {df.stock_code.nunique()} stocks, {df.date.nunique()} days')

    g = df.groupby('stock_code')
    df['prev_close'] = g['close'].shift(1)
    df['log_gap'] = np.log(df['open'] / df['prev_close'].replace(0, np.nan))

    # [fast, series-level ops only]
    df['gap_std']  = g['log_gap'].transform(lambda s: s.rolling(W, min_periods=WP).std())
    df['gap_absm'] = g['log_gap'].transform(
        lambda s: s.rolling(W, min_periods=WP).mean().abs())

    # CV = std/|mean|; where abs_mean is tiny → skip in neutralize step
    with np.errstate(divide='ignore', invalid='ignore'):
        df['gap_cv'] = df['gap_std'] / df['gap_absm']

    df['log_amount_20d'] = g['amount'].transform(
        lambda s: np.log(s.rolling(NW, min_periods=NP).mean().replace(0, np.nan)))

    usable = df.dropna(subset=['gap_std', 'gap_absm', 'gap_cv', 'log_amount_20d'])
    print(f'usable rows: {len(usable):,}')

    variants = [
        ('std_v1', 'gap_std',   -1),
        # ('absm_v1', 'gap_absm',  1),
        # ('cv_v1',   'gap_cv',   -1),
    ]

    def neutralize_zscores(sub):
        """OLS neutralize sub['factor_raw'] vs sub['log_amount_20d'], then MAD z-score."""
        sub = sub.copy()
        y = sub['factor_raw'].values.astype(float)
        x = sub['log_amount_20d'].values.astype(float)
        m = np.isfinite(y) & np.isfinite(x)
        if m.sum() < 30:
            sub['factor'] = np.nan
            return sub
        xv = x[m] - x[m].mean()
        beta = (xv * y[m]).sum() / (xv**2).sum() + 1e-12
        alpha = y[m].mean() - beta * x[m].mean()
        resid = np.full(len(y), np.nan)
        resid[m] = y[m] - (alpha + beta * x[m])
        sub['factor'] = pd.Series(resid, index=sub.index)
        med = sub['factor'].median()
        dev = (sub['factor'] - med).abs().median() * 1.4826 + 1e-12
        sub['factor'] = sub['factor'].clip(med - MAD_K*dev, med + MAD_K*dev)
        mu, sd = sub['factor'].mean(), sub['factor'].std() + 1e-12
        sub['factor'] = (sub['factor'] - mu) / sd
        return sub

    for label, col, sign in variants:
        tmp = usable.copy()
        tmp['factor_raw'] = tmp[col] * sign
        out = tmp.groupby('date', sort=False, group_keys=False).apply(neutralize_zscores)
        out = out[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
        out['stock_code'] = out['stock_code'].astype(str).str.zfill(6)
        path = BASE / 'data' / f'factor_gap_vol_{label}.csv'
        out.to_csv(path, index=False)
        print(f'  {label:20s}  {len(out):,} rows  '
              f'mean={out.factor.mean():+.4f}  std={out.factor.std():.4f}')


if __name__ == '__main__':
    main()
