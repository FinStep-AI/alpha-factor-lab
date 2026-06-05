#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: 隔夜跳空均值粘性 v1 (Gap Mean Intensity v1)
原理: 40日平滑隔夜跳空收益率均值。高值 = 持续跳空向上的股票（开盘持续高于前收），
     本质是把 '持续性隔夜收益' 提炼为单一粘性信号，而非简单的方向/幅度分解。
     与 overnight_momentum/gap_momentum 角度不同：
       - overnight_momentum 看重隔夜vs日内差；
       - gap_momentum  看重缺口方向一致性；
       本因子看重隔夜跳空的绝对平均水平——持续稳定高开的股票。

成交额 OLS 中性化 + MAD 5.2σ winsorize + z-score
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
RAW_CSV = BASE / 'data' / 'csi1000_kline_raw.csv'
OUT_CSV = BASE / 'data' / 'factor_gap_mean_v1.csv'

GAP_WINDOW    = 40   # gap smoothing window (calendar days, working days ≈ trading days)
GAP_MIN_P     = 30
NEUT_WINDOW   = 20
NEUT_MIN_P    = 10
MAD_K         = 5.2  # winsorize


def mad_winsorize(s: pd.Series, k: float = MAD_K) -> pd.Series:
    med = s.median()
    dev = (s - med).abs().median() * 1.4826 + 1e-12
    lo = med - k * dev
    hi = med + k * dev
    return s.clip(lo, hi)


def cross_section_neutralize(g: pd.DataFrame) -> pd.DataFrame:
    """OLS neutralize factor_raw vs log_amount_20d, then winsorize+zscore."""
    g = g.copy()
    y = g['factor_raw'].values.astype(float)
    x = g['log_amount_20d'].values.astype(float)

    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 30:
        g['factor'] = np.nan
        return g

    x_m, y_m = x[mask], y[mask]
    x_dm = x_m - x_m.mean()
    beta = (x_dm * y_m).sum() / (x_dm ** 2).sum() + 1e-12
    alpha = y_m.mean() - beta * x_m.mean()
    resid = np.full(len(y), np.nan)
    resid[mask] = y_m - (alpha + beta * x_m)

    g['factor'] = mad_winsorize(pd.Series(resid, index=g.index))
    g['factor'] = g['factor'].fillna(g['factor'].median())
    mu = g['factor'].mean()
    sd = g['factor'].std() + 1e-12
    g['factor'] = (g['factor'] - mu) / sd
    return g


def main():
    df = pd.read_csv(RAW_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)

    print(f"raw: {df['stock_code'].nunique()} stocks, {df['date'].nunique()} days")

    # --- per-stock pre-compute ---
    g = df.groupby('stock_code', sort=False)

    df['prev_close'] = g['close'].shift(1)
    df['log_gap']    = np.log((df['open'] / df['prev_close'].replace(0, np.nan)))

    df['gap_mean_40d'] = g['log_gap'].transform(
        lambda x: x.rolling(GAP_WINDOW, min_periods=GAP_MIN_P).mean()
    )
    df['log_amount_20d'] = g['amount'].transform(
        lambda x: np.log(x.rolling(NEUT_WINDOW, min_periods=NEUT_MIN_P).mean().replace(0, np.nan))
    )

    # Forward-fill: only every 5 trading days (weekly) has a meaningful change,
    # so we keep the daily cross-section intact.
    df['factor_raw'] = df['gap_mean_40d']

    # --- cross-section neutralize per date ---
    valid = df.dropna(subset=['factor_raw', 'log_amount_20d']).copy()
    print(f"rows with valid factor_raw: {len(valid):,}")

    out = (
        valid.groupby('date', sort=False, group_keys=False)
        .apply(cross_section_neutralize)
    )
    out = out[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    out['stock_code'] = out['stock_code'].astype(str).str.zfill(6)

    out.to_csv(OUT_CSV, index=False)
    print(f"saved → {OUT_CSV}  ({len(out):,} rows)")
    print(out.groupby('date')['stock_code'].count().describe())


if __name__ == '__main__':
    main()
