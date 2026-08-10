#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
KLINE = BASE / 'data/csi1000_kline_raw.csv'
FUND = BASE / 'data/csi1000_fundamental_cache.csv'
OUT = BASE / 'data/factor_roe_heat_instability_proxy_v1.csv'


def robust_zscore(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad < 1e-12:
        std = s.std()
        if pd.isna(std) or std < 1e-12:
            return s * np.nan
        return (s - s.mean()) / std
    clipped = s.clip(med - 3.5 * 1.4826 * mad, med + 3.5 * 1.4826 * mad)
    std = clipped.std()
    if pd.isna(std) or std < 1e-12:
        return clipped - clipped.mean()
    return (clipped - clipped.mean()) / std


def neutralize_cross_section(df: pd.DataFrame, y_col: str, x_col: str) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    valid = df[[y_col, x_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 20:
        return out
    x = valid[x_col].astype(float).values
    y = valid[y_col].astype(float).values
    X = np.column_stack([np.ones(len(valid)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    out.loc[valid.index] = robust_zscore(pd.Series(resid, index=valid.index))
    return out


def main():
    kline = pd.read_csv(KLINE)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date'])
    kline['log_mktcap_proxy'] = np.log(kline['amount'].rolling(20, min_periods=10).mean())

    fund = pd.read_csv(FUND)
    fund['report_date'] = pd.to_datetime(fund['report_date'])
    fund = fund.sort_values(['stock_code', 'report_date']).copy()

    g = fund.groupby('stock_code')
    fund['roe_l1'] = g['roe'].shift(1)
    fund['roe_l4'] = g['roe'].shift(4)
    fund['bps_l1'] = g['bps'].shift(1)
    fund['bps_l4'] = g['bps'].shift(4)
    fund['roe_yoy'] = fund['roe'] - fund['roe_l4']
    fund['roe_mean4'] = g['roe'].rolling(4, min_periods=3).mean().reset_index(level=0, drop=True)
    fund['roe_stability'] = g['roe'].rolling(4, min_periods=3).std().reset_index(level=0, drop=True)
    fund['bps_yoy'] = fund['bps'] / fund['bps_l4'] - 1
    fund['bps_qoq'] = fund['bps'] / fund['bps_l1'] - 1

    # 毛利率趋势稳定性代理：盈利改善 + 盈利水平 - 波动惩罚 - 扩表惩罚
    fund['raw_factor'] = (
        -0.25 * np.tanh(fund['roe_mean4'] / 8.0)
        + 0.55 * np.tanh(fund['roe_yoy'] / 5.0)
        + 0.35 * np.tanh(fund['roe_stability'] / 2.5)
        + 0.15 * np.tanh(fund['bps_yoy'] / 0.25)
        - 0.20 * np.tanh(fund['bps_qoq'] / 0.10)
    )

    # 财报滞后45天映射到日频
    fund['effective_date'] = fund['report_date'] + pd.Timedelta(days=45)
    daily_dates = pd.DataFrame({'date': sorted(kline['date'].unique())})

    pieces = []
    for stock, grp in fund.groupby('stock_code'):
        s = grp[['effective_date', 'raw_factor']].dropna().sort_values('effective_date')
        if s.empty:
            continue
        merged = pd.merge_asof(
            daily_dates,
            s.rename(columns={'effective_date': 'date'}),
            on='date',
            direction='backward'
        )
        merged['stock_code'] = stock
        pieces.append(merged)

    factor_daily = pd.concat(pieces, ignore_index=True)
    merged = kline[['date', 'stock_code', 'amount']].merge(
        factor_daily, on=['date', 'stock_code'], how='left'
    )
    merged['log_mktcap_proxy'] = np.log(merged['amount'].replace(0, np.nan))

    merged['factor'] = merged.groupby('date', group_keys=False).apply(
        lambda x: neutralize_cross_section(x, 'raw_factor', 'log_mktcap_proxy')
    ).reset_index(level=0, drop=True)

    out = merged[['date', 'stock_code', 'factor']].dropna().copy()
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    out.to_csv(OUT, index=False)
    print(f'saved {OUT} rows={len(out)} dates={out.date.nunique()} stocks={out.stock_code.nunique()}')
    print(out.head().to_string())


if __name__ == '__main__':
    main()
