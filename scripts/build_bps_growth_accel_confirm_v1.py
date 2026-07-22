#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BPS Growth Acceleration + ROE Confirmation v1

思路：
1. 用 BPS 同比增长率衡量内生积累；
2. 再看 BPS 同比增长率是否继续加速（latest_yoy - prev_yoy）；
3. 用 ROE 同比改善做确认，避免“纯账面扩张但盈利恶化”；
4. 截面 z-score 后等权合成；
5. 每个交易日相对 log_mktcap（bps * close）做 OLS 市值中性化。

输出：data/factor_bps_growth_accel_confirm_v1.csv
列：date, stock_code, factor
"""

import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUND_CSV = os.path.join(BASE, 'data', 'csi1000_fundamental_cache.csv')
KLINE_CSV = os.path.join(BASE, 'data', 'csi1000_kline_raw.csv')
OUT_CSV = os.path.join(BASE, 'data', 'factor_bps_growth_accel_confirm_v1.csv')

REPORT_LAG_DAYS = 45


def cs_zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad < 1e-12:
        clipped = s.clip(lower=s.quantile(0.01), upper=s.quantile(0.99))
    else:
        robust_z = (s - med) / (1.4826 * mad)
        clipped = s.where(robust_z.abs() <= 5, med + np.sign(robust_z) * 5 * 1.4826 * mad)
    std = clipped.std()
    if pd.isna(std) or std < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (clipped - clipped.mean()) / std


def neutralize(df: pd.DataFrame) -> pd.DataFrame:
    work = df.dropna(subset=['raw_factor', 'log_mktcap']).copy()
    if len(work) < 30:
        work['factor'] = work['raw_factor']
        return work[['stock_code', 'factor']]
    X = np.column_stack([np.ones(len(work)), work['log_mktcap'].values])
    y = work['raw_factor'].values
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    work['factor'] = y - X @ coef
    return work[['stock_code', 'factor']]


def build_report_factor(fund: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    avail = fund[fund['report_date'] <= cutoff].copy()
    records = []
    for code, g in avail.groupby('stock_code'):
        g = g.sort_values('report_date').copy()
        if len(g) < 6:
            continue
        g['month'] = g['report_date'].dt.month
        g['year'] = g['report_date'].dt.year
        latest = g.iloc[-1]
        prev_same = g[(g['year'] == latest['year'] - 1) & (g['month'] == latest['month'])]
        prev2_same = g[(g['year'] == latest['year'] - 2) & (g['month'] == latest['month'])]
        prev_q = g.iloc[-2]
        prev_q_same = g[(g['year'] == prev_q['year'] - 1) & (g['month'] == prev_q['month'])]

        if len(prev_same) == 0 or len(prev_q_same) == 0:
            continue

        prev_same = prev_same.iloc[-1]
        prev_q_same = prev_q_same.iloc[-1]

        bps_yoy = np.nan
        prev_bps_yoy = np.nan
        roe_yoy = np.nan

        if pd.notna(latest['bps']) and pd.notna(prev_same['bps']) and abs(prev_same['bps']) > 1e-6:
            bps_yoy = latest['bps'] / prev_same['bps'] - 1
        if pd.notna(prev_q['bps']) and pd.notna(prev_q_same['bps']) and abs(prev_q_same['bps']) > 1e-6:
            prev_bps_yoy = prev_q['bps'] / prev_q_same['bps'] - 1
        if pd.notna(latest['roe']) and pd.notna(prev_same['roe']) and abs(prev_same['roe']) > 0.1:
            roe_yoy = (latest['roe'] - prev_same['roe']) / abs(prev_same['roe'])

        if np.isnan(bps_yoy) or np.isnan(prev_bps_yoy):
            continue

        bps_accel = bps_yoy - prev_bps_yoy
        if not np.isfinite(bps_accel):
            continue

        records.append({
            'stock_code': code,
            'bps_yoy': bps_yoy,
            'bps_accel': bps_accel,
            'roe_yoy': 0.0 if np.isnan(roe_yoy) else roe_yoy,
        })

    fac = pd.DataFrame(records)
    if fac.empty:
        return fac

    for col in ['bps_yoy', 'bps_accel', 'roe_yoy']:
        fac[col] = fac[col].replace([np.inf, -np.inf], np.nan)
        fac = fac.dropna(subset=[col]) if col != 'roe_yoy' else fac
        fac[f'{col}_z'] = cs_zscore(fac[col].fillna(0.0))

    fac['raw_factor'] = 0.4 * fac['bps_yoy_z'] + 0.4 * fac['bps_accel_z'] + 0.2 * fac['roe_yoy_z']
    return fac[['stock_code', 'raw_factor']]


def main():
    fund = pd.read_csv(FUND_CSV)
    fund['report_date'] = pd.to_datetime(fund['report_date'])
    fund = fund.sort_values(['stock_code', 'report_date']).reset_index(drop=True)

    kline = pd.read_csv(KLINE_CSV)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)

    report_dates = sorted(fund['report_date'].unique())
    trade_dates = sorted(kline['date'].unique())
    by_report = {}
    for rd in report_dates:
        fac = build_report_factor(fund, rd)
        if len(fac) > 0:
            by_report[rd] = fac
            print(f'report {rd.date()}: {len(fac)} stocks')

    out = []
    lag = pd.Timedelta(days=REPORT_LAG_DAYS)
    prev_rd = None
    for td in trade_dates:
        available = [rd for rd in report_dates if rd + lag <= td]
        if not available:
            continue
        rd = max(available)
        fac = by_report.get(rd)
        if fac is None or fac.empty:
            continue
        day = kline[kline['date'] == td][['stock_code', 'close']].copy()
        base = fac.merge(day, on='stock_code', how='inner')
        base['mktcap_proxy'] = base['close'].clip(lower=0.01) * 1.0
        # 用 BPS*close 近似市值/股本代理；BPS缺失已在 fac 阶段过滤
        latest_bps_map = fund[fund['report_date'] == rd][['stock_code', 'bps']]
        base = base.merge(latest_bps_map, on='stock_code', how='left')
        base['mktcap_proxy'] = (base['close'].clip(lower=0.01) * base['bps'].clip(lower=0.01))
        base['log_mktcap'] = np.log(base['mktcap_proxy'].clip(lower=1e-4))
        neu = neutralize(base)
        neu['date'] = td.strftime('%Y-%m-%d')
        out.append(neu[['date', 'stock_code', 'factor']])
        if rd != prev_rd:
            print(f'use report {rd.date()} from trade date {td.date()}')
            prev_rd = rd

    result = pd.concat(out, ignore_index=True)
    result.to_csv(OUT_CSV, index=False)
    print(f'saved {OUT_CSV}, rows={len(result)}, dates={result.date.nunique()}, stocks={result.stock_code.nunique()}')


if __name__ == '__main__':
    main()
