
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
毛利率稳定改善代理因子 v1（Growth / Quality）

数据约束下没有毛利率，使用 ROE 与 BPS 近似经营质量：
- 高且稳定的 ROE → 更像高毛利 / 高盈利质量企业
- ROE 同比改善 → 盈利能力在变好
- BPS 同比增速过快 → 可能更多来自扩张/增资，给予温和惩罚

raw = 0.55 * tanh(roe_level / 8)
    + 0.35 * tanh(roe_yoy / 6)
    - 0.25 * tanh(roe_vol / 4)
    - 0.15 * tanh(bps_yoy / 20)

随后映射到日频，可交易滞后 45 天，并对 log_mktcap 做横截面中性化。
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
KLINE = DATA / 'csi1000_kline_raw.csv'
FUND = DATA / 'csi1000_fundamental_cache.csv'
OUT = DATA / 'factor_gross_margin_stability_proxy_v1.csv'
DISCLOSE_DELAY = 45
MAD_K = 5.0


def mad_clip(s, k=MAD_K):
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return s
    lo, hi = med - k * 1.4826 * mad, med + k * 1.4826 * mad
    return s.clip(lo, hi)


def neutralize_one_day(g):
    g = g.copy()
    x = g['log_mktcap'].astype(float)
    y = g['raw_factor'].astype(float)
    mask = x.notna() & y.notna()
    if mask.sum() < 20:
        g['factor'] = np.nan
        return g
    xv = x[mask].values
    yv = y[mask].values
    X = np.column_stack([np.ones(len(xv)), xv])
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    resid = yv - X @ beta
    s = pd.Series(resid, index=g.index[mask])
    s = mad_clip(s)
    std = s.std()
    if pd.isna(std) or std == 0:
        g['factor'] = np.nan
    else:
        g.loc[s.index, 'factor'] = (s - s.mean()) / std
    return g


def main():
    kline = pd.read_csv(KLINE, dtype={'stock_code': str})
    kline['date'] = pd.to_datetime(kline['date'])
    kline['stock_code'] = kline['stock_code'].str.strip()
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
    kline['mktcap'] = kline['close'] * kline['volume'] / kline['turnover'].replace(0, np.nan)
    kline['mktcap'] = kline['mktcap'].where(kline['mktcap'] > 0)
    kline['log_mktcap'] = np.log(kline['mktcap'])

    fund = pd.read_csv(FUND, dtype={'stock_code': str})
    fund['stock_code'] = fund['stock_code'].str.strip()
    fund['report_date'] = pd.to_datetime(fund['report_date'])
    fund = fund.sort_values(['stock_code', 'report_date']).reset_index(drop=True)

    rows = []
    for sc, g in fund.groupby('stock_code'):
        g = g.sort_values('report_date').reset_index(drop=True)
        for i in range(4, len(g)):
            roe_now = g.loc[i, 'roe']
            roe_prev4 = g.loc[i-4, 'roe']
            bps_now = g.loc[i, 'bps']
            bps_prev4 = g.loc[i-4, 'bps']
            hist = g.loc[max(0, i-3):i, 'roe'].astype(float)
            if hist.notna().sum() < 3:
                continue
            roe_level = hist.mean()
            roe_vol = hist.std(ddof=0)
            if pd.isna(roe_now) or pd.isna(roe_prev4):
                continue
            roe_yoy = roe_now - roe_prev4
            if pd.isna(bps_now) or pd.isna(bps_prev4) or bps_prev4 <= 0:
                bps_yoy = np.nan
            else:
                bps_yoy = (bps_now / bps_prev4 - 1.0) * 100.0
            raw = (
                0.55 * np.tanh(roe_level / 8.0)
                + 0.35 * np.tanh(roe_yoy / 6.0)
                - 0.25 * np.tanh((0.0 if pd.isna(roe_vol) else roe_vol) / 4.0)
                - 0.15 * np.tanh((0.0 if pd.isna(bps_yoy) else bps_yoy) / 20.0)
            )
            rows.append({
                'stock_code': sc,
                'effective_date': g.loc[i, 'report_date'] + pd.Timedelta(days=DISCLOSE_DELAY),
                'raw_factor': raw,
            })

    rpt = pd.DataFrame(rows).sort_values(['stock_code', 'effective_date'])

    out_frames = []
    for sc, px in kline.groupby('stock_code'):
        px = px[['date', 'stock_code', 'log_mktcap']].sort_values('date').reset_index(drop=True)
        rf = rpt[rpt['stock_code'] == sc][['effective_date', 'raw_factor']].sort_values('effective_date')
        if rf.empty:
            continue
        merged = pd.merge_asof(px, rf, left_on='date', right_on='effective_date', direction='backward')
        merged = merged[['date', 'stock_code', 'log_mktcap', 'raw_factor']]
        out_frames.append(merged)

    panel = pd.concat(out_frames, ignore_index=True)
    panel = panel.groupby('date', group_keys=False).apply(neutralize_one_day)
    out = panel[['date', 'stock_code', 'factor']].dropna().copy()
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    out.to_csv(OUT, index=False)
    print(f'saved {OUT} rows={len(out):,} dates={out.date.min()}~{out.date.max()}')


if __name__ == '__main__':
    main()
