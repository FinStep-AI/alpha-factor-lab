#!/usr/bin/env python3
"""
因子：ROE持续性 (ROE Persistence)
ID: roe_persistence_v1

文献动机：profit/earnings persistence 高的公司，未来收益更稳健。
这里用过去8个季度 ROE 的均值/标准差 近似持续性：
  persistence = mean(ROE_8q) / (std(ROE_8q) + 1e-6)

实现细节：
- report_date + 45d 作为可交易日
- 日频前向填充 60d
- 成交额20日均值的 log 做横截面 OLS 中性化
- MAD 截尾 + z-score
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd


FACTOR_ID = 'roe_persistence_v1'
OUT_FILE = f'data/factor_{FACTOR_ID}.csv'


def rolling_persistence(series, window=8):
    vals = series.astype(float).values
    out = np.full(len(vals), np.nan)
    for i in range(window - 1, len(vals)):
        y = vals[i - window + 1:i + 1]
        y = y[np.isfinite(y)]
        if len(y) < max(4, int(window * 0.7)):
            continue
        mu = np.mean(y)
        sd = np.std(y)
        if not np.isfinite(mu) or not np.isfinite(sd):
            continue
        out[i] = mu / (sd + 1e-6)
    return out


def neutralize(values, control):
    mask = np.isfinite(values) & np.isfinite(control)
    if mask.sum() < 30:
        return np.full_like(values, np.nan, dtype=float)
    y = values[mask]
    x = control[mask]
    X = np.column_stack([np.ones(len(x)), x])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        if mad < 1e-12:
            return np.full_like(values, np.nan, dtype=float)
        clipped = np.clip(resid, med - 5.2 * mad, med + 5.2 * mad)
        std = clipped.std()
        if std < 1e-12:
            return np.full_like(values, np.nan, dtype=float)
        z = (clipped - np.median(clipped)) / std
        out = np.full_like(values, np.nan, dtype=float)
        out[mask] = z
        return out
    except Exception:
        return np.full_like(values, np.nan, dtype=float)


def main():
    fund = pd.read_csv('data/csi1000_fundamental_cache.csv')
    kline = pd.read_csv('data/csi1000_kline_raw.csv')

    fund['report_date'] = pd.to_datetime(fund['report_date'])
    fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
    fund['roe'] = pd.to_numeric(fund['roe'], errors='coerce')
    # 限制极端值，避免 ROE 异常值主导持续性分母
    q01, q99 = fund['roe'].quantile([0.01, 0.99])
    fund['roe'] = fund['roe'].clip(q01, q99)
    fund = fund.sort_values(['stock_code', 'report_date']).reset_index(drop=True)

    fund['raw_factor'] = fund.groupby('stock_code')['roe'].transform(
        lambda s: pd.Series(rolling_persistence(s, 8), index=s.index)
    )
    factor_q = fund[['stock_code', 'report_date', 'raw_factor']].dropna().copy()
    factor_q['avail_date'] = factor_q['report_date'] + pd.DateOffset(days=45)
    factor_q = factor_q[['stock_code', 'avail_date', 'raw_factor']].rename(columns={'avail_date': 'date'})

    kline['date'] = pd.to_datetime(kline['date'])
    kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
    kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
    kline['log_amount_20d'] = kline.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean().clip(lower=1))
    )

    trading_dates = pd.Index(sorted(kline['date'].unique()))
    res = []
    for i, (stock, grp) in enumerate(factor_q.groupby('stock_code')):
        sf = grp[['date', 'raw_factor']].drop_duplicates('date', keep='last').set_index('date').sort_index()
        sf = sf.reindex(trading_dates, method='ffill', limit=60)
        sf['stock_code'] = stock
        sf = sf.dropna(subset=['raw_factor']).reset_index().rename(columns={'index': 'date'})
        res.append(sf)

    factor = pd.concat(res, ignore_index=True)
    factor = factor.merge(kline[['date', 'stock_code', 'log_amount_20d']], on=['date', 'stock_code'], how='inner')
    factor = factor.dropna(subset=['raw_factor', 'log_amount_20d'])

    out = []
    for date, grp in factor.groupby('date'):
        vals = grp['raw_factor'].values.astype(float)
        ctrl = grp['log_amount_20d'].values.astype(float)
        nz = neutralize(vals, ctrl)
        good = np.isfinite(nz)
        if good.sum() == 0:
            continue
        sub = grp.loc[good, ['date', 'stock_code']].copy()
        sub['factor'] = nz[good]
        out.append(sub)

    result = pd.concat(out, ignore_index=True)
    result['date'] = pd.to_datetime(result['date']).dt.strftime('%Y-%m-%d')
    result.to_csv(OUT_FILE, index=False, float_format='%.6f')
    print(f'saved {OUT_FILE} rows={len(result)} dates={result.date.min()}~{result.date.max()}')


if __name__ == '__main__':
    main()
