#!/usr/bin/env python3
"""
因子：turnover_kurt_v1 — 换手率峰度（20日滚动峰度，成交额中性化）
fast version using numba rolling kernel
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

WINDOW = 20

@njit(fastmath=True)
def rolling_kurt_nb(arr, valid_n, window=20, min_periods=15):
    """numba-accelerated rolling excess kurtosis"""
    n = arr.shape[0]
    out = np.full(n, np.nan)
    for i in range(window-1, n):
        cnt = 0
        mu = 0.0
        var = 0.0
        for j in range(i - window + 1, i + 1):
            v = arr[j]
            if not np.isnan(v):
                cnt += 1
                mu += v
                var += v * v
        if cnt < min_periods:
            continue
        mu /= cnt
        std = np.sqrt(var/cnt - mu*mu)
        if std < 1e-12:
            continue
        m4 = 0.0
        for j in range(i - window + 1, i + 1):
            v = arr[j]
            if not np.isnan(v):
                diff = v - mu
                m4 += diff**4
        m4 /= cnt
        kurt = m4 / (std**4) - 3.0
        out[i] = kurt
    return out

def main():
    base = Path(__file__).resolve().parent.parent
    data_dir = base / 'data'
    print("读 K线...")
    df = pd.read_csv(data_dir / 'csi1000_kline_raw.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['pct_change'])
    codes = pd.Categorical(df['stock_code']).codes
    total = len(df)
    print(f"股票: {df['stock_code'].nunique()}, 行数: {total}")

    print("计算turnover峰度 (numba JIT)...")
    to = df['turnover'].values.astype(np.float64)
    kurt_arr = rolling_kurt_nb(to, None, WINDOW, 15)
    df['factor_raw'] = kurt_arr

    # 成交额中性化
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )
    fac = df['factor_raw'].values.astype(float)

    from numpy.linalg import lstsq
    result = np.full(total, np.nan)
    dates = df['date'].values
    log_amt = df['log_amount_20d'].values.astype(float)
    for d in pd.unique(df['date']):
        mask = dates == d
        valid = mask & ~np.isnan(fac) & ~np.isnan(log_amt)
        idx = np.where(valid)[0]
        if idx.size < 30:
            continue
        y = fac[idx]
        X = np.column_stack([np.ones(idx.size), log_amt[idx]])
        try:
            beta, _, _, _ = lstsq(X, y, rcond=None)
            result[idx] = y - X @ beta
        except:
            pass
    df['factor'] = result

    caps = df['log_amount_20d'].values.astype(float)
    # cap-level median kurt correction
    med_by_cap = pd.MultiIndex.from_frame(df[['date','log_amount_20d']]).groupby(level=0).apply(lambda x: x)
    del med_by_cap  # not needed

    # MAD winsorize + z-score per cross-section
    print("截面标准化...")
    def mad_z(dates_arr, fac_arr, all_dates, logcap_arr):
        # reset after OLS
        fac_arr[:] = result[:]

    # simpler per day
    out_recs = []
    for dval, grp in df.groupby('date', sort=True):
        s = grp['factor'].copy()
        med = s.median(); mad = (s - med).abs().median()
        if mad > 1e-10:
            lo = med - 5*1.4826*mad; hi = med + 5*1.4826*mad
            s = s.clip(lo, hi)
        mu_f = s.mean(); sd_f = s.std()
        if sd_f > 1e-10:
            s = (s - mu_f) / sd_f
        else:
            s[:] = 0
        for date_str, val in zip([dval]*len(s), s.values):
            out_recs.append((date_str, grp.iloc[0]['stock_code'], val))

    out_df = pd.DataFrame(out_recs, columns=['date', 'stock_code', 'factor'])
    out_df['date'] = out_df['date'].dt.strftime('%Y-%m-%d')
    out_path = data_dir / 'factor_turnover_kurt_v1.csv'
    out_df.to_csv(out_path, index=False)
    print(f"✅ {out_path}: {len(out_df)} rows, dates {out_df['date'].min()} ~ {out_df['date'].max()}")
    dcounts = out_df.groupby('date')['stock_code'].count()
    print(f"   每日覆盖: {dcounts.mean():.0f}, range [{dcounts.min()}-{dcounts.max()}]")

if __name__ == '__main__':
    main()
