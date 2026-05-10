#!/usr/bin/env python3
"""
因子：ROE时序自回归持续性 v1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
复现 Chahine et al. (2018) "Profit Persistence and Stock Returns"
SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3211826

已入库的 roe_persistence_neg_v1 同源于论文但使用 CV（std/mean）度量平稳性。
本因子维度创新：直接使用 8q 滚动 ROE 序列自相关 as the core metric with
temporal persistence of earnings projected as quality characteristic.

输出 data/factor_roe_autocorr_pers_v1.csv
Barra: Quality / Growth
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

FUND    = 'data/csi1000_fundamental_cache.csv'
KLINE   = 'data/csi1000_kline_raw.csv'
OUTPUT  = 'data/factor_roe_autocorr_pers_v1.csv'
WINDOW  = 8   # quarters
HALF    = 4   # each half
LAG_DAYS = 45
MIN_VALID = 4


# ── helper ──

def rolling_autocorr(roes, window=WINDOW, half=HALF):
    """8q滚动窗口内，前后各4q的Pearson相关。"""
    n = len(roes)
    out = np.full(n, np.nan)
    if n < window:
        return out
    for i in range(window - 1, n):
        f = roes[i - window + 1 : i - half + 1]
        b = roes[i - half + 1     : i + 1]
        fv = f[np.isfinite(f)]
        bv = b[np.isfinite(b)]
        if len(fv) >= MIN_VALID and len(bv) >= MIN_VALID:
            sf, sb = fv.std(), bv.std()
            if sf > 1e-10 and sb > 1e-10:
                r = np.corrcoef(fv, bv)[0, 1]
                out[i] = float(r) if np.isfinite(r) else np.nan
    return out


# ── main ──

def main():
    print("[0] ROE Autocorr Persistence v1  |  w=8q lag=45d  |  SSRN 3211826",
          flush=True)

    # 1 · 基本面 → 算自相关
    print("[1] 基本面 + 8q ROE自相关 ...", flush=True)
    fund = (pd.read_csv(FUND)
            .assign(report_date=lambda x: pd.to_datetime(x['report_date']))
            .dropna(subset=['roe'])
            .sort_values(['stock_code','report_date'])
            .drop_duplicates(['stock_code','report_date'])
            .reset_index(drop=True))

    recs = []
    for sc, g in fund.groupby('stock_code'):
        g = g.sort_values('report_date').reset_index(drop=True)
        roes = g['roe'].values.astype(float)
        dates = g['report_date'].values
        corr_arr = rolling_autocorr(roes)
        for i in range(WINDOW-1, len(g)):
            r = corr_arr[i]
            if np.isfinite(r):
                recs.append({'stock_code': sc,
                             'report_date': pd.Timestamp(dates[i]),
                             'raw': r})

    raw = pd.DataFrame(recs)
    print(f"    观测 {len(raw):,},  {raw['stock_code'].nunique()} 只有效",
          flush=True)
    if raw.empty:
        print("[ERROR] 无数据"); return {}

    raw['avail_date'] = raw['report_date'] + pd.Timedelta(days=LAG_DAYS)

    # 2 · 展平到日频
    print("[2] 展平日频 ...", flush=True)
    kline  = pd.read_csv(KLINE, parse_dates=['date'])
    amt_lg = kline[['date','stock_code','amount']].drop_duplicates(
                 subset=['date','stock_code'])
    amt_lg['log_amount'] = np.log(amt_lg['amount'].clip(1))

    trade_dates = sorted(kline['date'].unique())
    rows = []
    for td in trade_dates:
        prev = raw[raw['avail_date'] <= td]
        if prev.empty:
            continue
        lat = prev.sort_values('avail_date').groupby('stock_code').last()
        for sc, row in lat.iterrows():
            rows.append({'date': td, 'stock_code': int(sc), 'raw': row['raw']})

    day = pd.DataFrame(rows).merge(
        amt_lg, on=['date','stock_code'], how='left')
    print(f"    展开 {len(day):,} 行, {day['stock_code'].nunique()} 只, "
          f"{day['date'].nunique()} 日", flush=True)

    # 3 · 截面OLS中 → MAD → z-score
    print("[3] OLS中性化 + MAD + z-score ...", flush=True)

    def cross_ols(y_arr, x_arr):
        mask = np.isfinite(y_arr) & np.isfinite(x_arr)
        if mask.sum() < 20:
            return y_arr.copy()
        X = np.column_stack([np.ones(int(mask.sum())), x_arr[mask]])
        b, *_ = np.linalg.lstsq(X, y_arr[mask], rcond=None)
        res = y_arr.copy()
        res[~mask] = np.nan
        res[mask]  = y_arr[mask] - X @ b
        return res

    day = day.sort_values('date')
    u_dates = sorted(day['date'].unique())

    records = []
    for td in u_dates:
        sub = day[day['date'] == td].copy()
        v   = sub['raw'].values.astype(float)
        amt = sub['log_amount'].values

        # OLS
        v = cross_ols(v, amt)

        # MAD winsorize
        val = pd.Series(v).dropna()
        if len(val) >= 10:
            med = val.median(); mad = np.median(np.abs(val - med))
            if mad > 1e-10:
                sc_mad = 1.4826 * mad
                v = np.where(~np.isnan(v), v, np.nan)
                v = pd.Series(v).clip(med - 3*sc_mad, med + 3*sc_mad).values

            # z-score
            mu = np.nanmean(v); sd = np.nanstd(v)
            if sd > 1e-10:
                v = (v - mu) / sd

        for code, val in zip(sub['stock_code'].values, v):
            if np.isfinite(val):
                records.append({'date': td.strftime('%Y-%m-%d'),
                                'stock_code': str(int(code)).zfill(6),
                                'factor_value': round(float(val), 6)})

    out = pd.DataFrame(records).drop_duplicates(['date','stock_code'])
    p = Path(OUTPUT); p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)

    print("\n=== 因子统计 ===", flush=True)
    print(f"  行数:     {len(out):,}", flush=True)
    print(f"  股票数:   {out['stock_code'].nunique()}", flush=True)
    print(f"  日期范围: {out['date'].min()} ~ {out['date'].max()}", flush=True)
    print(f"  均值:     {out['factor_value'].mean():+.4f}", flush=True)
    print(f"  标准差:   {out['factor_value'].std():.4f}", flush=True)
    print(f" 区间:     [{out['factor_value'].min():.4f}, "
          f"{out['factor_value'].max():.4f}]", flush=True)
    print(f"\n  ✅ {p}", flush=True)
    return out


if __name__ == '__main__':
    main()
