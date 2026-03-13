#!/usr/bin/env python3
"""
ROE Quality v1 Factor — Barra "Quality" style

Quality = mean(single-quarter ROE, last 8Q) / std(single-quarter ROE, last 8Q)
  = "ROE Sharpe" — stable positive earnings = high quality

Neutralized vs log(close) as size proxy.
"""

import pandas as pd
import numpy as np
import os, json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUND_CSV = os.path.join(BASE, "data", "csi1000_fundamental_cache.csv")
KLINE_CSV = os.path.join(BASE, "data", "csi1000_kline_raw.csv")
OUTPUT_DIR = os.path.join(BASE, "output", "roe_quality_v1")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    fund = pd.read_csv(FUND_CSV)
    fund['report_date'] = pd.to_datetime(fund['report_date'])
    fund = fund.sort_values(['stock_code', 'report_date']).reset_index(drop=True)

    kline = pd.read_csv(KLINE_CSV)
    kline = kline.rename(columns={'date': 'trade_date'})
    kline['trade_date'] = pd.to_datetime(kline['trade_date'])
    kline = kline.sort_values(['stock_code', 'trade_date']).reset_index(drop=True)
    return fund, kline


def compute_single_quarter_roe(fund):
    """Convert cumulative YTD ROE to single-quarter ROE."""
    fund = fund.copy()
    fund['quarter'] = fund['report_date'].dt.month
    fund['year'] = fund['report_date'].dt.year

    # Vectorized: pivot so we can compute differences
    records = []
    for code, grp in fund.groupby('stock_code'):
        grp = grp.sort_values('report_date').reset_index(drop=True)
        for i, row in grp.iterrows():
            q = row['quarter']
            cum_roe = row['roe']
            if pd.isna(cum_roe):
                continue
            if q == 3:  # Q1
                sq_roe = cum_roe
            else:
                prev_q = {6: 3, 9: 6, 12: 9}[q]
                prev = grp[(grp['year'] == row['year']) & (grp['quarter'] == prev_q)]
                if len(prev) == 0 or pd.isna(prev.iloc[0]['roe']):
                    sq_roe = np.nan
                else:
                    sq_roe = cum_roe - prev.iloc[0]['roe']
            records.append({
                'stock_code': code,
                'report_date': row['report_date'],
                'sq_roe': sq_roe,
            })
    return pd.DataFrame(records)


def compute_quality_at_report(sq_df, cutoff_report, n_quarters=8):
    """Compute quality factor using data up to cutoff_report."""
    avail = sq_df[sq_df['report_date'] <= cutoff_report].copy()
    records = []
    for code, grp in avail.groupby('stock_code'):
        grp = grp.sort_values('report_date').dropna(subset=['sq_roe'])
        if len(grp) < 4:
            continue
        tail = grp.tail(n_quarters)
        mean_roe = tail['sq_roe'].mean()
        std_roe = tail['sq_roe'].std()
        if std_roe < 0.01:
            quality = mean_roe * 100
        else:
            quality = mean_roe / std_roe
        records.append({'stock_code': code, 'quality_raw': quality})
    return pd.DataFrame(records)


def neutralize(df, kline_day):
    """Neutralize quality_raw vs log(close)."""
    merged = df.merge(kline_day[['stock_code', 'close']], on='stock_code', how='inner')
    if len(merged) < 50:
        merged['factor'] = merged['quality_raw']
        return merged[['stock_code', 'factor']]

    merged['log_close'] = np.log(merged['close'].clip(lower=0.01))
    X = np.column_stack([np.ones(len(merged)), merged['log_close'].values])
    y = merged['quality_raw'].values
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if mask.sum() < 50:
        merged['factor'] = merged['quality_raw']
        return merged[['stock_code', 'factor']]

    from numpy.linalg import lstsq
    coef, _, _, _ = lstsq(X[mask], y[mask], rcond=None)
    merged['factor'] = y - X @ coef
    return merged[['stock_code', 'factor']]


def main():
    print("Loading data...")
    fund, kline = load_data()

    print("Computing single-quarter ROE...")
    sq_df = compute_single_quarter_roe(fund)
    sq_df['report_date'] = pd.to_datetime(sq_df['report_date'])
    print(f"  {len(sq_df)} records, {sq_df['stock_code'].nunique()} stocks")

    report_dates = sorted(sq_df['report_date'].unique())
    trade_dates = sorted(kline['trade_date'].unique())
    REPORT_LAG = pd.Timedelta(days=45)

    # Pre-compute quality factor for each report cutoff
    print("Pre-computing quality per report period...")
    quality_by_report = {}
    for rd in report_dates:
        qf = compute_quality_at_report(sq_df, rd)
        if len(qf) > 0:
            quality_by_report[rd] = qf
    print(f"  {len(quality_by_report)} report periods with quality factors")

    # Map each trade date to latest available report
    print("Building factor time series...")
    factor_records = []
    prev_rd_used = None

    for td in trade_dates:
        available = [rd for rd in report_dates if rd + REPORT_LAG <= td]
        if not available:
            continue
        latest_rd = max(available)

        if latest_rd not in quality_by_report:
            continue

        qf = quality_by_report[latest_rd]
        kline_day = kline[kline['trade_date'] == td]
        neutralized = neutralize(qf, kline_day)

        for _, row in neutralized.iterrows():
            factor_records.append({
                'stock_code': row['stock_code'],
                'trade_date': td,
                'factor': row['factor']
            })

        if latest_rd != prev_rd_used:
            print(f"  {td.date()}: using report {latest_rd.date()}, {len(neutralized)} stocks")
            prev_rd_used = latest_rd

    factor_df = pd.DataFrame(factor_records)
    print(f"Factor TS: {len(factor_df)} records, {factor_df['trade_date'].nunique()} dates")

    # === Backtest ===
    print("\nRunning backtest...")
    N_GROUPS = 5
    FORWARD = 5
    COST = 0.003

    rebalance_dates = sorted(factor_df['trade_date'].unique())[::FORWARD]
    group_rets = {g: [] for g in range(1, N_GROUPS + 1)}
    ic_series = []

    for i in range(len(rebalance_dates) - 1):
        rd = rebalance_dates[i]
        rd_next = rebalance_dates[i + 1]

        fv = factor_df[factor_df['trade_date'] == rd][['stock_code', 'factor']].copy()
        if len(fv) < 100:
            continue

        lo, hi = fv['factor'].quantile([0.01, 0.99])
        fv['factor'] = fv['factor'].clip(lo, hi)

        p0 = kline[kline['trade_date'] == rd][['stock_code', 'close']].rename(columns={'close': 'c0'})
        p1 = kline[kline['trade_date'] == rd_next][['stock_code', 'close']].rename(columns={'close': 'c1'})
        m = fv.merge(p0, on='stock_code').merge(p1, on='stock_code')
        m['fwd_ret'] = m['c1'] / m['c0'] - 1

        if len(m) < 100:
            continue

        ic = m['factor'].corr(m['fwd_ret'])
        if not np.isnan(ic):
            ic_series.append({'date': str(rd.date()), 'ic': round(ic, 6)})

        m['group'] = pd.qcut(m['factor'], N_GROUPS, labels=False, duplicates='drop') + 1
        for g in range(1, N_GROUPS + 1):
            gr = m[m['group'] == g]['fwd_ret'].mean()
            group_rets[g].append({'date': str(rd.date()), 'return': gr if not np.isnan(gr) else 0})

    # Compute stats
    nav_data = {}
    g_sharpes, g_anns, g_mdds = [], [], []
    for g in range(1, N_GROUPS + 1):
        nav = 1.0
        navs = []
        rets = [r['return'] for r in group_rets[g]]
        for r in group_rets[g]:
            nav *= (1 + r['return'] - COST)
            navs.append({'date': r['date'], 'nav': round(nav, 6)})
        nav_data[f'G{g}'] = navs

        ann = np.mean(rets) * 52 if rets else 0
        sharpe = (np.mean(rets) / np.std(rets) * np.sqrt(52)) if rets and np.std(rets) > 0 else 0
        peak = 1
        mdd = 0
        for n in navs:
            peak = max(peak, n['nav'])
            dd = n['nav'] / peak - 1
            mdd = min(mdd, dd)
        g_sharpes.append(round(sharpe, 4))
        g_anns.append(round(ann, 4))
        g_mdds.append(round(mdd, 4))

    # LS
    ls_rets = []
    for i in range(min(len(group_rets[1]), len(group_rets[N_GROUPS]))):
        ls_rets.append(group_rets[N_GROUPS][i]['return'] - group_rets[1][i]['return'])
    ls_sharpe = (np.mean(ls_rets) / np.std(ls_rets) * np.sqrt(52)) if ls_rets and np.std(ls_rets) > 0 else 0

    ics = [x['ic'] for x in ic_series]
    ic_mean = np.mean(ics) if ics else 0
    ic_std = np.std(ics) if ics else 1
    ic_t = ic_mean / (ic_std / np.sqrt(len(ics))) if ics and ic_std > 0 else 0
    ic_pos = sum(1 for x in ics if x > 0) / len(ics) if ics else 0

    # Monotonicity
    mono = sum(1 for i in range(len(g_sharpes)-1) if g_sharpes[i] < g_sharpes[i+1]) / (len(g_sharpes)-1) if len(g_sharpes)>1 else 0

    metrics = {
        'ic_mean': round(ic_mean, 6),
        'ic_std': round(ic_std, 6),
        'ic_t': round(ic_t, 2),
        'ic_positive_ratio': round(ic_pos, 4),
        'rank_ic': round(ic_mean, 6),
        'rank_ic_std': round(ic_std, 6),
        'ir': round(ic_mean / ic_std if ic_std > 0 else 0, 4),
        'ic_count': len(ics),
        'long_short_sharpe': round(ls_sharpe, 4),
        'monotonicity': round(mono, 2),
        'group_sharpe': g_sharpes,
        'group_returns_annualized': g_anns,
        'group_mdd': g_mdds,
        'n_periods': len(ics)
    }

    print("\n=== ROE Quality v1 Results ===")
    print(f"IC: {ic_mean:.4f} (t={ic_t:.2f}), positive: {ic_pos:.1%}")
    print(f"LS Sharpe: {ls_sharpe:.2f}")
    print(f"Monotonicity: {mono:.2f}")
    print(f"G5 Sharpe: {g_sharpes[4]:.2f}")
    print(f"Group Sharpe: {g_sharpes}")
    print(f"Group Ann Ret: {g_anns}")
    print(f"Periods: {len(ics)}")

    # Save
    with open(os.path.join(OUTPUT_DIR, 'cumulative_returns.json'), 'w') as f:
        json.dump(nav_data, f)
    with open(os.path.join(OUTPUT_DIR, 'ic_series.json'), 'w') as f:
        json.dump(ic_series, f)
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {OUTPUT_DIR}/")
    return metrics


if __name__ == '__main__':
    main()
