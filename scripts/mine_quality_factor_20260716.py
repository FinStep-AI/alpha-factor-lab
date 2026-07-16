#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

WD = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
FACTOR_ID = 'roe_bps_delta_stability_v1'
FACTOR_CSV = WD / 'data' / f'factor_{FACTOR_ID}.csv'
OUTPUT_DIR = WD / 'output' / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / 'report.json'
RESEARCH_PATH = WD / 'factor-research.json'
FACTORS_PATH = WD / 'factors.json'


def winsorize_by_date(df, col, lower=0.01, upper=0.99):
    def _clip(s):
        q1 = s.quantile(lower)
        q2 = s.quantile(upper)
        return s.clip(q1, q2)
    return df.groupby('date')[col].transform(_clip)


def zscore_by_date(df, col):
    def _z(s):
        std = s.std(ddof=0)
        if std is None or np.isnan(std) or std == 0:
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std
    return df.groupby('date')[col].transform(_z)


def build_factor():
    fund = pd.read_csv(WD / 'data' / 'csi1000_fundamental_cache.csv')
    kline = pd.read_csv(WD / 'data' / 'csi1000_kline_raw.csv', usecols=['date', 'stock_code', 'amount'])

    fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
    kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
    fund['report_date'] = pd.to_datetime(fund['report_date'])
    kline['date'] = pd.to_datetime(kline['date'])

    fund = fund.sort_values(['stock_code', 'report_date']).copy()
    g = fund.groupby('stock_code', group_keys=False)

    fund['roe_delta_2q'] = g['roe'].transform(lambda s: s.diff(2))
    fund['bps_growth_2q'] = g['bps'].transform(lambda s: s.pct_change(2, fill_method=None))
    fund['roe_stability'] = 1.0 / (g['roe'].transform(lambda s: s.rolling(4, min_periods=3).std()) + 1e-6)

    # Quality: 盈利改善 + 净资产扩张 + ROE稳定性
    fund['raw_factor'] = (
        0.45 * fund['roe_delta_2q'] +
        0.35 * fund['bps_growth_2q'] +
        0.20 * fund['roe_stability']
    )

    fund['available_date'] = fund['report_date'] + pd.Timedelta(days=45)
    daily_base = kline[['date', 'stock_code', 'amount']].drop_duplicates(['date', 'stock_code']).sort_values(['stock_code', 'date'])
    daily_base['log_amount'] = np.log(daily_base['amount'].clip(lower=1))

    merged_parts = []
    right_base = fund[['stock_code', 'available_date', 'raw_factor']].copy()
    for code, left_grp in daily_base.groupby('stock_code', sort=False):
        right_grp = right_base[right_base['stock_code'] == code].sort_values('available_date')
        if right_grp.empty:
            continue
        part = pd.merge_asof(
            left_grp.sort_values('date'),
            right_grp,
            left_on='date',
            right_on='available_date',
            direction='backward'
        )
        merged_parts.append(part)
    merged = pd.concat(merged_parts, ignore_index=True)
    merged = merged.rename(columns={'stock_code_x': 'stock_code'}).drop(columns=['stock_code_y'], errors='ignore')

    df = merged.dropna(subset=['raw_factor', 'log_amount']).copy()
    df['raw_factor'] = winsorize_by_date(df, 'raw_factor')
    df['raw_z'] = zscore_by_date(df, 'raw_factor')
    df['log_amount_z'] = zscore_by_date(df, 'log_amount')

    resid = []
    for d, grp in df.groupby('date'):
        x = grp['log_amount_z'].to_numpy()
        y = grp['raw_z'].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        r = np.full(len(grp), np.nan)
        if mask.sum() >= 20:
            X = np.column_stack([np.ones(mask.sum()), x[mask]])
            beta = np.linalg.lstsq(X, y[mask], rcond=None)[0]
            r[mask] = y[mask] - X @ beta
        resid.append(pd.Series(r, index=grp.index))
    df['factor'] = pd.concat(resid).sort_index()
    df['factor'] = winsorize_by_date(df, 'factor')
    df['factor'] = zscore_by_date(df, 'factor')

    out = df[['date', 'stock_code', 'factor']].dropna().copy()
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    out.to_csv(FACTOR_CSV, index=False)
    return out


def run_backtest():
    cmd = [
        'python3', 'skills/alpha-factor-lab/scripts/factor_backtest.py',
        '--factor', str(FACTOR_CSV),
        '--returns', 'data/csi1000_returns.csv',
        '--n-groups', '5',
        '--forward-days', '5',
        '--cost', '0.003',
        '--output-dir', str(OUTPUT_DIR),
        '--output-report', str(REPORT_PATH),
    ]
    res = subprocess.run(cmd, cwd=WD, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise SystemExit(res.returncode)
    return json.loads(REPORT_PATH.read_text())


def append_research(report):
    metrics = report['metrics']
    passed = (
        (metrics.get('ic_mean') or 0) > 0.02 and
        (metrics.get('ic_t_stat') or 0) > 2.0 and
        ((metrics.get('group_metrics') or {}).get('group_5', {}).get('sharpe') or 0) > 0.8 and
        (metrics.get('monotonicity') or 0) > 0.8
    )
    record = {
        'id': FACTOR_ID,
        'source_type': '自研',
        'source_title': 'ROE改善-BPS扩张稳定性因子',
        'source_url': '',
        'source_author': 'OpenClaw 因子猎人',
        'source_year': 2026,
        'source_journal': '',
        'original_metric': {
            'market': 'A股中小盘',
            'period': '内部研究',
            'ic_mean': None,
            'sharpe': None,
            'description': '结合ROE两季度改善、BPS两季度扩张、ROE稳定性的Quality/Growth混合因子。'
        },
        'our_metric': {
            'market': '中证1000',
            'period': report['period'],
            'ic_mean': round(metrics.get('ic_mean') or 0, 6),
            'ic_t': round(metrics.get('ic_t_stat') or 0, 6),
            'sharpe': round((((metrics.get('group_metrics') or {}).get('group_5', {}) or {}).get('sharpe_ratio') or 0), 6),
            'monotonicity': round(metrics.get('monotonicity') or 0, 6)
        },
        'diff_notes': '使用45天财报滞后映射到日频，并对log(amount)做横截面中性化。',
        'local_factor_id': FACTOR_ID,
        'conclusion': '入库' if passed else '淘汰',
        'date': '2026-07-16',
        'status': 'active' if passed else 'failed'
    }
    data = json.loads(RESEARCH_PATH.read_text())
    data.append(record)
    RESEARCH_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return passed, record


def main():
    out = build_factor()
    print(f'factor rows={len(out)} dates={out.date.nunique()} stocks={out.stock_code.nunique()}')
    report = run_backtest()
    passed, record = append_research(report)
    print(json.dumps({'passed': passed, 'record': record, 'report_metrics': report['metrics']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
