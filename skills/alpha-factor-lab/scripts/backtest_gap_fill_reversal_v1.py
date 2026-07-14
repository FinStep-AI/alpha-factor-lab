#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gap_fill_reversal_v1
====================
跳空回补反转因子：
- 度量过去20日“跳空后日内被回补”的强度
- 上跳空后若经常被日内回补（open 高开, close 回落），说明隔夜情绪兑现不足，后续偏弱
- 下跳空后若经常被日内回补（open 低开, close 回升），说明恐慌被快速承接，后续偏强
- 因此定义 signed_fill = sign(gap) * fill_ratio * |gap|，并自动选择有效方向
"""

import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

WINDOW = 20
FORWARD_DAYS = 5
REBALANCE_FREQ = 5
N_GROUPS = 5
COST = 0.003
WINSORIZE_PCT = 0.05
DATA_CUTOFF = '2026-07-13'
FACTOR_ID = 'gap_fill_reversal_v1'

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / 'data' / 'csi1000_kline_raw.csv'
OUTPUT_DIR = BASE_DIR / 'output' / FACTOR_ID
SCRIPTS_DIR = BASE_DIR / 'skills' / 'alpha-factor-lab' / 'scripts'
sys.path.insert(0, str(SCRIPTS_DIR))
from factor_backtest import compute_group_returns, compute_ic_dynamic, compute_metrics, save_backtest_data, newey_west_t_stat


def winsorize_cross_section(s: pd.Series, pct: float = 0.05) -> pd.Series:
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)


def zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if pd.isna(std) or std < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def neutralize_one_day(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    x = g['log_mktcap'].values
    y = g['raw_factor'].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        g['factor_neu'] = np.nan
        return g
    X = np.column_stack([np.ones(mask.sum()), x[mask]])
    beta, _, _, _ = np.linalg.lstsq(X, y[mask], rcond=None)
    resid = y[mask] - X @ beta
    out = pd.Series(np.nan, index=g.index)
    out.loc[g.index[mask]] = resid
    g['factor_neu'] = out
    return g


def build_factor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['stock_code', 'date']).copy()
    df['prev_close'] = df.groupby('stock_code')['close'].shift(1)
    df['gap_ret'] = (df['open'] - df['prev_close']) / df['prev_close']
    df['fill_ratio'] = np.where(
        df['gap_ret'] > 0,
        (df['open'] - df['close']) / (df['open'] - df['prev_close'] + 1e-8),
        np.where(
            df['gap_ret'] < 0,
            (df['close'] - df['open']) / (df['prev_close'] - df['open'] + 1e-8),
            np.nan,
        ),
    )
    signed_fill = np.sign(df['gap_ret']) * df['fill_ratio'] * np.abs(df['gap_ret'])
    raw = signed_fill.groupby(df['stock_code']).transform(lambda x: x.rolling(WINDOW, min_periods=15).mean())

    turn_nz = df['turnover'].replace(0, np.nan)
    log_mktcap = np.log((df['amount'] / turn_nz).replace(0, np.nan))

    out = df[['date', 'stock_code']].copy()
    out['raw_factor'] = raw
    out['log_mktcap'] = log_mktcap
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=['raw_factor', 'log_mktcap'])
    return out


print(f'[1] 构建 {FACTOR_ID} 因子…')
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] <= DATA_CUTOFF].copy()
print(f'   rows={len(df)} stocks={df.stock_code.nunique()} range={df.date.min().date()}~{df.date.max().date()}')

raw = build_factor(df)
print(f'[2] 原始因子完成 raw_rows={len(raw)}')

raw['raw_factor'] = raw.groupby('date')['raw_factor'].transform(lambda s: winsorize_cross_section(s, WINSORIZE_PCT))
raw = raw.groupby('date', group_keys=False).apply(neutralize_one_day)
raw = raw.dropna(subset=['factor_neu']).copy()
raw['factor_value'] = raw.groupby('date')['factor_neu'].transform(zscore)
raw = raw[['date', 'stock_code', 'factor_value']].dropna()
print(f'[3] 中性化完成 panel_rows={len(raw)} dates={raw.date.nunique()}')

factor_mat = raw.pivot_table(index='date', columns='stock_code', values='factor_value').sort_index()
close_p = df.pivot_table(index='date', columns='stock_code', values='close').sort_index()
ret = close_p.pct_change()

common_dates = factor_mat.index.intersection(ret.index)
common_stocks = factor_mat.columns.intersection(ret.columns)
fa = factor_mat.loc[common_dates, common_stocks]
ra = ret.loc[common_dates, common_stocks]

print('[4] 方向探索…')
ic_p = compute_ic_dynamic(fa, ra, FORWARD_DAYS, 'pearson')
gr_p, turns_p, hi_p = compute_group_returns(fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
me_p = compute_metrics(gr_p, ic_p, ic_p, turns_p, N_GROUPS, holdings_info=hi_p)

ic_n = compute_ic_dynamic(-fa, ra, FORWARD_DAYS, 'pearson')
gr_n, turns_n, hi_n = compute_group_returns(-fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
me_n = compute_metrics(gr_n, ic_n, ic_n, turns_n, N_GROUPS, holdings_info=hi_n)

if float(me_n.get('long_short_sharpe', 0) or 0) > float(me_p.get('long_short_sharpe', 0) or 0):
    fa_use = -fa
    direction = -1
    direction_desc = '反向（高跳空回补强度=后续偏弱，做空高值 / 做多低值）'
else:
    fa_use = fa
    direction = 1
    direction_desc = '正向（高跳空回补强度=后续偏强，做多高值）'
print(f'   选择方向: {direction_desc}')

print('[5] 最终回测…')
ic_p = compute_ic_dynamic(fa_use, ra, FORWARD_DAYS, 'pearson')
ic_s = compute_ic_dynamic(fa_use, ra, FORWARD_DAYS, 'spearman')
gr, turns, hi = compute_group_returns(fa_use, ra, N_GROUPS, REBALANCE_FREQ, COST)
me = compute_metrics(gr, ic_p, ic_s, turns, N_GROUPS, holdings_info=hi)
nw = newey_west_t_stat(ic_p)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_backtest_data(gr, ic_p, ic_s, str(OUTPUT_DIR))

report = {
    'factor_id': FACTOR_ID,
    'period': f'{common_dates.min().date()} ~ {common_dates.max().date()}',
    'n_stocks': int(len(common_stocks)),
    'window': WINDOW,
    'forward_days': FORWARD_DAYS,
    'rebalance_freq': REBALANCE_FREQ,
    'n_groups': N_GROUPS,
    'cost': COST,
    'direction': direction,
    'direction_desc': direction_desc,
    'ic_mean': float(me.get('ic_mean', 0) or 0),
    'ic_ir': float(me.get('ir', 0) or 0),
    't_stat_nw': float(nw.get('t_stat', 0) or 0),
    'p_value_nw': float(nw.get('p_value', 1) or 1),
    'long_short_sharpe': float(me.get('long_short_sharpe', 0) or 0),
    'long_short_mdd': float(me.get('long_short_mdd', 0) or 0),
    'long_short_ann_return': float(me.get('long_short_ann_return', 0) or 0),
    'long_short_cumulative_return': float(me.get('long_short_cumulative_return', 0) or 0),
    'monotonicity': float(me.get('monotonicity', 0) or 0),
    'group_returns_annualized': me.get('group_returns_annualized', []),
    'turnover_mean': float(me.get('turnover_mean', 0) or 0),
}
report['valid'] = abs(report['ic_mean']) > 0.015 and abs(report['t_stat_nw']) > 2 and abs(report['long_short_sharpe']) > 0.5
(OUTPUT_DIR / 'backtest_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

print('=' * 60)
print(FACTOR_ID)
for k in ['ic_mean','ic_ir','t_stat_nw','long_short_sharpe','monotonicity','turnover_mean']:
    print(f'{k}: {report[k]}')
print('direction_desc:', report['direction_desc'])
print('group_returns_annualized:', report['group_returns_annualized'])
print('valid:', report['valid'])
print('=' * 60)
