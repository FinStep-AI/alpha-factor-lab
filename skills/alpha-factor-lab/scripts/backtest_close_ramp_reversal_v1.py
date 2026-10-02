#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
close_ramp_reversal_v1
======================
尾盘异动/收盘拉抬反转因子：
- 用日内收盘位置(close在high-low区间中的相对位置) × 日内收益(open→close)刻画“临近收盘被拉到高位”的强度
- 再用隔夜跳空幅度做轻微惩罚，尽量剔除纯信息型高开，保留更像尾盘资金推动的成分
- 过去20日均值后取反：近期越频繁出现“收在高位且日内被拉升”，越像尾盘资金集中推动，后续更容易均值回归

回测设置：
- 中证1000日线
- 20日窗口
- OLS市值中性化（log_mktcap = log(amount/turnover)）
- 5% winsorize
- 5组分层
- 5日前瞻收益
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
FACTOR_ID = 'close_ramp_reversal_v1'

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
    prev_close = df.groupby('stock_code')['close'].shift(1)

    intraday_pos = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    intraday_ret = (df['close'] - df['open']) / df['open'].replace(0, np.nan)
    gap_ret = (df['open'] - prev_close) / prev_close

    ramp_strength = intraday_pos * intraday_ret.clip(-0.2, 0.2)
    gap_penalty = gap_ret.abs().clip(0, 0.15)
    signed_push = ramp_strength * (1 - 0.5 * gap_penalty / 0.15)

    raw = -signed_push.groupby(df['stock_code']).transform(
        lambda x: x.rolling(WINDOW, min_periods=15).mean()
    )

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

print('[4] 回测…')
ic_p = compute_ic_dynamic(fa, ra, FORWARD_DAYS, 'pearson')
ic_s = compute_ic_dynamic(fa, ra, FORWARD_DAYS, 'spearman')
gr, turns, hi = compute_group_returns(fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
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
    'ic_mean': float(me.get('ic_mean', 0) or 0),
    'ic_ir': float(me.get('ir', 0) or 0),
    't_stat_nw': float(nw.get('t_stat', 0) or 0),
    'p_value_nw': float(nw.get('p_value', 1) or 1),
    'long_short_sharpe': float(me.get('long_short_sharpe', 0) or 0),
    'long_short_mdd': float(me.get('long_short_mdd', 0) or 0),
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
print('group_returns_annualized:', report['group_returns_annualized'])
print('valid:', report['valid'])
print('=' * 60)
