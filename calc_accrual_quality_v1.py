#!/usr/bin/env python3
"""
因子: Accrual Quality Score v1 (AQS)
ID:   accrual_quality_v1
论文: Sloan (1996) JAR — Accrual Anomaly
      Dechow & Dichev (2002) — Cash Flow–Accruals
      Novy-Marx (2013) — "The Other Side of Value" (Profitability / Quality)
      Huang, I. et al — Composite Accrual / Profitability Quality

本仓库基本面数据无应计绝对值，只有 ROE 与 BPS。
方向: 以两段季度截面 ROE 作 converges + ROE 残差(截面 Z-score)，
    作为「盈利质量 / 低应计操纵」的代理变量(signal)，覆盖从财报披露决定后起诈
"""

import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

# ── 载入 ────────────────────────────────────────────────────────────────────────
fund  = pd.read_csv(os.path.join(DATA, 'csi1000_fundamental_cache.csv'))
fund['report_date'] = pd.to_datetime(fund['report_date'])
fund['stock_code']  = fund['stock_code'].astype(str).str.zfill(6)

kline = pd.read_csv(os.path.join(DATA, 'csi1000_kline_raw.csv'),
                    usecols=['date', 'stock_code', 'amount'])
kline['date']       = pd.to_datetime(kline['date'])
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)

trade_dates = sorted(kline['date'].unique())

# ── 报告 → 可用首个 kline 日 映射 ──────────────────────────────────────────────
# 若报告日当天已在 kline 中则用当天；否则顺移到最近下一个交易日（正向推）。
rpt_dates = sorted(fund['report_date'].unique())
rpt2k = {}
td_idx  = pd.DatetimeIndex(trade_dates)
for rd in rpt_dates:
    avail = td_idx[td_idx >= rd]
    rpt2k[rd] = avail[0] if len(avail) else rd

# 检查渲染后是否唯一
mapped = sorted(set(rpt2k.values()))
print(f"[INFO] unique kline dates mapped: {len(mapped)}", file=sys.stderr)
for rd, kd in sorted(rpt2k.items()):
    n_k = int((td_idx >= kd).sum())
    print(f"  report={rd.date()}  kline={kd.date()}  days_forward={n_k}", file=sys.stderr)

fund['kline_date'] = fund['report_date'].map(rpt2k)
fund = fund.sort_values(['stock_code', 'report_date'])\
           .drop_duplicates(['stock_code', 'report_date'], keep='first')

# ── 每期截面: roe_z + roe_yoy_z → composite Q ──────────────────────────────────
kdates = sorted(fund['kline_date'].unique())
print(f"[INFO] kline cross-section dates: {[str(d.date()) for d in kdates]}", file=sys.stderr)

prev_map = {}          # (stock, cross_date) → prev roe
records = []           # DataFrame rows, later merged

for i, cd in enumerate(kdates):
    sub = (fund[fund['kline_date'] == cd]
           .dropna(subset=['roe'])[['stock_code', 'roe']].copy())
    sub['stock_code'] = sub['stock_code'].astype(str).str.zfill(6)
    sub = sub[sub['roe'].between(-80, 250)]
    if len(sub) < 200:
        continue

    # 1) roe_z (截面无效包 z-score)
    mu = sub['roe'].mean();  sig = sub['roe'].std(ddof=0)
    if sig < 1e-8:
        continue
    sub['roe_z'] = (sub['roe'] - mu) / sig

    # 2) roe_yoy_z: 同比本期 - 上年同期，同只股票
    if i >= 4:
        prev_cd = kdates[i - 4]
        prev = (fund[fund['kline_date'] == prev_cd]
                .dropna(subset=['roe'])[['stock_code', 'roe']]
                .drop_duplicates('stock_code', keep='last')
                .rename(columns={'roe': 'roe_prev'})
                .set_index('stock_code')['roe_prev'])
        sub['stock_code'] = sub['stock_code'].astype(str).str.zfill(6)
        sub['roe_prev'] = sub['stock_code'].map(prev)
        sub = sub.dropna(subset=['roe_prev'])
        sub['roe_yoy'] = sub['roe'] - sub['roe_prev']
        yoy_mu = sub['roe_yoy'].mean()
        yoy_sig = sub['roe_yoy'].std(ddof=0)
        if yoy_sig < 1e-8:
            sub['yoy_z'] = 0.0
        else:
            sub['yoy_z'] = (sub['roe_yoy'] - yoy_mu) / yoy_sig
    else:
        sub['yoy_z'] = 0.0

    # 3) composite Q signal = 0.55×截面上 + 0.45×同比变动
    sub['raw_aqs'] = 0.55 * sub['roe_z'] + 0.45 * sub['yoy_z']

    for _, row in sub.iterrows():
        records.append((str(cd.date()),
                        str(int(row['stock_code'])).zfill(6),
                        float(row['raw_aqs'])))

raw = pd.DataFrame(records, columns=['date', 'stock_code', 'raw_aqs'])
print(f"[INFO] raw pre-neutralization rows: {len(raw)}", file=sys.stderr)

# ── 截面中性化: 成交额OLS + MAD+z-score ───────────────────────────────────────
amt_day = kline.groupby('date')['amount'].median()
amt_day.index = amt_day.index.strftime('%Y-%m-%d')

out_rows = []
for dt_str, grp in raw.groupby('date', sort=False):
    amt_med = amt_day.get(dt_str, np.nan)
    if np.isnan(amt_med):
        continue
    day_amt = kline.loc[kline['date'].dt.strftime('%Y-%m-%d') == dt_str,
                        ['stock_code', 'amount']].copy()
    day_amt['stock_code'] = day_amt['stock_code'].astype(str).str.zfill(6)
    amt_map = dict(zip(day_amt['stock_code'], np.log(day_amt['amount'] + 1)))

    g = grp.copy()
    g['stock_code'] = g['stock_code'].astype(str).str.zfill(6)
    g['log_amt'] = g['stock_code'].map(amt_map)
    g = g.dropna(subset=['raw_aqs', 'log_amt'])
    if len(g) < 100:
        continue

    X = np.column_stack([np.ones(len(g)), g['log_amt'].values])
    y = g['raw_aqs'].values
    try:
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X @ b
    except Exception:
        r = y

    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    if mad < 1e-10:
        continue
    r_clip = np.clip(r, med - 5.2*mad, med + 5.2*mad)
    std = float(np.std(r_clip))
    if std < 1e-10:
        continue
    z = (r_clip - np.median(r_clip)) / std
    g = g.copy()
    g['factor_value'] = z
    out_rows.append(g[['date', 'stock_code', 'factor_value']])

out = pd.concat(out_rows, ignore_index=True)
out.to_csv(os.path.join(DATA, 'factor_accrual_quality_v1.csv'), index=False)
print(f"[OK] wrote {len(out)} rows to data/factor_accrual_quality_v1.csv", file=sys.stderr)
print(out['date'].value_counts().sort_index().to_string(), file=sys.stderr)
print(out['factor_value'].describe().to_string(), file=sys.stderr)
