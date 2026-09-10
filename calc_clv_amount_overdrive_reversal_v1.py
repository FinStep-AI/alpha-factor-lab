#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
FACTOR_ID = 'clv_amount_overdrive_reversal_v1'
OUT = BASE / f'data/factor_{FACTOR_ID}.csv'


def robust_z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad < 1e-12:
        std = s.std()
        if pd.isna(std) or std < 1e-12:
            return pd.Series(np.nan, index=s.index)
        return ((s - s.mean()) / std).clip(-5, 5)
    return ((s - med) / (1.4826 * mad)).clip(-5, 5)


def neutralize_cs(df: pd.DataFrame, raw_col: str, cap_col: str) -> pd.Series:
    vals = df[raw_col].astype(float).values
    ctrl = df[cap_col].astype(float).values
    mask = np.isfinite(vals) & np.isfinite(ctrl)
    out = np.full(len(df), np.nan)
    if mask.sum() < 30:
        return pd.Series(out, index=df.index)
    y = vals[mask]
    x = ctrl[mask]
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    if mad < 1e-12:
        return pd.Series(out, index=df.index)
    resid = np.clip(resid, med - 5.2 * mad, med + 5.2 * mad)
    std = resid.std()
    if std < 1e-12:
        return pd.Series(out, index=df.index)
    out[np.where(mask)[0]] = (resid - np.median(resid)) / std
    return pd.Series(out, index=df.index)


df = pd.read_csv(BASE / 'data/csi1000_kline_raw.csv', usecols=['date', 'stock_code', 'open', 'close', 'high', 'low', 'amount', 'amplitude', 'turnover'])
df['date'] = pd.to_datetime(df['date'])
df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
for c in ['open', 'close', 'high', 'low', 'amount', 'amplitude', 'turnover']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.sort_values(['stock_code', 'date']).drop_duplicates(['date', 'stock_code'])

# 收盘在日内区间中的位置 [-1, 1]
span = (df['high'] - df['low']).replace(0, np.nan)
clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / span
body = (df['close'] - df['open']) / df['open'].replace(0, np.nan)
body = body.clip(-0.12, 0.12)

# 量能/波动异常：越大越容易是拥挤/透支
log_amount = np.log(df['amount'].clip(lower=1))

g = df.groupby('stock_code')
df['amt_z20'] = g['amount'].transform(lambda s: robust_z(np.log(s.rolling(20, min_periods=10).mean().clip(lower=1))))
df['amp_z20'] = g['amplitude'].transform(lambda s: robust_z(s.rolling(20, min_periods=10).mean()))
df['turn_z20'] = g['turnover'].transform(lambda s: robust_z(s.rolling(20, min_periods=10).mean()))
df['clv_ma5'] = g.apply(lambda x: clv.loc[x.index].rolling(5, min_periods=3).mean()).reset_index(level=0, drop=True)
df['body_ma3'] = g.apply(lambda x: body.loc[x.index].rolling(3, min_periods=2).mean()).reset_index(level=0, drop=True)

overdrive = 0.45 * df['amt_z20'] + 0.30 * df['amp_z20'] + 0.25 * df['turn_z20']
# 高CLV本来偏强，但若伴随过热 overdrive，则做反转；低过热时保留顺势强度
raw = (0.70 * df['clv_ma5'] + 0.30 * df['body_ma3']) * (1.0 - overdrive)
df['raw_factor'] = raw.replace([np.inf, -np.inf], np.nan)

# 市值代理中性化
mktcap_proxy = df['close'].clip(lower=0.01) * df['amount'].clip(lower=1) / (df['turnover'].replace(0, np.nan) + 1e-6)
df['log_mktcap'] = np.log(mktcap_proxy.clip(lower=1))
df = df.dropna(subset=['raw_factor', 'log_mktcap'])

outs = []
for date, grp in df.groupby('date'):
    nz = neutralize_cs(grp, 'raw_factor', 'log_mktcap')
    good = nz.notna()
    if good.sum() < 30:
        continue
    sub = grp.loc[good, ['date', 'stock_code']].copy()
    sub['factor'] = nz.loc[good].values
    outs.append(sub)

result = pd.concat(outs, ignore_index=True)
result['date'] = pd.to_datetime(result['date']).dt.strftime('%Y-%m-%d')
result.to_csv(OUT, index=False, float_format='%.6f')
print(f'saved {OUT} rows={len(result)} dates={result.date.min()}~{result.date.max()} stocks={result.stock_code.nunique()}')
print(result.head().to_string())
