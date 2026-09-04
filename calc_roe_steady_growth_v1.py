#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
fund_path = BASE / 'data/csi1000_fundamental_cache.csv'
kline_path = BASE / 'data/csi1000_kline_raw.csv'
out_path = BASE / 'data/factor_roe_steady_growth_v1.csv'


def winsorize_series(s, n=3.0):
    x = s.astype(float).copy()
    med = x.median()
    mad = (x - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return x
    lo = med - 1.4826 * n * mad
    hi = med + 1.4826 * n * mad
    return x.clip(lo, hi)


def zscore(s):
    s = s.astype(float)
    std = s.std()
    if pd.isna(std) or std < 1e-12:
        return s * np.nan
    return (s - s.mean()) / std


def neutralize(df_day):
    x = df_day[['factor_raw', 'log_mktcap']].dropna().copy()
    if len(x) < 20:
        return pd.Series(index=df_day.index, dtype=float)
    y = x['factor_raw'].values
    X = np.column_stack([np.ones(len(x)), x['log_mktcap'].values])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    resid = pd.Series(resid, index=x.index)
    resid = winsorize_series(resid, 3.0)
    resid = zscore(resid)
    out = pd.Series(index=df_day.index, dtype=float)
    out.loc[resid.index] = resid
    return out


fund = pd.read_csv(fund_path)
fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
fund['report_date'] = pd.to_datetime(fund['report_date'])
fund = fund.sort_values(['stock_code', 'report_date'])

# 仅使用当前可得字段做“稳健增长”代理：高ROE水平 + ROE同比改善 + BPS同比扩张约束
fund['roe_l1'] = fund.groupby('stock_code')['roe'].shift(1)
fund['roe_l4'] = fund.groupby('stock_code')['roe'].shift(4)
fund['bps_l4'] = fund.groupby('stock_code')['bps'].shift(4)
fund['roe_ma4'] = fund.groupby('stock_code')['roe'].transform(lambda s: s.rolling(4, min_periods=3).mean())
fund['roe_std4'] = fund.groupby('stock_code')['roe'].transform(lambda s: s.rolling(4, min_periods=3).std())
fund['roe_yoy'] = fund['roe'] - fund['roe_l4']
fund['roe_qoq'] = fund['roe'] - fund['roe_l1']
fund['bps_yoy'] = fund['bps'] / fund['bps_l4'] - 1.0

# 稳健成长：高水平、同比改善、环比不塌、低波动、BPS温和扩张
fund['factor_raw'] = (
    0.45 * np.tanh(fund['roe_ma4'] / 8.0)
    + 0.30 * np.tanh(fund['roe_yoy'] / 5.0)
    + 0.15 * np.tanh(fund['roe_qoq'] / 3.0)
    - 0.20 * np.tanh(fund['roe_std4'] / 3.0)
    - 0.15 * np.tanh(fund['bps_yoy'] / 0.30)
)

# 财报滞后45天映射到交易日
fund['effective_date'] = fund['report_date'] + pd.Timedelta(days=45)
fund_daily = fund[['stock_code', 'effective_date', 'factor_raw']].rename(columns={'effective_date': 'date'})

k = pd.read_csv(kline_path, usecols=['date', 'stock_code', 'close', 'amount'])
k['date'] = pd.to_datetime(k['date'])
k['stock_code'] = k['stock_code'].astype(str).str.zfill(6)
k = k.sort_values(['stock_code', 'date'])
k['mktcap_proxy'] = k['close'] * k['amount'].rolling(20, min_periods=5).mean().reset_index(drop=True)
# 上面这一行只是占位，下面重算分组 rolling，避免串股
k['amt20'] = k.groupby('stock_code')['amount'].transform(lambda s: s.rolling(20, min_periods=5).mean())
k['mktcap_proxy'] = k['close'] * k['amt20']
k['log_mktcap'] = np.log(k['mktcap_proxy'].clip(lower=1e-6))

# merge_asof 按股票逐只映射最近已生效财报
parts = []
for stock, kd in k.groupby('stock_code', sort=False):
    fd = fund_daily[fund_daily['stock_code'] == stock].sort_values('date')
    kd = kd.sort_values('date')
    if fd.empty:
        kd['factor_raw'] = np.nan
    else:
        kd = pd.merge_asof(kd, fd[['date', 'factor_raw']], on='date', direction='backward')
    parts.append(kd[['date', 'stock_code', 'factor_raw', 'log_mktcap']])
merged = pd.concat(parts, ignore_index=True)

merged['factor'] = merged.groupby('date', group_keys=False).apply(neutralize)
out = merged[['date', 'stock_code', 'factor']].dropna().copy()
out['date'] = out['date'].dt.strftime('%Y-%m-%d')
out.to_csv(out_path, index=False)
print(f'Saved {len(out):,} rows to {out_path}')
print(out.head().to_string())
