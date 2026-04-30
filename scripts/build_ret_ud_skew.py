# -*- coding: utf-8 -*-
"""
因子构建：涨跌分布不对称度 (Return Up-Down Skewness)
ID: ret_ud_skew_v1
概念：过去20日累计对数收益率中，上涨日的相对权重。
     越高 = 多数上涨日放量/幅度大 ≠ 均匀分布 ≠ 真实信号区分度。
     中性化后进行截面排序。
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJ = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')

kline = pd.read_csv(PROJ / 'data' / 'csi1000_kline_raw.csv', parse_dates=['date'])
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
kline['pct_change'] = pd.to_numeric(kline['pct_change'], errors='coerce')

W = 20

def rolling_ret_skew(grp):
    vals = grp['pct_change'].values.astype(float)
    n = len(vals)
    out = np.full(n, np.nan)
    for i in range(W - 1, n):
        w = vals[i - W + 1: i + 1]
        mu = w.mean()
        if np.abs(mu) < 1e-10:
            continue
        out[i] = (np.where(w > 0, w - mu, 0).sum() - np.where(w < 0, w - mu, 0).sum()) / \
                 (np.abs(w - mu).sum() + 1e-10)
    return pd.Series(out, index=grp.index)

print("计算涨跌分布不对称度...")
kline['ret_ud_skew'] = kline.groupby('stock_code', group_keys=False).apply(rolling_ret_skew)
kline['amount_20d'] = kline.groupby('stock_code')['amount'].transform(lambda x: x.rolling(20, min_periods=10).mean())

print("成交额中性化...")
kline['neutral_factor'] = np.nan
for date, grp in kline.groupby('date'):
    idx = grp.index
    f  = grp['ret_ud_skew'].values.astype(float)
    na = grp['amount_20d'].values.astype(float)
    mask = np.isfinite(f) & np.isfinite(na) & (na > 0)
    if mask.sum() < 30:
        continue
    f_m, n_m = f[mask], na[mask]
    med_f, mad_f = np.median(f_m), np.median(np.abs(f_m - np.median(f_m)))
    if mad_f > 0:
        f_m = np.clip(f_m, med_f - 5.5 * 1.4826 * mad_f, med_f + 5.5 * 1.4826 * mad_f)
    X = np.column_stack([np.ones(len(n_m)), np.log(n_m + 1)])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, f_m, rcond=None)
        resid = f_m - X @ beta
        std = resid.std()
        if std > 0:
            resid = (resid - resid.mean()) / std
        kline.loc[idx[mask], 'neutral_factor'] = resid
    except Exception:
        pass

out = kline[['date', 'stock_code', 'neutral_factor']].dropna(subset=['neutral_factor'])
outPath = PROJ / 'data' / 'factor_ret_ud_skew_v1.csv'
out[['date', 'stock_code', 'neutral_factor']].rename(columns={'neutral_factor': 'factor_value'}).to_csv(outPath, index=False)
print(f"因子CSV → {outPath}  ({len(out)} 行)")
