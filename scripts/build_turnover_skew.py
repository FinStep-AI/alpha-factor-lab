# -*- coding: utf-8 -*-
"""快速因子：Turnover截面离散度（Turnover Spread）vs同类均值"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJ = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')
kline = pd.read_csv(PROJ / 'data' / 'csi1000_kline_raw.csv', parse_dates=['date'])
kline = kline.sort_values(['stock_code','date']).reset_index(drop=True)
kline['pct_change'] = pd.to_numeric(kline['pct_change'], errors='coerce')

WINDOW = 20

def rolling_turnover_spread(grp):
    """过去N日换手率分布离散度（截面）— 这里在每个日期各股间计算"""
    pass

# 改为截面计算：每日各股票过去20日换手率的 skewness
print("构建turnover_skew因子...")
kline['turnover_20d_skew'] = kline.groupby('stock_code')['turnover'].transform(
    lambda x: x.rolling(20, min_periods=10).skew()
)
kline['amount_20d'] = kline.groupby('stock_code')['amount'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

kline['neutral_factor'] = np.nan
for date, grp in kline.groupby('date'):
    idx = grp.index
    f  = grp['turnover_20d_skew'].values.astype(float)
    na = grp['amount_20d'].values.astype(float)
    mask = np.isfinite(f) & np.isfinite(na) & (na > 0)
    if mask.sum() < 30:
        continue
    f_m, n_m = f[mask], na[mask]
    med_f, mad_f = np.median(f_m), np.median(np.abs(f_m - np.median(f_m)))
    if mad_f > 0:
        f_m = np.clip(f_m, med_f - 5.5*1.4826*mad_f, med_f + 5.5*1.4826*mad_f)
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
Path(PROJ / 'data' / 'factor_turnover_skew_v1.csv').write_text('')
out[['date', 'stock_code', 'neutral_factor']].rename(columns={'neutral_factor':'factor_value'}).to_csv(
    PROJ / 'data' / 'factor_turnover_skew_v1.csv', index=False)
print(f"Done: {len(out)} rows, latest {out['date'].max()}")
