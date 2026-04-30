# -*- coding: utf-8 -*-
"""
因子构建：涨日成交量集中度 (Volume-Up Concentration)
ID: vol_up_concentration_v1
概念：过去20日上涨日(close>open)成交量之和 / 总成交量。
    高值 = 资金集中在上涨日推动 = 净买入信息流 → 做多（正向）
数据：日线OHLCV + turnover
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJ = Path('/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab')

# ── 加载数据 ────────────────────────────────────────────────────────
kline = pd.read_csv(PROJ / 'data' / 'csi1000_kline_raw.csv', parse_dates=['date'])
ret   = pd.read_csv(PROJ / 'data' / 'csi1000_returns.csv', parse_dates=['date'])

print(f"K线: {kline['date'].min()} ~ {kline['date'].max()}, 共{kline['stock_code'].nunique()}只")
print(f"收益: {ret['date'].min()} ~ {ret['date'].max()}")

# ── 排序 & 基础特征 ──────────────────────────────────────────────
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)
kline['pct_change'] = pd.to_numeric(kline['pct_change'], errors='coerce')

# 标记上涨日(印象日)
kline['is_up'] = (kline['close'] > kline['open']).astype(float)

# ── 20日滚动窗口 ──────────────────────────────────────────────────
WINDOW = 20

def rolling_up_vol_ratio(group):
    """上涨日成交量占比(rolling 20d)"""
    vol = group['volume'].values.astype(float)
    is_up = group['is_up'].values.astype(float)
    n = len(group)
    out = np.full(n, np.nan)
    for i in range(WINDOW - 1, n):
        w_vol = vol[i - WINDOW + 1: i + 1]
        w_is_up = is_up[i - WINDOW + 1: i + 1]
        total = w_vol.sum()
        if total > 0:
            out[i] = (w_vol * w_is_up).sum() / total
    return pd.Series(out, index=group.index)

print("计算滚动涨日成交量占比...")
concentration = kline.groupby('stock_code', group_keys=False).apply(rolling_up_vol_ratio)
kline['vol_up_conc'] = concentration.values

# ── 成交额中性化 ──────────────────────────────────────────────────
def ols_neutralize(factor_series, neutralizer, winsor=5.5):
    """截面OLS中性化：对某一截面，回归残差即为中性化因子"""
    from numpy.polynomial import polynomial as P
    out = pd.Series(np.nan, index=factor_series.index)
    for date, grp in factor_series.groupby(level='date') if isinstance(factor_series.index, pd.MultiIndex) else factor_series.groupby(kline['date']):
        idx = grp.index
        f = grp.values.astype(float)
        n_vals = neutralizer.loc[idx].values.astype(float)
        mask = np.isfinite(f) & np.isfinite(n_vals) & (n_vals > 0)
        if mask.sum() < 30:
            continue
        f_m, n_m = f[mask], n_vals[mask]
        # MAD winsorize
        med_f, mad_f = np.median(f_m), np.median(np.abs(f_m - np.median(f_m)))
        if mad_f > 0:
            f_m = np.clip(f_m, med_f - winsor * 1.4826 * mad_f, med_f + winsor * 1.4826 * mad_f)
        # OLS残差
        X = np.column_stack([np.ones(len(n_m)), np.log(n_m + 1)])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, f_m, rcond=None)
            resid = f_m - X @ beta
            # z-score标准化
            std = resid.std()
            if std > 0:
                resid = (resid - resid.mean()) / std
            out[idx[mask]] = resid
        except Exception:
            pass
    return out

print("成交额中性化...")
kline['log_amount'] = np.log(kline['amount'].fillna(1) + 1)
kline['amount_20d'] = kline.groupby('stock_code')['amount'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)
kline['neutral_factor'] = np.nan
for date, grp in kline.groupby('date'):
    idx = grp.index
    f  = grp['vol_up_conc'].values.astype(float)
    na = grp['amount_20d'].values.astype(float)
    mask = np.isfinite(f) & np.isfinite(na) & (na > 0)
    if mask.sum() < 30:
        continue
    f_m, n_m = f[mask], na[mask]
    # MAD winsorize
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

# ── 输出因子CSV ──────────────────────────────────────────────────
out = kline[['date', 'stock_code', 'vol_up_conc', 'neutral_factor']].dropna(subset=['neutral_factor'])
out = out.rename(columns={'neutral_factor': 'factor_value'})
outPath = PROJ / 'data' / 'factor_vol_up_conc_v1.csv'
out[['date', 'stock_code', 'factor_value']].to_csv(outPath, index=False)
print(f"因子CSV → {outPath}  ({len(out)} 行, 最新日期 {out['date'].max()})")
