#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资产周转率变化代理因子 v1
思路：用季度销售代理 = BPS * ROE，构造 sales_proxy / BPS ≈ ROE 的资本效率变化，
重点奖励 ROE 同比改善且改善加速度为正、同时 BPS 同比扩张不过度失真（避免纯扩表）。
最后映射到日频，并对 log(20日平均成交额) 做截面 OLS 市值中性化，再 z-score。

raw = 0.55*z(roe_yoy) + 0.25*z(roe_accel) + 0.20*z(bps_yoy)
      - 0.20*z(abs(bps_yoy - roe_yoy))

解释：
- roe_yoy：资本效率改善主项（Growth）
- roe_accel：改善是否在加速
- bps_yoy：净资产稳步扩张可辅助确认成长不是一次性
- gap penalty：若 BPS 扩张远快于 ROE 改善，可能是低效扩表，予以惩罚
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def cs_robust_z(s: pd.Series) -> pd.Series:
    m = s.notna() & np.isfinite(s)
    if m.sum() < 20:
        return pd.Series(np.nan, index=s.index)
    x = s[m]
    med = x.median()
    mad = (x - med).abs().median()
    if pd.isna(mad) or mad < 1e-12:
        std = x.std()
        if pd.isna(std) or std < 1e-12:
            out = pd.Series(0.0, index=s.index)
            out[~m] = np.nan
            return out
        out = (s - x.mean()) / std
        return out
    z = (s - med) / (1.4826 * mad)
    return z.clip(-5, 5)


def neutralize(group: pd.DataFrame) -> pd.Series:
    y = group['raw']
    x = group['log_amount_20d']
    mask = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 30:
        return pd.Series(np.nan, index=group.index)
    X = np.column_stack([np.ones(mask.sum()), x[mask].values])
    beta = np.linalg.lstsq(X, y[mask].values, rcond=None)[0]
    resid = y[mask].values - X @ beta
    out = pd.Series(np.nan, index=group.index)
    out.loc[mask] = resid
    return out


def main():
    funda = pd.read_csv(DATA / 'csi1000_fundamental_cache.csv')
    kline = pd.read_csv(DATA / 'csi1000_kline_raw.csv', usecols=['date','stock_code','amount'])
    funda['report_date'] = pd.to_datetime(funda['report_date'])
    funda = funda.sort_values(['stock_code','report_date']).copy()

    for col in ['roe','bps']:
        funda[col] = pd.to_numeric(funda[col], errors='coerce')

    g = funda.groupby('stock_code', group_keys=False)
    funda['roe_lag4'] = g['roe'].shift(4)
    funda['bps_lag4'] = g['bps'].shift(4)
    funda['roe_yoy'] = funda['roe'] - funda['roe_lag4']
    funda['bps_yoy'] = (funda['bps'] / funda['bps_lag4'] - 1.0).replace([np.inf, -np.inf], np.nan)
    funda['roe_accel'] = funda.groupby('stock_code')['roe_yoy'].diff()
    funda['gap_penalty'] = (funda['bps_yoy'] - funda['roe_yoy'] / 100.0).abs()

    funda['z_roe_yoy'] = funda.groupby('report_date')['roe_yoy'].transform(cs_robust_z)
    funda['z_roe_accel'] = funda.groupby('report_date')['roe_accel'].transform(cs_robust_z)
    funda['z_bps_yoy'] = funda.groupby('report_date')['bps_yoy'].transform(cs_robust_z)
    funda['z_gap'] = funda.groupby('report_date')['gap_penalty'].transform(cs_robust_z)

    funda['raw_report'] = (
        0.55 * funda['z_roe_yoy'] +
        0.25 * funda['z_roe_accel'] +
        0.20 * funda['z_bps_yoy'] -
        0.20 * funda['z_gap']
    )

    # 财报可用日：季度后45天
    funda['date'] = funda['report_date'] + pd.Timedelta(days=45)
    rep = funda[['date','stock_code','raw_report']].dropna().copy()

    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['stock_code','date'])
    kline['log_amount_20d'] = kline.groupby('stock_code')['amount'].transform(
        lambda s: np.log(s.rolling(20, min_periods=10).mean() + 1.0)
    )
    panel = kline[['date','stock_code','log_amount_20d']].copy()

    panel = panel.merge(rep, on=['date','stock_code'], how='left').sort_values(['stock_code','date'])
    panel['raw_report'] = panel.groupby('stock_code')['raw_report'].ffill()

    # 上市后拿到首份财报前不填值
    first_date = rep.groupby('stock_code')['date'].min().rename('first_factor_date')
    panel = panel.merge(first_date, on='stock_code', how='left')
    panel.loc[panel['date'] < panel['first_factor_date'], 'raw_report'] = np.nan

    panel = panel.rename(columns={'raw_report':'raw'})
    panel['neutral'] = panel.groupby('date', group_keys=False).apply(neutralize)
    panel['factor_value'] = panel.groupby('date')['neutral'].transform(cs_robust_z)

    out = panel[['date','stock_code','factor_value']].dropna().copy()
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    out_path = DATA / 'factor_asset_turnover_proxy_growth_v1.csv'
    out.to_csv(out_path, index=False)
    print(out_path)
    print(out.head())
    print('rows', len(out), 'dates', out['date'].nunique())

if __name__ == '__main__':
    main()
