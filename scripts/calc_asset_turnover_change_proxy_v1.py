#!/usr/bin/env python3
"""
资产周转率变化代理因子 v1
Barra style: Growth / Quality

在仅有 bps / roe 的财报字段下，用 ROE / BPS 近似“盈利相对净资产效率”，
再结合其同比变化与加速度，构造资产周转/经营效率改善代理。

raw = 0.50 * atan(turn_eff_level / 1.5)
    + 0.35 * atan(turn_eff_yoy / 0.35)
    + 0.15 * atan(turn_eff_accel / 0.25)
    - 0.20 * atan(bps_yoy / 0.30)

其中：
  turn_eff = roe / bps
含义：
  - turn_eff_level: 当前经营效率水平
  - turn_eff_yoy: 效率同比改善
  - turn_eff_accel: 效率改善的二阶加速度
  - bps_yoy: 净资产扩张惩罚，避免把“单纯扩表”误判为效率提升

财报按 45 天滞后映射到日频；横截面对 log(20日平均成交额) 做 OLS 中性化。
输出: data/factor_asset_turnover_change_proxy_v1.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FUND_PATH = BASE / 'data' / 'csi1000_fundamental_cache.csv'
KLINE_PATH = BASE / 'data' / 'csi1000_kline_raw.csv'
OUT_PATH = BASE / 'data' / 'factor_asset_turnover_change_proxy_v1.csv'

REPORT_LAG = pd.Timedelta(days=45)


def mad_clip(s, k=5):
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad < 1e-12:
        return s
    width = 1.4826 * mad * k
    return s.clip(med - width, med + width)


def zscore(s):
    std = s.std()
    if pd.isna(std) or std < 1e-12:
        return s * 0.0
    return (s - s.mean()) / std


def load_data():
    fund = pd.read_csv(FUND_PATH)
    kline = pd.read_csv(KLINE_PATH)
    fund['report_date'] = pd.to_datetime(fund['report_date'])
    kline['date'] = pd.to_datetime(kline['date'])
    return fund, kline


def build_report_factor(fund: pd.DataFrame) -> pd.DataFrame:
    fund = fund.copy()
    fund = fund.sort_values(['stock_code', 'report_date']).drop_duplicates(['stock_code', 'report_date'])

    fund['roe'] = pd.to_numeric(fund['roe'], errors='coerce')
    fund['bps'] = pd.to_numeric(fund['bps'], errors='coerce')
    fund.loc[fund['roe'].abs() > 200, 'roe'] = np.nan
    fund.loc[fund['bps'] <= 0, 'bps'] = np.nan

    # 经营效率代理：盈利 / 净资产厚度
    fund['turn_eff'] = fund['roe'] / fund['bps']
    fund.loc[fund['turn_eff'].abs() > 20, 'turn_eff'] = np.nan

    g = fund.groupby('stock_code')
    fund['turn_eff_yoy'] = g['turn_eff'].shift(0) - g['turn_eff'].shift(4)
    fund['turn_eff_accel'] = fund['turn_eff_yoy'] - g['turn_eff_yoy'].shift(4)
    fund['bps_yoy'] = fund['bps'] / g['bps'].shift(4) - 1

    fund['raw'] = (
        0.50 * np.arctan(fund['turn_eff'] / 1.5)
        + 0.35 * np.arctan(fund['turn_eff_yoy'] / 0.35)
        + 0.15 * np.arctan(fund['turn_eff_accel'] / 0.25)
        - 0.20 * np.arctan(fund['bps_yoy'] / 0.30)
    )

    out = fund[['stock_code', 'report_date', 'raw']].dropna().copy()
    out['available_from'] = out['report_date'] + REPORT_LAG
    return out


def map_to_daily(report_factor: pd.DataFrame, kline: pd.DataFrame) -> pd.DataFrame:
    kline = kline.sort_values(['stock_code', 'date']).copy()
    kline['amount_20d'] = kline.groupby('stock_code')['amount'].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    kline['log_amount_20d'] = np.log(kline['amount_20d'].clip(lower=1))

    results = []
    for sc, px in kline.groupby('stock_code'):
        rf = report_factor[report_factor['stock_code'] == sc].sort_values('available_from')
        if rf.empty:
            continue
        px2 = px[['date', 'stock_code', 'log_amount_20d']].sort_values('date')
        merged = pd.merge_asof(
            px2,
            rf[['available_from', 'raw', 'stock_code']].rename(columns={'available_from': 'date'}),
            on='date', by='stock_code', direction='backward'
        )
        results.append(merged)
    daily = pd.concat(results, ignore_index=True)
    daily = daily.dropna(subset=['raw', 'log_amount_20d'])
    return daily


def neutralize_daily(daily: pd.DataFrame) -> pd.DataFrame:
    out = []
    for dt, grp in daily.groupby('date'):
        grp = grp.copy()
        if len(grp) < 50:
            continue
        grp['raw_clip'] = mad_clip(grp['raw'])
        x = grp['log_amount_20d'].to_numpy()
        y = grp['raw_clip'].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 50:
            continue
        X = np.column_stack([np.ones(mask.sum()), x[mask]])
        beta = np.linalg.lstsq(X, y[mask], rcond=None)[0]
        resid = np.full(len(grp), np.nan)
        resid[mask] = y[mask] - X @ beta
        grp['factor_value'] = zscore(pd.Series(resid, index=grp.index)).values
        out.append(grp[['date', 'stock_code', 'factor_value']])
    return pd.concat(out, ignore_index=True).dropna()


def main():
    fund, kline = load_data()
    report_factor = build_report_factor(fund)
    daily = map_to_daily(report_factor, kline)
    result = neutralize_daily(daily)
    result['stock_code'] = result['stock_code'].astype(int)
    result.to_csv(OUT_PATH, index=False)
    print(f'saved {OUT_PATH}')
    print(f'rows={len(result)} dates={result.date.nunique()} stocks={result.stock_code.nunique()}')
    print(result.head().to_string())


if __name__ == '__main__':
    main()
