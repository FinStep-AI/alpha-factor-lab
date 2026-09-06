#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd


def robust_z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad < 1e-12:
        std = s.std()
        if pd.isna(std) or std < 1e-12:
            return pd.Series(np.nan, index=s.index)
        return ((s - s.mean()) / std).clip(-5, 5)
    return (((s - med) / (1.4826 * mad)).clip(-5, 5))


def neutralize_cs(df: pd.DataFrame, raw_col: str, cap_col: str) -> pd.Series:
    y = df[raw_col].astype(float).values
    x = df[cap_col].astype(float).values
    mask = np.isfinite(y) & np.isfinite(x)
    out = np.full(len(df), np.nan)
    if mask.sum() < 30:
        return pd.Series(out, index=df.index)
    yy = y[mask]
    xx = x[mask]
    X = np.column_stack([np.ones(len(xx)), xx])
    beta = np.linalg.lstsq(X, yy, rcond=None)[0]
    resid = yy - X @ beta
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    if mad < 1e-12:
        return pd.Series(out, index=df.index)
    resid = np.clip(resid, med - 5.2 * mad, med + 5.2 * mad)
    std = resid.std()
    if std < 1e-12:
        return pd.Series(out, index=df.index)
    z = (resid - np.median(resid)) / std
    out[np.where(mask)[0]] = z
    return pd.Series(out, index=df.index)


def compute_factor(revenue_file, kline_file, output_file):
    fund = pd.read_csv(revenue_file)
    fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
    fund['report_date'] = pd.to_datetime(fund['end_date'])
    fund['avail_date'] = pd.to_datetime(fund['info_publ_date'])
    cols = [
        'operate_revenue_yoy','net_profit_cut_yoy','current_ratio','quick_ratio','debt_assets_ratio',
        'inventory_turnover_rate','account_receivable_operate_revenue','cash_inflow_current_debt',
        'inventory_current_asset','account_receivable_yoy','gross_income_ratio'
    ]
    for c in cols:
        fund[c] = pd.to_numeric(fund[c], errors='coerce')
    fund = fund.sort_values(['stock_code','report_date']).drop_duplicates(['stock_code','report_date'], keep='last')
    g = fund.groupby('stock_code')

    fund['rev_yoy_ma2'] = g['operate_revenue_yoy'].transform(lambda s: s.rolling(2, min_periods=2).mean())
    fund['profit_yoy_ma2'] = g['net_profit_cut_yoy'].transform(lambda s: s.rolling(2, min_periods=2).mean())
    fund['rev_profit_gap'] = fund['net_profit_cut_yoy'] - fund['operate_revenue_yoy']
    fund['rev_profit_gap_ma2'] = g['rev_profit_gap'].transform(lambda s: s.rolling(2, min_periods=2).mean())
    fund['ar_ratio_delta'] = fund['account_receivable_operate_revenue'] - g['account_receivable_operate_revenue'].shift(1)
    fund['ar_yoy_ma2'] = g['account_receivable_yoy'].transform(lambda s: s.rolling(2, min_periods=2).mean())
    fund['inv_turn_delta'] = fund['inventory_turnover_rate'] - g['inventory_turnover_rate'].shift(1)
    fund['inv_ca_delta'] = fund['inventory_current_asset'] - g['inventory_current_asset'].shift(1)
    fund['liq_buffer'] = 0.5 * fund['current_ratio'] + 0.5 * fund['quick_ratio']
    fund['debt_delta'] = fund['debt_assets_ratio'] - g['debt_assets_ratio'].shift(1)
    fund['cash_debt_ma2'] = g['cash_inflow_current_debt'].transform(lambda s: s.rolling(2, min_periods=2).mean())
    fund['gm_level_ma2'] = g['gross_income_ratio'].transform(lambda s: s.rolling(2, min_periods=2).mean())

    def cross_section(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.copy()
        grp['z_rev_growth'] = robust_z(grp['rev_yoy_ma2'])
        grp['z_profit_growth'] = robust_z(grp['profit_yoy_ma2'])
        grp['z_gap'] = robust_z(grp['rev_profit_gap_ma2'])
        grp['z_ar_delta'] = robust_z(grp['ar_ratio_delta'])
        grp['z_ar_yoy'] = robust_z(grp['ar_yoy_ma2'])
        grp['z_inv_turn'] = robust_z(grp['inv_turn_delta'])
        grp['z_inv_ca'] = robust_z(grp['inv_ca_delta'])
        grp['z_liq'] = robust_z(grp['liq_buffer'])
        grp['z_debt_delta'] = robust_z(grp['debt_delta'])
        grp['z_cash_debt'] = robust_z(grp['cash_debt_ma2'])
        grp['z_gm'] = robust_z(grp['gm_level_ma2'])
        grp['raw_factor'] = (
            0.22 * grp['z_rev_growth'] +
            0.10 * grp['z_profit_growth'] -
            0.18 * grp['z_gap'] -
            0.12 * grp['z_ar_delta'] -
            0.08 * grp['z_ar_yoy'] +
            0.10 * grp['z_inv_turn'] -
            0.08 * grp['z_inv_ca'] +
            0.10 * grp['z_liq'] -
            0.07 * grp['z_debt_delta'] +
            0.08 * grp['z_cash_debt'] +
            0.07 * grp['z_gm']
        )
        return grp

    fund = fund.groupby('report_date', group_keys=False).apply(cross_section)
    fund = fund.replace([np.inf, -np.inf], np.nan).dropna(subset=['raw_factor', 'avail_date'])
    factor_q = fund[['stock_code', 'avail_date', 'raw_factor']].rename(columns={'avail_date': 'date'})

    kline = pd.read_csv(kline_file, usecols=['date', 'stock_code', 'close', 'amount', 'turnover'])
    kline['date'] = pd.to_datetime(kline['date'])
    kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
    kline = kline.sort_values(['stock_code', 'date']).drop_duplicates(['date', 'stock_code'])
    kline['mktcap_proxy'] = kline['close'].clip(lower=0.01) * kline['amount'].clip(lower=1) / (kline['turnover'].replace(0, np.nan) + 1e-6)
    kline['log_mktcap'] = np.log(kline['mktcap_proxy'].clip(lower=1))
    trade_dates = pd.Index(sorted(kline['date'].unique()))

    aligned = []
    for stock, grp in factor_q.groupby('stock_code'):
        sf = grp[['date', 'raw_factor']].drop_duplicates('date', keep='last').set_index('date').sort_index()
        sf = sf.reindex(trade_dates, method='ffill', limit=80)
        sf['stock_code'] = stock
        sf = sf.dropna(subset=['raw_factor']).reset_index().rename(columns={'index': 'date'})
        aligned.append(sf)
    factor = pd.concat(aligned, ignore_index=True)
    factor = factor.merge(kline[['date', 'stock_code', 'log_mktcap']], on=['date', 'stock_code'], how='inner')
    factor = factor.dropna(subset=['raw_factor', 'log_mktcap'])

    out = []
    for date, grp in factor.groupby('date'):
        nz = neutralize_cs(grp, 'raw_factor', 'log_mktcap')
        good = nz.notna()
        if good.sum() < 30:
            continue
        sub = grp.loc[good, ['date', 'stock_code']].copy()
        sub['factor'] = nz.loc[good].values
        out.append(sub)

    result = pd.concat(out, ignore_index=True)
    result['date'] = pd.to_datetime(result['date']).dt.strftime('%Y-%m-%d')
    result.to_csv(output_file, index=False, float_format='%.6f')
    print(f'saved {output_file} rows={len(result)} dates={result.date.min()}~{result.date.max()} stocks={result.stock_code.nunique()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--revenue', default='data/csi1000_revenue_yoy_raw.csv')
    parser.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    parser.add_argument('--output', default='data/factor_cashflow_guarded_growth_v1.csv')
    args = parser.parse_args()
    compute_factor(args.revenue, args.kline, args.output)
