"""
因子：ROE-BPS rank协同改善 v3 → 事件驱动版
ID: roe_bps_synergy_v2
Barra: Growth

逻辑（事件驱动，不改周）：
  每季度对全截面做：
    rank_roe = (delta_roe_pp) cross-sectional rank
    rank_bps = (delta_bps_pct) cross-sectional rank
    synergy  = rank_roe × rank_bps
  取 REPORT_DATE + REPORT_LAG 对应的交易日，直接作为因子值在那里可用

输出格式与 factor_backtest.py 兼容的日度 CSV（每季度日期交叉截面一行）
后续由 backtest 负责计算 IC，不再预映射到所有交易日
"""
import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')


def compute_factor(fund_file='data/csi1000_fundamental_cache.csv',
                   kline_file='data/csi1000_kline_raw.csv',
                   output_file='data/factor_roe_bps_synergy_v2.csv'):
    # ── 1. 读数据 ──────────────────────────────────────────────────────────
    fund = pd.read_csv(fund_file)
    fund['report_date'] = pd.to_datetime(fund['report_date'])
    fund = fund.dropna(subset=['roe', 'bps', 'report_date']).copy()
    fund = fund.sort_values(['stock_code', 'report_date'])
    fund = fund.drop_duplicates(subset=['stock_code', 'report_date'], keep='first')

    # ── 2. 环比变化 ─────────────────────────────────────────────────────────
    fund['delta_roe_pp'] = fund.groupby('stock_code')['roe'].diff()
    fund['prev_bps'] = fund.groupby('stock_code')['bps'].shift(1)
    fund['delta_bps_pct'] = (fund['bps'] - fund['prev_bps']) / fund['prev_bps'] * 100
    fund = fund.dropna(subset=['delta_roe_pp', 'delta_bps_pct']).copy()

    # ── 3. 每报告日全截面 rank，乘积得 synergy ──────────────────────────────
    fund['rank_roe'] = fund.groupby('report_date')['delta_roe_pp'].rank(pct=True)
    fund['rank_bps'] = fund.groupby('report_date')['delta_bps_pct'].rank(pct=True)
    fund['synergy'] = fund['rank_roe'] * fund['rank_bps']

    REPORT_LAG = 45
    fund['avail_date'] = fund['report_date'] + pd.Timedelta(days=REPORT_LAG)

    # ── 4. 取“报告可用后的第一个交易日” ─────────────────────────────────────
    kline = pd.read_csv(kline_file)
    kline['date'] = pd.to_datetime(kline['date'])
    trade_dates = sorted(kline['date'].unique())

    # 为每个 avail_date 找 >= avail_date 的第一个交易日
    from pandas.core.reshape.merge import merge_asof
    avail_cal = pd.DataFrame({'avail_date': sorted(fund['avail_date'].unique())}).sort_values('avail_date')
    trade_cal = pd.DataFrame({'trade_date': trade_dates}).sort_values('trade_date')
    avail_cal = merge_asof(trade_cal, avail_cal, left_on='trade_date', right_on='avail_date',
                           direction='backward')
    # avail_date ≤ trade_date 的最近一档
    date_map = dict(zip(avail_cal['avail_date'], avail_cal['trade_date']))
    fund['factor_date'] = fund['avail_date'].map(date_map)
    fund = fund.dropna(subset=['factor_date'])

    # ── 5. 构建季度因子矩阵 ──────────────────────────────────────────────────
    factor_mat = fund.pivot_table(index='factor_date', columns='stock_code',
                                  values='synergy', aggfunc='first')
    # ── 6. 去极值+z-score ─────────────────────────────────────────────────
    def zscore_row(row):
        vals = row.values.astype(float)
        med  = np.nanmedian(vals)
        mad  = np.nanmedian(np.abs(vals - med))
        if mad < 1e-12:
            return pd.Series(np.zeros(len(row)), index=row.index)
        upper = med + 5.2 * 1.4826 * mad
        lower = med - 5.2 * 1.4826 * mad
        vals_clip = np.clip(vals, lower, upper)
        m, s = np.nanmean(vals_clip), np.nanstd(vals_clip)
        if s < 1e-12:
            return pd.Series(np.zeros(len(row)), index=row.index)
        return pd.Series((vals_clip - m) / s, index=row.index)

    factor_z = factor_mat.apply(zscore_row, axis=1)

    # ── 7. 市值中性化（log_amount_ma20） ───────────────────────────────────
    kline['log_amount'] = np.log(kline['amount'].clip(lower=1) + 1)
    kline['log_amount_ma20'] = (
        kline.groupby('stock_code')['log_amount']
                .transform(lambda x: x.rolling(20, min_periods=10).mean())
    )
    mkt_wide = kline.pivot_table(index='date', columns='stock_code',
                                  values='log_amount_ma20', aggfunc='last')

    common_dates = factor_z.index.intersection(mkt_wide.index)
    common_scs   = factor_z.columns.intersection(mkt_wide.columns)
    factor_z = factor_z.loc[common_dates, common_scs]
    mkt_wide = mkt_wide.loc[common_dates, common_scs]

    def neutral_row(idx, row, row_mkt):
        y = row.values.astype(float)
        x = row_mkt.values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 30:
            return row
        X = np.column_stack([np.ones(mask.sum()), x[mask]])
        beta, _, _, _ = np.linalg.lstsq(X, y[mask], rcond=None)
        resid = np.full(len(y), np.nan)
        resid[mask] = y[mask] - X @ beta
        r_med, r_std = np.nanmedian(resid[mask]), np.nanstd(resid[mask])
        if r_std < 1e-12:
            return row * 0
        new_row = row.copy()
        new_row.iloc[:] = np.where(mask, (resid - r_med) / r_std, np.nan)
        return new_row

    factor_neutral = pd.DataFrame(index=factor_z.index, columns=factor_z.columns, dtype=float)
    for dt in factor_z.index:
        factor_neutral.loc[dt] = neutral_row(
            dt, factor_z.loc[dt], mkt_wide.loc[dt]
        ).values

    # ── 8. 输出 ─────────────────────────────────────────────────────────────
    records = []
    for dt, row in factor_neutral.dropna(how='all').iterrows():
        for sc, val in row.dropna().items():
            records.append({'date': dt, 'stock_code': sc, 'factor_value': round(float(val), 6)})
    pd.DataFrame(records).to_csv(output_file, index=False)
    print(f"Done: {len(records)} factor obs → {output_file}")
    print(f"Factor dates ({len(factor_neutral)} quarters):",
          factor_neutral.index.min(), '~', factor_neutral.index.max())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fund',  default='data/csi1000_fundamental_cache.csv')
    p.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    p.add_argument('--output',default='data/factor_roe_bps_synergy_v2.csv')
    a = p.parse_args()
    compute_factor(a.fund, a.kline, a.output)
