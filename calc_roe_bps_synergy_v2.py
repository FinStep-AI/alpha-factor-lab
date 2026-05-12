"""
因子：ROE-BPS rank协同改善 (ROE-BPS Rank Synergy v2)
ID: roe_bps_synergy_v2
Barra: Growth

逻辑：
  1. 每季度对全截面 delta_roe_pp做 rank (0~1)
  2. 同截面 delta_bps_pct做 rank (0~1)
  3. synergy = rank_roe * rank_bps  （两者都靠前的值趋近1，一个大一个小趋近0）
  4. 去极值: rank*rank ∈ [0,1] 天然有界，无需额外去极
  5. 取季度末报告后45天为因子值，备用
 yielded approach正常。对季度报告做时间对齐，用报告日期加滞后映射到交易日，然后通过asof merge对齐每日kline。不过考虑到报告频率很低，每季度更新一次信号，在调仓周期内保持续是正确的做法。
  6. 对 daily factor 做 cross-sectional MAD + z-score
  7. 对 z-score 做 log_amount_ma20 OLS 中性化
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

    # ── 2. ROE 环比（百分点变化） & BPS 环比（%） ──────────────────────────
    fund['delta_roe_pp'] = fund.groupby('stock_code')['roe'].diff()
    fund['prev_bps'] = fund.groupby('stock_code')['bps'].shift(1)
    fund['delta_bps_pct'] = (fund['bps'] - fund['prev_bps']) / fund['prev_bps'] * 100
    fund = fund.dropna(subset=['delta_roe_pp', 'delta_bps_pct']).copy()

    # ── 3. 每报告日全截面做 rank (0~1)，乘积得 synergy ─────────────────────
    fund['rank_roe'] = fund.groupby('report_date')['delta_roe_pp'].rank(pct=True)
    fund['rank_bps'] = fund.groupby('report_date')['delta_bps_pct'].rank(pct=True)
    fund['synergy_rank'] = fund['rank_roe'] * fund['rank_bps']

    # ── 4. 报告日映射交易日 ──────────────────────────────────────────────────
    REPORT_LAG = 45
    fund['avail_date'] = fund['report_date'] + pd.Timedelta(days=REPORT_LAG)
    fund = fund.drop_duplicates(subset=['stock_code', 'avail_date'], keep='last')

    # ── 5. 读kline & 对齐 ───────────────────────────────────────────────────
    kline = pd.read_csv(kline_file)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['date', 'stock_code']).reset_index(drop=True)

    kline_expanded = kline[['date', 'stock_code']].drop_duplicates().sort_values(
        ['stock_code', 'date'])

    factor_list = []
    from pandas.core.reshape.merge import merge_asof
    for sc, grp in kline_expanded.groupby('stock_code', sort=False):
        grp = grp.sort_values('date')
        sub = fund[fund['stock_code'] == sc][['avail_date', 'stock_code', 'synergy_rank']].sort_values('avail_date')
        if sub.empty:
            continue
        merged = merge_asof(grp, sub, left_on='date', right_on='avail_date',
                            by='stock_code', direction='backward')
        factor_list.append(merged[['date', 'stock_code', 'synergy_rank']].copy())

    factor_df = pd.concat(factor_list, ignore_index=True)
    factor_df = factor_df.drop_duplicates(subset=['date', 'stock_code'], keep='last')

    # ── 6. 全局去极值+z-score ────────────────────────────────────────────────
    def _cross_std(group):
        vals = group['synergy_rank'].values.astype(float)
        med  = np.nanmedian(vals)
        mad  = np.nanmedian(np.abs(vals - med))
        if mad < 1e-12:
            group['factor_value'] = 0.0
            return group
        upper = med + 5.2 * 1.4826 * mad
        lower = med - 5.2 * 1.4826 * mad
        vals_clip = np.clip(vals, lower, upper)
        m, s = np.nanmean(vals_clip), np.nanstd(vals_clip)
        if s < 1e-12:
            group['factor_value'] = 0.0
        else:
            group['factor_value'] = (vals_clip - m) / s
        return group

    factor_df = factor_df.groupby('date', group_keys=False).apply(_cross_std)

    # ── 7. 市值代理（log_amount_ma20）+ 中性化 ──────────────────────────────
    kline['log_amount'] = np.log(kline['amount'].clip(lower=1) + 1)
    kline['log_amount_ma20'] = (
        kline.groupby('stock_code')['log_amount']
                .transform(lambda x: x.rolling(20, min_periods=10).mean())
    )

    merged2 = factor_df.merge(
        kline[['date', 'stock_code', 'log_amount_ma20']].drop_duplicates(),
        on=['date', 'stock_code'], how='inner'
    )
    merged2 = merged2.dropna(subset=['factor_value', 'log_amount_ma20'])

    residuals = []
    for dt, grp in merged2.groupby('date'):
        x = grp['log_amount_ma20'].values.astype(float)
        y = grp['factor_value'].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 30:
            grp['factor_neutral'] = 0.0
            residuals.append(grp[['date', 'stock_code', 'factor_neutral']])
            continue
        X = np.column_stack([np.ones(mask.sum()), x[mask]])
        beta, _, _, _ = np.linalg.lstsq(X, y[mask], rcond=None)
        resid = np.full(len(x), np.nan)
        resid[mask] = y[mask] - X @ beta
        r_med = np.nanmedian(resid[mask])
        r_std = np.nanstd(resid[mask])
        if r_std < 1e-12:
            grp['factor_neutral'] = 0.0
        else:
            grp['factor_neutral'] = pd.Series(
                np.where(mask, (resid - r_med) / r_std, np.nan), index=grp.index
            )
        residuals.append(grp[['date', 'stock_code', 'factor_neutral']])

    result = pd.concat(residuals, ignore_index=True)
    result = result.dropna(subset=['factor_neutral'])
    result = result.sort_values(['date', 'stock_code']).reset_index(drop=True)

    result[['date', 'stock_code', 'factor_neutral']].rename(
        columns={'factor_neutral': 'factor_value'}
    ).to_csv(output_file, index=False)
    print(f"Done: {len(result)} rows → {output_file}")
    print(f"Dates: {result['date'].min()} ~ {result['date'].max()}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fund',  default='data/csi1000_fundamental_cache.csv')
    p.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    p.add_argument('--output',default='data/factor_roe_bps_synergy_v2.csv')
    a = p.parse_args()
    compute_factor(a.fund, a.kline, a.output)
