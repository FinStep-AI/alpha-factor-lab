"""
因子：ROE协同改善 (ROE-BPS Synergistic Momentum)
ID: roe_bps_synergy_v1
Barra: Growth

逻辑：
  得 BP S cr ↑ nyi当同 → 盈利质量和资源增长同步
  当同为下降 → 双重恶化
  方向不一 → 噪声，置0去噪
  对 cross-sectional 因子值做 log_mktcap OLS 中性化

数据对齐：报告期结束 + REPORT_LAG 天滞后后 public，映射到最近交易日
"""
import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')


def compute_factor(fund_file='data/csi1000_fundamental_cache.csv',
                   kline_file='data/csi1000_kline_raw.csv',
                   output_file='data/factor_roe_bps_synergy_v1.csv'):
    # ── 1. 读数据 ──────────────────────────────────────────────────────────
    fund = pd.read_csv(fund_file)
    fund['report_date'] = pd.to_datetime(fund['report_date'])
    fund = fund.dropna(subset=['roe', 'bps', 'report_date']).copy()
    fund = fund.sort_values(['stock_code', 'report_date'])
    fund = fund.drop_duplicates(subset=['stock_code', 'report_date'], keep='first')

    # ── 2. 计算 ROE&BPS 环比变化（%, 即百分点变化） ─────────────────────────
    # roe: 已经是%，直接差分
    fund['delta_roe_pp'] = fund.groupby('stock_code')['roe'].diff()  # 百分点变化
    # bps: 计算相对变化（%）
    fund['prev_bps'] = fund.groupby('stock_code')['bps'].shift(1)
    fund['delta_bps_pct'] = (fund['bps'] - fund['prev_bps']) / fund['prev_bps'] * 100

    # 将 bps 对齐到当季 ROE 可用日（report_date + lag）
    REPORT_LAG = 45
    fund['avail_date'] = fund['report_date'] + pd.Timedelta(days=REPORT_LAG)
    fund = fund.dropna(subset=['delta_roe_pp', 'delta_bps_pct']).copy()

    # 只保留每只股票最新一期的 avail
    fund = fund.sort_values('avail_date').drop_duplicates(
        subset=['stock_code', 'avail_date'], keep='last')

    # ── 3. 计算协同信号：同向为正，反向为负，一为0则置0去噪 ────────────────
    signal_raw = fund['delta_roe_pp'] * fund['delta_bps_pct']
    # 方向不一（即信号<0）：说明一个改善一个恶化，视为噪声，置0
    fund['synergy_raw'] = signal_raw.where(signal_raw > 0, 0.0)

    # ── 4. 读kline ─────────────────────────────────────────────────────────
    kline = pd.read_csv(kline_file)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['date', 'stock_code']).reset_index(drop=True)

    # ── 5. Merge asof: 每个交易日取最近可用的 synergy ────────────────────────
    fund_s = fund[['avail_date', 'stock_code', 'synergy_raw']].sort_values('avail_date')
    kline_expanded = kline[['date', 'stock_code']].drop_duplicates().sort_values(
        ['stock_code', 'date'])

    factor_list = []
    from pandas.core.reshape.merge import merge_asof
    for sc, grp in kline_expanded.groupby('stock_code', sort=False):
        grp = grp.sort_values('date')
        sub = fund_s[fund_s['stock_code'] == sc].sort_values('avail_date')
        if sub.empty:
            continue
        merged = merge_asof(
            grp, sub,
            left_on='date', right_on='avail_date',
            by='stock_code', direction='backward'
        )
        factor_list.append(merged[['date', 'stock_code', 'synergy_raw']].copy())

    factor_df = pd.concat(factor_list, ignore_index=True)
    factor_df = factor_df.drop_duplicates(subset=['date', 'stock_code'], keep='last')

    # ── 6. 全截面标准化（MAD + z-score）──────────────────────────────────────
    def _cross_std(group):
        vals = group['synergy_raw'].values.astype(float)
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - med))
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

    # ── 7. 市值中性化（OLS） ──────────────────────────────────────────────────
    kline['log_amount'] = np.log(kline['amount'].clip(lower=1) + 1)
    # 用 20 日 log_amount 均值为代理市值
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
        x_m, y_m = x[mask], y[mask]
        X = np.column_stack([np.ones(len(x_m)), x_m])
        beta, _, _, _ = np.linalg.lstsq(X, y_m, rcond=None)
        resid = np.full(len(x), np.nan)
        resid[mask] = y_m - X @ beta
        r_med = np.nanmedian(resid[mask])
        r_std = np.nanstd(resid[mask])
        if r_std < 1e-12:
            grp['factor_neutral'] = 0.0
        else:
            out = pd.Series(np.where(mask, (resid - r_med) / r_std, np.nan),
                            index=grp.index)
            grp['factor_neutral'] = out
        residuals.append(grp[['date', 'stock_code', 'factor_neutral']])

    result = pd.concat(residuals, ignore_index=True)
    result = result.dropna(subset=['factor_neutral'])
    result = result.sort_values(['date', 'stock_code']).reset_index(drop=True)

    # ── 8. 输出 ─────────────────────────────────────────────────────────────
    result[['date', 'stock_code', 'factor_neutral']].rename(
        columns={'factor_neutral': 'factor_value'}
    ).to_csv(output_file, index=False)
    print(f"Done: {len(result)} rows → {output_file}")
    print(f"Dates: {result['date'].min()} ~ {result['date'].max()}")
    print(f"Stocks per date (sample): {result.groupby('date')['stock_code'].count().describe()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fund', default='data/csi1000_fundamental_cache.csv')
    parser.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    parser.add_argument('--output', default='data/factor_roe_bps_synergy_v1.csv')
    args = parser.parse_args()
    compute_factor(args.fund, args.kline, args.output)
