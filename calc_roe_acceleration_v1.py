"""
因子：ROE加速度 (ROE Level Speed) v1
ID: roe_acceleration_v1
Barra: Growth

逻辑：
  ROE_MA4 = 4季滚动均值（平滑季度噪音）
  accel = 当前季ROE_MA4 - 上一季ROE_MA4
  正=ROE加速改善 → 动量方向上头 → 未来收益更高 (Growth信号)

中性化：成交额20日均值OLS中性化 + MAD去极值 + z-score
"""

import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')


def compute_factor(fund_file='data/csi1000_fundamental_cache.csv',
                   kline_file='data/csi1000_kline_raw.csv',
                   output_file='data/factor_roe_acceleration_v1.csv'):
    """计算ROE加速度因子，对齐到每日交易日。"""

    # ── 读取数据 ────────────────────────────────────────────────────────────
    fund = pd.read_csv(fund_file)
    fund['report_date'] = pd.to_datetime(fund['report_date'])
    fund = fund.dropna(subset=['roe', 'report_date']).copy()
    fund_s = fund.sort_values(['stock_code', 'report_date']).drop_duplicates(
        subset=['stock_code', 'report_date'], keep='first'
    ).reset_index(drop=True)

    kline = pd.read_csv(kline_file)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['date', 'stock_code']).reset_index(drop=True)

    # ── 计算 ROE_MA4 和 accel_raw ────────────────────────────────────────────
    fund_s['roe_ma4'] = (
        fund_s.groupby('stock_code')['roe']
        .transform(lambda x: x.rolling(4, min_periods=3).mean())
    )
    fund_s['prev_ma4'] = fund_s.groupby('stock_code')['roe_ma4'].shift(1)
    fund_s['accel_raw'] = fund_s['roe_ma4'] - fund_s['prev_ma4']

    # ── 报告日映射 → 交易日 (report + 45天延迟) ────────────────────────────
    REPORT_LAG = 45
    fund_s['avail_date'] = fund_s['report_date'] + pd.Timedelta(days=REPORT_LAG)
    fund_s = fund_s.dropna(subset=['accel_raw'])

    # 只保留每只股票每报告期最新一条
    fund_s = fund_s.drop_duplicates(subset=['stock_code', 'avail_date'], keep='last')

    # ── Merge with kline 借 tag 快速扩展 ─────────────────────────────────────
    # 生成全量 kline 的 avail_date_snapshot标签 (最近Avail报告)
    kline_dates = sorted(kline['date'].unique())

    # 用 asof merge: 对齐 kline 和 fund 的 avail_date
    fund_s = fund_s.sort_values('avail_date')
    kline_merge = kline[['date', 'stock_code']].drop_duplicates().sort_values('date')

    # pd.merge_asof 只能一次对齐一列，这里做 merge_asof per stock ( slows for many)
    # Instead: create pivot and asof merge on date

    # 展平: factor 每个 avail_date × sc 一行，卖方 → buyer 日期
    # 用简单并集日期对齐：对每个 avdate，找所有 kline date <= avdate 中的最大值
    # faster: 用 merge_asof with assumption

    # Pivot fund for O(1) lookup by date→sc→accel
    fund_s_sorted = fund_s.sort_values(['stock_code', 'avail_date'])  # for asof

    # compute factor in vectorized fashion: for each kline date, take latest fund avail
    from pandas.core.reshape.merge import merge_asof

    # Expand kline with all sc, create a 'merge_key' through date
    kline_expanded = kline[['date', 'stock_code']].drop_duplicates().sort_values(['stock_code', 'date'])

    # asof merge per stock (using method='nearest')
    accel_aligned_list = []
    for sc, grp_k in kline_expanded.groupby('stock_code'):
        fund_sc = fund_s[fund_s['stock_code'] == sc][['avail_date', 'accel_raw']].sort_values('avail_date')
        if fund_sc.empty:
            continue
        merged = pd.merge_asof(
            grp_k.sort_values('date'),
            fund_sc,
            left_on='date',
            right_on='avail_date',
            direction='backward'
        )
        merged['factor_raw'] = merged['accel_raw']
        accel_aligned_list.append(merged[['date', 'stock_code', 'factor_raw']])

    raw_factor = pd.concat(accel_aligned_list, ignore_index=True)
    print(f"  [进度] 原始因子: {len(raw_factor):,} 条, {raw_factor['date'].nunique()} 个交易日")

    # ── 成交额中性化 ─────────────────────────────────────────────────────────
    amt_20d = (
        kline.groupby(['stock_code', 'date'])['amount']
        .last().reset_index()
        .sort_values(['stock_code', 'date'])
    )
    amt_20d['log_amount_20d'] = (
        amt_20d.groupby('stock_code')['amount']
        .transform(lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1))
        .reset_index(drop=True)
    )

    # pivot
    factor_pivot = raw_factor.pivot_table(index='date', columns='stock_code', values='factor_raw')
    amt_pivot = amt_20d.pivot_table(index='date', columns='stock_code', values='log_amount_20d')
    common_dates = factor_pivot.index.intersection(amt_pivot.index).sort_values()
    print(f"  [进度] 对齐交易日: {len(common_dates)}")

    result_records = []
    for i, td in enumerate(common_dates):
        fv = factor_pivot.loc[td].dropna()
        av = amt_pivot.loc[td]
        sc_common = fv.index.intersection(av.index)
        if len(sc_common) < 50:
            continue

        v = fv[sc_common].values.astype(float)
        x = av[sc_common].values.astype(float)
        mask = np.isfinite(v) & np.isfinite(x)
        if mask.sum() < 50:
            continue
        v2, x2 = v[mask], x[mask]

        try:
            X = np.column_stack([np.ones(len(x2)), x2])
            beta, _, _, _ = np.linalg.lstsq(X, v2, rcond=None)
            resid = v2 - X @ beta
            med = np.median(resid)
            mad = np.median(np.abs(resid - med)) + 1e-10
            # MAD标准化 → Winsorize → z-score
            r_mad = (resid - med) / (1.4826 * mad)
            r_clip = np.clip(r_mad, -5, 5)
            std = r_clip.std()
            if std < 1e-10:
                continue
            z = (r_clip - r_clip.mean()) / std

            for sc, val in zip(sc_common[mask], z):
                result_records.append({'date': td, 'stock_code': sc, 'factor': float(val)})
        except Exception as e:
            continue

        if i % 80 == 0:
            print(f"  [进度] {i}/{len(common_dates)} 项完成 ({i/len(common_dates)*100:.1f}%)")

    factor_df = pd.DataFrame(result_records)
    if factor_df.empty:
        print("[错误] 因子为空")
        return factor_df

    # ── 前向填充 + 截断末尾 ────────────────────────────────────────────────
    factor_df = factor_df.sort_values(['stock_code', 'date'])
    factor_df['factor'] = factor_df.groupby('stock_code')['factor'].ffill(limit=5)
    all_dates = sorted(factor_df['date'].unique())
    if len(all_dates) > 5:
        cutoff = all_dates[-5]
        factor_df = factor_df[factor_df['date'] <= cutoff].copy()

    print(f"\n[完成] 因子记录: {len(factor_df):,} 条")
    print(f"        日期区间: {factor_df['date'].min().date()} ~ {factor_df['date'].max().date()}")
    print(f"        股票覆盖: {factor_df['stock_code'].nunique()} 只")
    print(f"        均值={factor_df['factor'].mean():.4f}, std={factor_df['factor'].std():.4f}")

    factor_df.to_csv(output_file, index=False)
    print(f"[完成] 保存: {output_file}")
    return factor_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fundamental', default='data/csi1000_fundamental_cache.csv')
    parser.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    parser.add_argument('--output', default='data/factor_roe_acceleration_v1.csv')
    args = parser.parse_args()
    compute_factor(args.fundamental, args.kline, args.output)
