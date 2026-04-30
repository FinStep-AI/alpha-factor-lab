"""
因子：ROE时序趋势质量 v2（全量滚动版）
ID: roe_trend_quality_v2
Barra: Growth

核心改进（vs v1）:
  - 报告日覆盖率扩展：并非仅在 report_date+45 那个时间点，而是在**所有K线交易日**,
    对一只股票的概念上做 rolling β(from 12-rolling-quarter ROE)，取最新季度报告可得的β
  - 因子值扩展至所有交易日 (前向填充+后向填充60天)
"""

import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')


def compute_factor(fund_file='data/csi1000_fundamental_cache.csv',
                   kline_file='data/csi1000_kline_raw.csv',
                   output_file='data/factor_roe_trend_quality_v2.csv',
                   window_quarters=12):
    """计算ROE时序趋势因子 v2（全量滚动）."""

    print("[1] 读取基本面数据...")
    fund = pd.read_csv(fund_file)
    fund['report_date'] = pd.to_datetime(fund['report_date'])
    fund = fund.dropna(subset=['roe', 'report_date']).copy()
    fund = fund.sort_values(['stock_code', 'report_date']) \
               .drop_duplicates(subset=['stock_code', 'report_date'], keep='first') \
               .reset_index(drop=True)
    print(f"      {fund['stock_code'].nunique()} 只, {fund['report_date'].min().date()} ~ {fund['report_date'].max().date()}")

    print("[2] 计算每报告期每只股票12季ROE的β斜率 × R²...")
    records = []
    for sc, grp in fund.groupby('stock_code'):
        g = grp.sort_values('report_date').reset_index(drop=True)
        n = len(g)
        if n < window_quarters:
            continue
        for i in range(window_quarters - 1, n):
            win = g.iloc[i - window_quarters + 1:i + 1]
            roes = win['roe'].values.astype(float)
            ts = np.arange(window_quarters, dtype=float)
            valid = np.isfinite(roes)
            if valid.sum() < 6:
                continue
            t_v, y_v = ts[valid], roes[valid]
            b, a = np.polyfit(t_v, y_v, 1)

            # R²
            y_pred = a + b * t_v
            ss_tot = np.sum((y_v - y_v.mean()) ** 2)
            ss_res = np.sum((y_v - y_pred) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

            avail_date = win['report_date'].iloc[-1]
            records.append({
                'stock_code': sc,
                'avail_date': avail_date,
                'beta': b,
                'r2': r2,
                'score': b * r2
            })

    df = pd.DataFrame(records)
    if df.empty:
        print("[错误] 无因子记录")
        return df
    print(f"      {len(df):,} 条, {df['stock_code'].nunique()} 只")

    print("[3] 读取K线，构建交易日索引...")
    kline = pd.read_csv(kline_file)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['date', 'stock_code']).reset_index(drop=True)
    all_trade_dates = sorted(kline['date'].drop_duplicates())
    all_sc = sorted(kline['stock_code'].drop_duplicates())
    print(f"      {len(all_trade_dates)} 交易日, {len(all_sc)} 只")

    print("[4] 对齐报告期到交易日 + 前向填充 (60天)...")
    REPORT_LAG = 45
    df['trade_date'] = df['avail_date'] + pd.Timedelta(days=REPORT_LAG)
    df = df.drop_duplicates(subset=['stock_code', 'trade_date'], keep='last')

    # 构建完整日 × 股 表
    full_index = pd.MultiIndex.from_product([all_trade_dates, all_sc], names=['date', 'stock_code'])
    full_df = pd.DataFrame(index=full_index).reset_index()

    # merge factor records
    factor_merge = df[['trade_date', 'stock_code', 'score', 'beta', 'r2']].rename(
        columns={'trade_date': 'date', 'score': 'factor_raw'})
    full_df = pd.merge(full_df, factor_merge, on=['date', 'stock_code'], how='left')

    # 前向填充每个股票
    full_df['factor_raw'] = full_df.groupby('stock_code')['factor_raw'].ffill(limit=60)
    full_df = full_df.dropna(subset=['factor_raw'])
    print(f"      扩展后: {len(full_df):,} 条")

    print("[5] 成交额中性化 (OLS + MAD + z-score)...")
    amt_20d = kline.groupby('stock_code').apply(
        lambda x: x.set_index('date')['amount'].rolling(20, min_periods=10).mean()
    ).reset_index()
    amt_20d.columns = ['stock_code', 'date', 'amt_ma20']
    amt_20d['log_amount_20d'] = np.log(amt_20d['amt_ma20'].clip(lower=1))

    merged = pd.merge(full_df[['date', 'stock_code', 'factor_raw']],
                       amt_20d[['date', 'stock_code', 'log_amount_20d']],
                       on=['date', 'stock_code'], how='inner')
    merged = merged.dropna(subset=['factor_raw', 'log_amount_20d'])

    result_records = []
    dates = sorted(merged['date'].unique())
    for i, td in enumerate(dates):
        sub = merged[merged['date'] == td].copy()
        if len(sub) < 50:
            continue
        v = sub['factor_raw'].values.astype(float)
        x = sub['log_amount_20d'].values.astype(float)
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
            r_mad = (resid - med) / (1.4826 * mad)
            r_clip = np.clip(r_mad, -5, 5)
            std = r_clip.std()
            if std < 1e-10: continue
            z = (r_clip - r_clip.mean()) / std
            scs = sub['stock_code'].values[mask]
            for sc, val in zip(scs, z):
                result_records.append({'date': td, 'stock_code': sc, 'factor': float(val)})
        except Exception:
            continue
        if i % 80 == 0:
            print(f"      [{i}/{len(dates)}] ({i/len(dates)*100:.0f}%)")

    factor_df = pd.DataFrame(result_records)
    if factor_df.empty:
        print("[错误] 因子为空")
        return factor_df

    # 前向填充，防止多出的报告期空白
    factor_df = factor_df.sort_values(['stock_code', 'date'])
    factor_df['factor'] = factor_df.groupby('stock_code')['factor'].ffill(limit=90)

    print(f"\n[完成] {len(factor_df):,} 条, {factor_df['stock_code'].nunique()} 只")
    print(f"        日期: {factor_df['date'].min().date()} ~ {factor_df['date'].max().date()}")

    factor_df.to_csv(output_file, index=False)
    print(f"[完成] 保存: {output_file}")
    return factor_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fundamental', default='data/csi1000_fundamental_cache.csv')
    parser.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    parser.add_argument('--output', default='data/factor_roe_trend_quality_v2.csv')
    args = parser.parse_args()
    compute_factor(args.fundamental, args.kline, args.output)
