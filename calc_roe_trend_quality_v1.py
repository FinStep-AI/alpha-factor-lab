"""
因子：ROE时序趋势质量 v1
ID: roe_trend_quality_v1
Barra: Growth

构造逻辑：
  - 对每只股票，取最近12个季度的ROE序列
  - 做线性回归: ROE_i = α + β×t + ε
  - β = ROE趋势斜率（向上/下倾斜程度）
  - R² = ROE趋势稳定性（拟合优度）
  - score = β × R²（趋势强且一致 → 高质量成长信号）

中性化：成交额OLS中性化 + MAD去极值 + z-score
"""

import pandas as pd
import numpy as np
import argparse
import warnings
from numpy.polynomial.polynomial import polyfit
warnings.filterwarnings('ignore')


def compute_factor(fund_file='data/csi1000_fundamental_cache.csv',
                   kline_file='data/csi1000_kline_raw.csv',
                   output_file='data/factor_roe_trend_quality_v1.csv',
                   window_quarters=12):
    """计算ROE时序趋势因子。"""

    print("[1] 读取基本面数据...")
    fund = pd.read_csv(fund_file)
    fund['report_date'] = pd.to_datetime(fund['report_date'])
    fund = fund.dropna(subset=['roe', 'report_date']).copy()
    fund = fund.sort_values(['stock_code', 'report_date']) \
               .drop_duplicates(subset=['stock_code', 'report_date'], keep='first') \
               .reset_index(drop=True)

    # 只保留了 report_date 足够老的数据来做 window=12
    print(f"     原始: {fund['stock_code'].nunique()} 只, {fund['report_date'].min().date()} ~ {fund['report_date'].max().date()}")

    print("[2] 计算每只股票每季的ROE时序质量...")
    records = []
    for sc, grp in fund.groupby('stock_code'):
        g = grp.sort_values('report_date').reset_index(drop=True)
        n = len(g)
        if n < window_quarters:
            continue
        # 滚动窗口: 12个季度 → 趋势斜率 × R²
        for i in range(window_quarters - 1, n):
            win = g.iloc[i - window_quarters + 1:i + 1]
            ts_num = np.arange(window_quarters)
            roes = win['roe'].values.astype(float)

            if np.isnan(roes).sum() > 3:
                continue

            # 线性回归: y = a + b*t
            valid = np.isfinite(roes)
            if valid.sum() < 6:
                continue
            t = ts_num[valid]
            y = roes[valid]
            b, a = np.polyfit(t, y, 1)  # b=slope, a=intercept

            # R² = (cov(t,y)/σ_t/σ_y)²
            y_pred = a + b * t
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

            score = b * r2  # 方向×一致性

            avail_date = win['report_date'].iloc[-1]
            records.append({
                'stock_code': sc,
                'avail_date': avail_date,
                'beta': b,
                'r2': r2,
                'score': score
            })

    df = pd.DataFrame(records)
    if df.empty:
        print("[错误] 没有计算到任何因子值")
        return df

    print(f"     原始记录: {len(df):,} 条, {df['stock_code'].nunique()} 只")

    print("[3] 报告日延迟 → 交易日映射 (45天)...")
    REPORT_LAG = 45
    df['trade_date'] = df['avail_date'] + pd.Timedelta(days=REPORT_LAG)
    df = df.drop_duplicates(subset=['stock_code', 'trade_date'], keep='last')

    print("[4] 读取K线数据，计算成交额中性化...")
    kline = pd.read_csv(kline_file)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['date', 'stock_code']).reset_index(drop=True)

    # 计算20日平均成交额(log)
    amt_20d = kline.groupby('stock_code').apply(
        lambda x: x.set_index('date')['amount'].rolling(20, min_periods=10).mean()
    ).reset_index()
    amt_20d.columns = ['stock_code', 'date', 'amt_ma20']
    amt_20d['log_amount_20d'] = np.log(amt_20d['amt_ma20'].clip(lower=1))

    print("[5] 对齐数据...")
    df = df.rename(columns={'trade_date': 'date', 'score': 'factor_raw'})
    merged = pd.merge(df[['date', 'stock_code', 'factor_raw']],
                       amt_20d[['date', 'stock_code', 'log_amount_20d']],
                       on=['date', 'stock_code'], how='inner')
    merged = merged.dropna(subset=['factor_raw', 'log_amount_20d'])

    print(f"     对齐后: {len(merged):,} 条, {merged['stock_code'].nunique()} 只")

    print("[6] 截面OLS中性化 + MAD缩尾 + z-score...")
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
            # OLS: v = β0 + β1 * x
            X = np.column_stack([np.ones(len(x2)), x2])
            beta, _, _, _ = np.linalg.lstsq(X, v2, rcond=None)
            resid = v2 - X @ beta

            # MAD
            med = np.median(resid)
            mad = np.median(np.abs(resid - med)) + 1e-10
            r_mad = (resid - med) / (1.4826 * mad)
            r_clip = np.clip(r_mad, -5, 5)

            std = r_clip.std()
            if std < 1e-10:
                continue
            z = (r_clip - r_clip.mean()) / std

            scs = sub['stock_code'].values[mask]
            for sc, val in zip(scs, z):
                result_records.append({'date': td, 'stock_code': sc, 'factor': float(val)})
        except Exception as e:
            continue

        if i % 60 == 0:
            print(f"      [{i}/{len(dates)}] ({i/len(dates)*100:.0f}%)")

    factor_df = pd.DataFrame(result_records)
    if factor_df.empty:
        print("[错误] 因子为空")
        return factor_df

    # 前向填充：每个报告日因子值向前延伸至下一个报告日
    factor_df = factor_df.sort_values(['stock_code', 'date'])
    factor_df['date'] = pd.to_datetime(factor_df['date'])

    # 用 kline 所有交易日做索引扩展
    all_trade_dates = sorted(kline['date'].drop_duplicates())
    all_sc = sorted(kline['stock_code'].drop_duplicates())
    idx = pd.MultiIndex.from_product([all_trade_dates, all_sc], names=['date', 'stock_code'])
    full = pd.DataFrame(index=idx).reset_index()
    full = pd.merge(full, factor_df, on=['date', 'stock_code'], how='left')
    full['factor'] = full.groupby('stock_code')['factor'].ffill(limit=120)  # 最多延伸120天
    full = full.dropna(subset=['factor'])

    print(f"\n[完成] 因子记录: {len(full):,} 条")
    print(f"        日期区间: {full['date'].min().date()} ~ {full['date'].max().date()}")
    print(f"        股票覆盖: {full['stock_code'].nunique()} 只")
    print(f"        均值={full['factor'].mean():.4f}, std={full['factor'].std():.4f}")

    full[['date', 'stock_code', 'factor']].to_csv(output_file, index=False)
    print(f"[完成] 保存: {output_file}")
    return full[['date', 'stock_code', 'factor']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fundamental', default='data/csi1000_fundamental_cache.csv')
    parser.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    parser.add_argument('--output', default='data/factor_roe_trend_quality_v1.csv')
    parser.add_argument('--window', type=int, default=12, help='滚动季度窗口')
    args = parser.parse_args()
    compute_factor(args.fundamental, args.kline, args.output, args.window)
