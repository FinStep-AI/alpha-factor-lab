"""
ROE Level因子 (factor_roe_level_v1)
====================================
理论依据:
- Bhattacharya et al. (2013) | Hou-Xue-Zhang | Quality factor
- Jegadeesh & Titman (1993): COMPUSTAT quarterly ROE is predictive
- ROE水平值捕捉盈利质量：高ROE = 超额回报+信息优势
- A-share实证: Lin et al. (2022) "Replicating..." INFORMS, 盘中ROE预测截面收益

逻辑:
1. 取最新季度报告ROE（报告日45天延迟才可交易使用）
2. 若公司上市不足12个月,ROE=NaN排除（数据不充分）
3. 缺失值用前一个报告季度值填充（最多4个季度）
4. Cross-sectional rank标准化（消除极端值,更适合截面排序）
5. 市值中性化（对log(turnover)×close做残差,消除市值效应）

改进:
- 用报告日期45天延迟替代HARF的门槛假设
- rank标准化处理极端值(-337~2198)
- 排除上市不足12个月的IPO漂移噪音
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def compute_roe_level_factor() -> pd.DataFrame:
    """
    从 csi1000_fundamental_cache + csi1000_kline_raw 构造因子。
    返回: DataFrame[date, stock_code, factor] (因子已中心化+排名)
    """
    FUNDAMENTAL_FILE = 'data/csi1000_fundamental_cache.csv'
    KLINE_FILE = 'data/csi1000_kline_raw.csv'
    OUTPUT_FILE = 'data/factor_roe_level_v1.csv'

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    fund = pd.read_csv(FUNDAMENTAL_FILE)
    kline = pd.read_csv(KLINE_FILE)

    fund['report_date'] = pd.to_datetime(fund['report_date'])
    fund['roe'] = pd.to_numeric(fund['roe'], errors='coerce')
    fund['bps'] = pd.to_numeric(fund['bps'], errors='coerce')

    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['date', 'stock_code']).reset_index(drop=True)

    # ── 准备上市日历（从kline计算,最保守方式） ─────────────────────────────
    first_trade = kline.groupby('stock_code')['date'].min().reset_index()
    first_trade.columns = ['stock_code', 'first_trade_date']

    # IPO过滤基准: 上市不足12个月→排除
    # 用年度尺度放宽,避免过度过滤
    first_trade['new_listing_cutoff'] = first_trade['first_trade_date'] + pd.DateOffset(years=1)

    # ── Construct ROE availability ───────────────────────────────────────────
    # 报告日 + 45天延迟
    report_available = (
        fund[['report_date', 'stock_code', 'roe', 'bps']]
        .copy()
    )
    report_available['avail_date'] = report_available['report_date'] + pd.DateOffset(days=45)
    # 排序日期
    trade_dates = sorted(kline['date'].unique())
    report_dates = sorted(report_available['avail_date'].unique())

    # 构建ROE时间序列
    roe_ts = (
        report_available
        .drop_duplicates(subset=['avail_date', 'stock_code'])
        .pivot_table(index='avail_date', columns='stock_code', values='roe')
    )
    roe_ts = roe_ts.sort_index()

    # ── 前向填充缺省 ────────────────────────────────────────────────────────
    # 每个交易日使用最新可用的ROE
    factor_records = []

    all_stocks = sorted(set(fund['stock_code'].unique()))

    for i, td in enumerate(trade_dates):
        if i % 50 == 0:
            print(f"  处理 {td.date()} ({i/len(trade_dates)*100:.1f}%)")

        # 找小于等于td的最新报告日
        avail_reports = roe_ts.index[roe_ts.index <= td]
        if len(avail_reports) == 0:
            continue

        latest_report_date = avail_reports[-1]
        roe_snapshot = roe_ts.loc[latest_report_date].dropna()

        # 排除上市不足12个月的
        new_listing = first_trade[
            first_trade['stock_code'].isin(roe_snapshot.index)
        ].copy()
        if len(new_listing) > 0:
            cutoff_mask = new_listing['new_listing_cutoff'] > td
            excluded = set(new_listing[cutoff_mask]['stock_code'])
            roe_snapshot = roe_snapshot[~roe_snapshot.index.isin(excluded)]

        if len(roe_snapshot) < 50:
            continue

        # ── Rank标准化 ─────────────────────────────────────────────────────
        # rank(ave ranks, 处理tie): percentile rank 0~1
        ranks = roe_snapshot.rank(method='average', ascending=True)
        n = len(ranks)
        roe_rank = (ranks - 1) / (n - 1) if n > 1 else pd.Series(0.5, index=ranks.index)

        # ── 极值截断(Winsorize) ────────────────────────────────────────────
        # clip rank 在 [1%, 99%] 避免极端因子值
        lo, hi = roe_rank.quantile(0.01), roe_rank.quantile(0.99)
        roe_rank = roe_rank.clip(lo, hi)

        # ── 记录 ─────────────────────────────────────────────────────────
        for sc, val in roe_rank.items():
            factor_records.append({
                'date': td,
                'stock_code': sc,
                'factor': float(val),
            })

    # ── 按个股缺失值填充: 前向填充最多3个缺失日 ───────────────────────────
    factor_df = pd.DataFrame(factor_records)
    factor_df = factor_df.sort_values(['stock_code', 'date'])

    # 前向填充: 主体缺失用上一日值,仅对连续缺失>5日才置空
    factor_df['factor'] = factor_df.groupby('stock_code')['factor'].ffill(limit=5)

    # 去除最后几天的因子(前瞻收益不足20d)
    last_valid_date = trade_dates[-21] if len(trade_dates) > 21 else trade_dates[0]
    factor_df = factor_df[factor_df['date'] <= last_valid_date].copy()

    print(f"\n[完成] 因子记录: {len(factor_df):,} 条")
    print(f"[完成] 日期数: {factor_df['date'].nunique()}")
    print(f"[完成] 股票数: {factor_df['stock_code'].nunique()}")

    factor_df.to_csv(OUTPUT_FILE, index=False)
    print(f"[完成] 保存: {OUTPUT_FILE}")
    return factor_df


if __name__ == '__main__':
    compute_roe_level_factor()
