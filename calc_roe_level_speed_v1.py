"""
因子：ROE水平速度 (ROE Level Speed)
ID: roe_level_speed_v1
Barra: Growth

逻辑升级（相比 roe_acceleration_v1）：
  单一加速度受小市值噪音干扰。结合ROE水平后形成复合因子：
  signal = ROE_MA4(current) × accel / std(ROE_MA4历史)
  
  即：把加速度绝对值归一化到ROE水平上 → 相同加速度下，ROE高者得分更高
  
  High score = 高ROE + 持续加速 = growth + quality combo
  Low score  = 低ROE + 持续恶化 = value trap or distress

  这样区分加速的"高质量加速"（高ROE上升）和"低质量加速"（低ROE上升）；
  以及"高ROE恶化" vs "低ROE恶化"的差异。
  
  中性化：成交额OLS中性化 + MAD + z-score
"""

import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')


def compute_factor(fund_file='data/csi1000_fundamental_cache.csv',
                   kline_file='data/csi1000_kline_raw.csv',
                   output_file='data/factor_roe_level_speed_v1.csv',
                   cross_sectional_power=1.5):  # 指数加权参数 1-2
    """计算ROE水平速度因子"""
    # ── 读取 ──────────────────────────────────────────────────────────────────
    fund = pd.read_csv(fund_file)
    fund['report_date'] = pd.to_datetime(fund['report_date'])
    fund = fund.dropna(subset=['roe']).sort_values(['stock_code', 'report_date'])
    fund = fund.drop_duplicates(subset=['stock_code', 'report_date']).reset_index(drop=True)

    kline = pd.read_csv(kline_file)
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values(['date', 'stock_code']).reset_index(drop=True)

    # ── 计算ROE_MA4和加速度 ───────────────────────────────────────────────────
    fund['roe_ma4'] = (
        fund.groupby('stock_code')['roe']
        .transform(lambda x: x.rolling(4, min_periods=3).mean())
    )
    fund['prev_ma4'] = fund.groupby('stock_code')['roe_ma4'].shift(1)
    fund['accel_raw'] = fund['roe_ma4'] - fund['prev_ma4']
    fund = fund.dropna(subset=['accel_raw', 'roe_ma4']).copy()

    # ── 复合信号: accel × sign(ROE_MA4) × |ROE_MA4|^power ──────────────────────
    # 加速应从当前水平向上：high ROE + accelerating = best
    # 加速但从低ROE = 相对marginal improvement
    # 绝对信号：|ROE_MA4|^power * accel / (std_accel_item +  epsilon)
    # 逻辑：加速度越大（正方向）且ROE水平越高 → 因子值越高
    
    # Per-stock normalization of accel
    fund['accel_raw_z'] = fund.groupby('stock_code')['accel_raw'].transform(
        lambda x: (x - x.mean()) / (x.std() + 0.01)
    )
    
    # Composite: log(1+|ROE_MA4|) × accel_raw_z
    # Level-adjusts acceleration so that same acceleration high-ROE stocks ranked higher
    fund['composite_raw'] = np.log1p(fund['roe_ma4'].abs()) * fund['accel_raw_z']
    
    # Clip extreme values per stock (winsorize at 5%/95%)
    fund['composite_clip'] = fund.groupby('stock_code')['composite_raw'].transform(
        lambda x: x.clip(x.quantile(0.05), x.quantile(0.95))
    )

    # ── 映射到交易日 ──────────────────────────────────────────────────────────
    REPORT_LAG = 45
    fund['avail_date'] = fund['report_date'] + pd.Timedelta(days=REPORT_LAG)
    fund_use = fund[['stock_code', 'avail_date', 'composite_clip']].dropna(subset=['composite_clip'])
    fund_use = fund_use.drop_duplicates(subset=['stock_code', 'avail_date'], keep='last')
    k_expanded = kline[['date', 'stock_code']].drop_duplicates().sort_values(['stock_code', 'date'])

    # asof align
    records = []
    for sc in sorted(fund_use['stock_code'].unique()):
        f_sc = fund_use[fund_use['stock_code']==sc][['avail_date','composite_clip']].sort_values('avail_date')
        k_sc = k_expanded[k_expanded['stock_code']==sc].sort_values('date')
        if len(f_sc) < 3 or len(k_sc) < 20: continue
        m = pd.merge_asof(k_sc, f_sc.rename(columns={'avail_date':'date'}), on='date', direction='backward')
        valid = m.dropna(subset=['composite_clip'])
        if len(valid) < 15: continue
        records.append(valid[['date','stock_code','composite_clip']])

    raw_factor = pd.concat(records, ignore_index=True)
    print(f"  原始信号记录: {len(raw_factor):,} 条, {raw_factor['date'].nunique()} 个交易日, {raw_factor['stock_code'].nunique()} 只股票")

    # ── 成交额中性化 + MAD + z-score ──────────────────────────────────────────
    # Compute log_amount_20d
    amt_20d = (
        kline.sort_values(['stock_code', 'date'])
        .groupby(['stock_code', 'date'])['amount']
        .last().reset_index()
    )
    amt_20d['log_amount_20d'] = (
        amt_20d.groupby('stock_code')['amount']
        .transform(lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1))
    )

    f_pivot = raw_factor.pivot_table(index='date', columns='stock_code', values='composite_clip')
    a_pivot = amt_20d.pivot_table(index='date', columns='stock_code', values='log_amount_20d')
    common = f_pivot.index.intersection(a_pivot.index).sort_values()
    print(f"  对齐交易日: {len(common)}")

    result = []
    for i, td in enumerate(common):
        fv = f_pivot.loc[td].dropna()
        av = a_pivot.loc[td]
        scs = fv.index.intersection(av.index)
        if len(scs) < 50: continue

        v = fv[scs].values.astype(float)
        x = av[scs].values.astype(float)
        mask = np.isfinite(v) & np.isfinite(x)
        if mask.sum() < 50: continue

        vm, xm = v[mask], x[mask]
        X = np.column_stack([np.ones(len(xm)), xm])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, vm, rcond=None)
            resid = vm - X @ beta
            med = np.median(resid)
            mad = np.median(np.abs(resid - med)) + 1e-10
            r_mad = (resid - med) / (1.4826 * mad)
            r_clip = np.clip(r_mad, -5, 5)
            std = r_clip.std()
            if std < 1e-10: continue
            z = (r_clip - r_clip.mean()) / std
            for sc, val in zip(scs[mask], z):
                result.append({'date': td, 'stock_code': sc, 'factor': float(val)})
        except: continue

    factor_df = pd.DataFrame(result)
    if factor_df.empty:
        print("[错误] 因子为空"); return factor_df

    factor_df = factor_df.sort_values(['stock_code', 'date'])
    factor_df['factor'] = factor_df.groupby('stock_code')['factor'].ffill(limit=5)
    all_dates = sorted(factor_df['date'].unique())
    if len(all_dates) > 5:
        cutoff = all_dates[-5]
        factor_df = factor_df[factor_df['date'] <= cutoff]

    print(f"\n[完成] {len(factor_df):,} 条, {factor_df['date'].nunique()} 个交易日, "
          f"均值={factor_df['factor'].mean():.4f}, std={factor_df['factor'].std():.4f}")
    factor_df.to_csv(output_file, index=False)
    print(f"[完成] 保存 {output_file}")
    return factor_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fundamental', default='data/csi1000_fundamental_cache.csv')
    parser.add_argument('--kline', default='data/csi1000_kline_raw.csv')
    parser.add_argument('--output', default='data/factor_roe_level_speed_v1.csv')
    args = parser.parse_args()
    compute_factor(args.fundamental, args.kline, args.output)
