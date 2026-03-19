"""
质量成长复合因子 (quality_growth_v1)
====================================
逻辑：
1. ROE同比变化(delta_roe) → 盈利趋势信号 
2. BPS同比增速(bps_growth) → 净资产积累信号
3. 复合：rank(delta_roe) + rank(bps_growth) → 等权复合
4. 市值中性化

为什么复合比单因子好：
- delta_roe捕捉盈利改善方向
- bps_growth捕捉净资产积累速度（高留存+高ROE）
- 两者结合=盈利在改善且净资产在快速增长=真正的quality growth
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

fund = pd.read_csv('data/csi1000_fundamental_cache.csv')
kline = pd.read_csv('data/csi1000_kline_raw.csv')

fund = fund.sort_values(['stock_code', 'report_date']).reset_index(drop=True)
fund['roe'] = pd.to_numeric(fund['roe'], errors='coerce')
fund['bps'] = pd.to_numeric(fund['bps'], errors='coerce')

# Winsorize
for col in ['roe', 'bps']:
    lo, hi = fund[col].quantile(0.01), fund[col].quantile(0.99)
    fund[col] = fund[col].clip(lo, hi)

roe_pivot = fund.pivot_table(index='stock_code', columns='report_date', values='roe')
bps_pivot = fund.pivot_table(index='stock_code', columns='report_date', values='bps')

# 同比变化
delta_roe = roe_pivot.diff(periods=4, axis=1)

# BPS同比增速 (百分比)
bps_growth = bps_pivot.pct_change(periods=4, axis=1)
# clip极端BPS增速
bps_growth = bps_growth.clip(-2, 5)

report_dates = sorted(fund['report_date'].unique())
trade_dates = sorted(kline['date'].unique())

report_to_available = {
    '03-31': (0, '05-01'),
    '06-30': (0, '09-01'),
    '09-30': (0, '11-01'),
    '12-31': (1, '05-01'),
}

# 对每个报告期，构建截面rank复合因子
factor_records = []
for rd in report_dates:
    mm_dd = rd[5:]
    year = int(rd[:4])
    if mm_dd not in report_to_available:
        continue
    year_offset, avail_mmdd = report_to_available[mm_dd]
    avail_date = f"{year + year_offset}-{avail_mmdd}"
    
    if rd not in delta_roe.columns or rd not in bps_growth.columns:
        continue
    
    dr = delta_roe[rd].dropna()
    bg = bps_growth[rd].dropna()
    
    # 取交集
    common = dr.index.intersection(bg.index)
    if len(common) < 100:
        continue
    
    dr_common = dr[common]
    bg_common = bg[common]
    
    # 截面rank标准化到[0,1]
    dr_rank = dr_common.rank(pct=True)
    bg_rank = bg_common.rank(pct=True)
    
    # 等权复合
    composite = dr_rank + bg_rank
    
    print(f"  {rd} → {avail_date}, N={len(composite)}, corr(dr,bg)={dr_common.corr(bg_common):.3f}")
    
    for sc, v in composite.items():
        factor_records.append({
            'report_date': rd, 'available_date': avail_date,
            'stock_code': sc, 'raw_factor': v
        })

factor_df = pd.DataFrame(factor_records)

# 映射到日频
all_factors_daily = []
for td in trade_dates:
    available = factor_df[factor_df['available_date'] <= td]
    if len(available) == 0:
        continue
    latest = available.sort_values('available_date').groupby('stock_code').tail(1)
    for _, row in latest.iterrows():
        all_factors_daily.append({
            'date': td, 'stock_code': row['stock_code'], 'factor_value': row['raw_factor']
        })

daily_factor = pd.DataFrame(all_factors_daily)
print(f"日频记录: {len(daily_factor)}")

# 市值中性化
kline_subset = kline[['date', 'stock_code', 'amount']].copy()
kline_subset['log_amount'] = np.log1p(kline_subset['amount'])
kline_subset = kline_subset.sort_values(['stock_code', 'date'])
kline_subset['ma20_log_amount'] = kline_subset.groupby('stock_code')['log_amount'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

daily_factor = daily_factor.merge(
    kline_subset[['date', 'stock_code', 'ma20_log_amount']], 
    on=['date', 'stock_code'], how='left'
)

def neutralize(group):
    y = group['factor_value']
    x = group['ma20_log_amount']
    mask = y.notna() & x.notna()
    if mask.sum() < 50:
        group['factor_neutral'] = np.nan
        return group
    y_clean = y[mask]; x_clean = x[mask]
    y_low, y_high = y_clean.quantile(0.01), y_clean.quantile(0.99)
    y_clean = y_clean.clip(y_low, y_high)
    x_mat = np.column_stack([np.ones(len(x_clean)), x_clean.values])
    try:
        beta = np.linalg.lstsq(x_mat, y_clean.values, rcond=None)[0]
        residuals = y_clean.values - x_mat @ beta
        std = np.std(residuals)
        if std > 0: residuals = residuals / std
        group.loc[mask, 'factor_neutral'] = residuals
    except:
        group['factor_neutral'] = np.nan
    return group

print("市值中性化...")
daily_factor = daily_factor.groupby('date', group_keys=False).apply(neutralize)

output = daily_factor[['date', 'stock_code', 'factor_neutral']].dropna()
output.columns = ['date', 'stock_code', 'factor_value']
output = output.sort_values(['date', 'stock_code']).reset_index(drop=True)

print(f"输出: {output.shape}, 日期: {output['date'].min()} ~ {output['date'].max()}")
print(output['factor_value'].describe())
output.to_csv('data/factor_quality_growth.csv', index=False)
print("✅ 已保存 data/factor_quality_growth.csv")
