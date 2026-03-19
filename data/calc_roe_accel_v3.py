"""
ROE改善加速度因子 v3 (roe_accel_v3)
====================================
改进：2期加速度 = delta-ROE(t) - delta-ROE(t-2)
平滑单期噪音，更稳定的加速信号
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

fund = pd.read_csv('data/csi1000_fundamental_cache.csv')
kline = pd.read_csv('data/csi1000_kline_raw.csv')

fund = fund.sort_values(['stock_code', 'report_date']).reset_index(drop=True)
fund['roe'] = pd.to_numeric(fund['roe'], errors='coerce')
roe_low, roe_high = fund['roe'].quantile(0.01), fund['roe'].quantile(0.99)
fund['roe'] = fund['roe'].clip(roe_low, roe_high)

roe_pivot = fund.pivot_table(index='stock_code', columns='report_date', values='roe')
delta_roe = roe_pivot.diff(periods=4, axis=1)
# 2期加速度
accel_roe = delta_roe.diff(periods=2, axis=1)

report_dates = sorted(fund['report_date'].unique())
trade_dates = sorted(kline['date'].unique())

report_to_available = {
    '03-31': (0, '05-01'),
    '06-30': (0, '09-01'),
    '09-30': (0, '11-01'),
    '12-31': (1, '05-01'),
}

factor_records = []
for rd in report_dates:
    mm_dd = rd[5:]
    year = int(rd[:4])
    if mm_dd not in report_to_available:
        continue
    year_offset, avail_mmdd = report_to_available[mm_dd]
    avail_date = f"{year + year_offset}-{avail_mmdd}"
    if rd not in accel_roe.columns:
        continue
    vals = accel_roe[rd].dropna()
    if len(vals) < 100:
        continue
    print(f"  {rd} → {avail_date}, N={len(vals)}")
    for sc, v in vals.items():
        factor_records.append({
            'report_date': rd, 'available_date': avail_date,
            'stock_code': sc, 'raw_factor': v
        })

factor_df = pd.DataFrame(factor_records)

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
print(f"日频: {len(daily_factor)}")

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
print(f"输出: {output.shape}, {output['date'].min()} ~ {output['date'].max()}")
output.to_csv('data/factor_roe_accel_v3.csv', index=False)
print("✅")
