"""
ROE改善加速度因子 (roe_accel_v1)
=================================
逻辑：
1. 取每个报告期的ROE
2. 计算同比delta-ROE = ROE(t) - ROE(t-4)，消除季节性
3. 计算加速度 = delta-ROE(t) - delta-ROE(t-1)，即环比变化
4. 将季度数据映射到日频（财报公布后至下一期间持续有效）
5. 市值中性化

信号：加速度>0说明盈利改善在加速（或亏损在减速），正向因子
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 1. 加载数据
fund = pd.read_csv('data/csi1000_fundamental_cache.csv')
kline = pd.read_csv('data/csi1000_kline_raw.csv')

print(f"基本面数据: {fund.shape}, K线数据: {kline.shape}")

# 2. 对报告期排序
fund = fund.sort_values(['stock_code', 'report_date']).reset_index(drop=True)

# 处理ROE的NaN和异常值
fund['roe'] = pd.to_numeric(fund['roe'], errors='coerce')
# Winsorize极端值
roe_low, roe_high = fund['roe'].quantile(0.01), fund['roe'].quantile(0.99)
fund['roe'] = fund['roe'].clip(roe_low, roe_high)

# 3. 计算同比delta-ROE
# 每个报告期有个季度标签，同比=往前shift 4个季度
report_dates = sorted(fund['report_date'].unique())
print(f"报告期数: {len(report_dates)}")
print(f"报告期: {report_dates}")

# 构建pivot表
roe_pivot = fund.pivot_table(index='stock_code', columns='report_date', values='roe')
print(f"ROE矩阵: {roe_pivot.shape}")

# 同比变化 (shift 4列 = 4个季度 = 1年)
delta_roe = roe_pivot.diff(periods=4, axis=1)
print(f"delta-ROE前4期为NaN (同比需要去年数据)")

# 加速度 = delta-ROE的环比变化 (shift 1列)
accel_roe = delta_roe.diff(periods=1, axis=1)
print(f"加速度再损失1期")

# 4. 映射到日频
# 财报公布日假设：
# Q1(03-31) → 04-30公布
# Q2(06-30) → 08-31公布  
# Q3(09-30) → 10-31公布
# Q4(12-31) → 04-30(次年)公布
report_to_available = {
    '03-31': (0, '05-01'),   # Q1→5月1日可用
    '06-30': (0, '09-01'),   # Q2→9月1日可用
    '09-30': (0, '11-01'),   # Q3→11月1日可用
    '12-31': (1, '05-01'),   # Q4→次年5月1日可用
}

trade_dates = sorted(kline['date'].unique())
print(f"交易日范围: {trade_dates[0]} ~ {trade_dates[-1]}, 共{len(trade_dates)}天")

# 构建每个交易日可用的最新因子值
factor_records = []

for rd in report_dates:
    mm_dd = rd[5:]  # '06-30'
    year = int(rd[:4])
    
    if mm_dd in report_to_available:
        year_offset, avail_mmdd = report_to_available[mm_dd]
        avail_date = f"{year + year_offset}-{avail_mmdd}"
    else:
        continue
    
    # 获取这个报告期的加速度值
    if rd not in accel_roe.columns:
        continue
    
    vals = accel_roe[rd].dropna()
    if len(vals) == 0:
        continue
    
    print(f"  报告期 {rd} → 可用日期 {avail_date}, 有效股票数: {len(vals)}")
    
    for sc, v in vals.items():
        factor_records.append({
            'report_date': rd,
            'available_date': avail_date,
            'stock_code': sc,
            'raw_factor': v
        })

factor_df = pd.DataFrame(factor_records)
print(f"\n因子记录数: {len(factor_df)}")

# 对每个交易日，找到最近可用的因子值
kline_dates = pd.DataFrame({'date': trade_dates})

# 将available_date映射到交易日
all_factors_daily = []

for td in trade_dates:
    # 找到available_date <= td的最新一批
    available = factor_df[factor_df['available_date'] <= td]
    if len(available) == 0:
        continue
    
    # 每只股票取最新的available_date
    latest = available.sort_values('available_date').groupby('stock_code').tail(1)
    
    for _, row in latest.iterrows():
        all_factors_daily.append({
            'date': td,
            'stock_code': row['stock_code'],
            'factor_value': row['raw_factor']
        })

daily_factor = pd.DataFrame(all_factors_daily)
print(f"日频因子记录数: {len(daily_factor)}")

# 5. 市值中性化
# 用成交额作为市值代理（与之前因子一致）
kline_subset = kline[['date', 'stock_code', 'amount']].copy()
kline_subset['log_amount'] = np.log1p(kline_subset['amount'])

# 计算20日平均成交额作为市值代理
kline_subset = kline_subset.sort_values(['stock_code', 'date'])
kline_subset['ma20_log_amount'] = kline_subset.groupby('stock_code')['log_amount'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

daily_factor = daily_factor.merge(
    kline_subset[['date', 'stock_code', 'ma20_log_amount']], 
    on=['date', 'stock_code'], 
    how='left'
)

# 截面回归中性化
def neutralize(group):
    y = group['factor_value']
    x = group['ma20_log_amount']
    
    mask = y.notna() & x.notna()
    if mask.sum() < 50:
        group['factor_neutral'] = np.nan
        return group
    
    y_clean = y[mask]
    x_clean = x[mask]
    
    # Winsorize
    y_low, y_high = y_clean.quantile(0.01), y_clean.quantile(0.99)
    y_clean = y_clean.clip(y_low, y_high)
    
    # 回归
    x_mat = np.column_stack([np.ones(len(x_clean)), x_clean.values])
    try:
        beta = np.linalg.lstsq(x_mat, y_clean.values, rcond=None)[0]
        residuals = y_clean.values - x_mat @ beta
        # 标准化
        std = np.std(residuals)
        if std > 0:
            residuals = residuals / std
        group.loc[mask, 'factor_neutral'] = residuals
    except:
        group['factor_neutral'] = np.nan
    
    return group

print("市值中性化...")
daily_factor = daily_factor.groupby('date', group_keys=False).apply(neutralize)

# 6. 输出
output = daily_factor[['date', 'stock_code', 'factor_neutral']].dropna()
output.columns = ['date', 'stock_code', 'factor_value']
output = output.sort_values(['date', 'stock_code']).reset_index(drop=True)

print(f"\n最终输出: {output.shape}")
print(f"日期范围: {output['date'].min()} ~ {output['date'].max()}")
print(f"每日平均股票数: {output.groupby('date')['stock_code'].count().mean():.0f}")
print(f"\n因子统计:")
print(output['factor_value'].describe())

output.to_csv('data/factor_roe_accel.csv', index=False)
print("\n✅ 已保存到 data/factor_roe_accel.csv")
