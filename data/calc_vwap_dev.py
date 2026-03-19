"""
VWAP偏离度因子 (vwap_dev_v1)
==============================
close/(amount/volume/100) → close相对VWAP的偏离
20日均值，反向使用，市值中性化
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

kline = pd.read_csv('data/csi1000_kline_raw.csv')
kline = kline.sort_values(['stock_code', 'date']).reset_index(drop=True)

# VWAP = amount / (volume * 100)  (volume是手，1手=100股)
kline['vwap'] = kline['amount'] / (kline['volume'] * 100)
kline.loc[(kline['volume'] == 0) | (kline['vwap'] <= 0), 'vwap'] = np.nan

# close/vwap - 1 → 偏离度
kline['close_vwap_dev'] = kline['close'] / kline['vwap'] - 1

print("Close/VWAP deviation stats:")
print(kline['close_vwap_dev'].describe())
print(f"Mean: {kline['close_vwap_dev'].mean():.6f}")

# clip极端值
kline['close_vwap_dev'] = kline['close_vwap_dev'].clip(-0.05, 0.05)

# 20日滚动均值
kline['vwap_dev_20'] = kline.groupby('stock_code')['close_vwap_dev'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

# 反向：收盘持续低于VWAP → 可能是打压吸筹 → 做多
# 但也试正向看看
kline['factor_raw'] = -kline['vwap_dev_20']  # 先试反向

# 市值中性化
kline['log_amount'] = np.log1p(kline['amount'])
kline['ma20_log_amount'] = kline.groupby('stock_code')['log_amount'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

def neutralize(group):
    y = group['factor_raw']
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
daily_factor = kline.groupby('date', group_keys=False).apply(neutralize)

output = daily_factor[['date', 'stock_code', 'factor_neutral']].dropna()
output.columns = ['date', 'stock_code', 'factor_value']
output = output.sort_values(['date', 'stock_code']).reset_index(drop=True)

print(f"输出: {output.shape}, 日期: {output['date'].min()} ~ {output['date'].max()}")
print(f"每日股票数: {output.groupby('date')['stock_code'].count().mean():.0f}")
print(output['factor_value'].describe())
output.to_csv('data/factor_vwap_dev.csv', index=False)
print("✅ 已保存 data/factor_vwap_dev.csv")
