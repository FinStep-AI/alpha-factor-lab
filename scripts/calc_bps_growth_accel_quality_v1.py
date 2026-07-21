
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
fund_path = BASE / 'data' / 'csi1000_fundamental_cache.csv'
kline_path = BASE / 'data' / 'csi1000_kline_raw.csv'
out_path = BASE / 'data' / 'factor_bps_growth_accel_quality_v1.csv'

fund = pd.read_csv(fund_path)
fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
fund['report_date'] = pd.to_datetime(fund['report_date'])
fund = fund.sort_values(['stock_code','report_date']).copy()

# 基本面特征：BPS同比增长、增长加速度、ROE改善
fund['bps_yoy'] = fund.groupby('stock_code')['bps'].pct_change(4)
fund['bps_yoy_prev'] = fund.groupby('stock_code')['bps_yoy'].shift(1)
fund['bps_yoy_accel'] = fund['bps_yoy'] - fund['bps_yoy_prev']
fund['roe_yoy'] = fund.groupby('stock_code')['roe'].diff(4)
fund['roe_qoq'] = fund.groupby('stock_code')['roe'].diff(1)

# 稳定性/确认项：要求BPS增长加速同时ROE没有恶化
# 用tanh压缩极端值，避免小样本季报跳变过大
fund['raw_factor'] = (
    np.tanh(fund['bps_yoy_accel'].fillna(0) * 2.5) *
    (0.6 + 0.4 * np.tanh(fund['roe_yoy'].fillna(0) / 8.0)) *
    (0.7 + 0.3 * np.tanh(fund['roe_qoq'].fillna(0) / 4.0))
)

# 使用最近已披露财报映射到日频
kline = pd.read_csv(kline_path, usecols=['date','stock_code','close'])
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
kline['date'] = pd.to_datetime(kline['date'])
kline = kline.sort_values(['stock_code','date']).copy()

# 用close近似总市值暴露（shares constant within stock），做横截面市值中性化
kline['log_mktcap_proxy'] = np.log(kline['close'].clip(lower=0.1))

merged = pd.merge_asof(
    kline.sort_values('date'),
    fund[['stock_code','report_date','raw_factor']].sort_values('report_date'),
    left_on='date',
    right_on='report_date',
    by='stock_code',
    direction='backward'
)

merged = merged.dropna(subset=['raw_factor','log_mktcap_proxy']).copy()


def neutralize(group: pd.DataFrame) -> pd.DataFrame:
    x = group['log_mktcap_proxy'].to_numpy(dtype=float)
    y = group['raw_factor'].to_numpy(dtype=float)
    if len(group) < 20 or np.nanstd(x) < 1e-8:
        group['factor'] = y - np.nanmean(y)
        return group
    x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
    y_mean = np.nanmean(y)
    beta = np.dot(x, y - y_mean) / (np.dot(x, x) + 1e-12)
    resid = y - y_mean - beta * x
    resid = (resid - np.nanmean(resid)) / (np.nanstd(resid) + 1e-12)
    group['factor'] = resid
    return group

merged = merged.groupby('date', group_keys=False).apply(neutralize)
out = merged[['date','stock_code','factor']].dropna().copy()
out.to_csv(out_path, index=False)
print(f'saved {out_path} rows={len(out)} dates={out.date.nunique()} stocks={out.stock_code.nunique()}')
print(out.head())
