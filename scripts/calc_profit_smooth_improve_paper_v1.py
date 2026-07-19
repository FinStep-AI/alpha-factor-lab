#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parents[1]
FUND = BASE / 'data' / 'csi1000_fundamental_cache.csv'
KLINE = BASE / 'data' / 'csi1000_kline_raw.csv'
OUT = BASE / 'data' / 'factor_profit_smooth_improve_paper_v1.csv'
LAG_DAYS = 45

fund = pd.read_csv(FUND)
kline = pd.read_csv(KLINE)
fund['report_date'] = pd.to_datetime(fund['report_date'])
kline['date'] = pd.to_datetime(kline['date'])
fund['stock_code'] = fund['stock_code'].astype(int)
kline['stock_code'] = kline['stock_code'].astype(int)
fund = fund.sort_values(['stock_code','report_date']).drop_duplicates(['stock_code','report_date'])
kline = kline.sort_values(['stock_code','date']).drop_duplicates(['stock_code','date'])

g = fund.groupby('stock_code')
fund['roe_yoy'] = g['roe'].diff(4)
fund['roe_yoy_ma2'] = g['roe_yoy'].transform(lambda s: s.rolling(2, min_periods=2).mean())
fund['roe_std4'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).std())
fund['roe_level_ma4'] = g['roe'].transform(lambda s: s.rolling(4, min_periods=3).mean())
# 平滑改善 × 正ROE门控 / 波动惩罚
fund['raw'] = fund['roe_yoy_ma2'] / (fund['roe_std4'].abs() + 1.0) * np.tanh(fund['roe_level_ma4'] / 8.0)
fund['avail_date'] = fund['report_date'] + pd.Timedelta(days=LAG_DAYS)
fund = fund.dropna(subset=['raw','avail_date'])

kline['log_amount_20d'] = np.log(
    kline.groupby('stock_code')['amount'].transform(lambda s: s.rolling(20, min_periods=10).mean()).clip(lower=1)
)

records = []
trade_dates = sorted(kline['date'].unique())
for td in trade_dates:
    subf = fund[fund['avail_date'] <= td]
    if subf.empty:
        continue
    latest = subf.sort_values(['stock_code','avail_date']).groupby('stock_code').tail(1)
    daily = kline.loc[kline['date'] == td, ['date','stock_code','log_amount_20d']].merge(
        latest[['stock_code','raw']], on='stock_code', how='left'
    ).dropna(subset=['raw','log_amount_20d'])
    if len(daily) < 30:
        continue
    y = daily['raw'].astype(float).values
    x = daily['log_amount_20d'].astype(float).values
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 20:
        continue
    X = np.column_stack([np.ones(mask.sum()), x[mask]])
    b, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
    resid = np.full_like(y, np.nan, dtype=float)
    resid[mask] = y[mask] - X @ b
    s = pd.Series(resid)
    med = s.median()
    mad = np.median(np.abs(s.dropna() - med)) if s.notna().sum() else np.nan
    if pd.notna(mad) and mad > 1e-12:
        scale = 1.4826 * mad
        s = s.clip(med - 3 * scale, med + 3 * scale)
    mu, sd = s.mean(), s.std()
    if pd.notna(sd) and sd > 1e-12:
        z = (s - mu) / sd
    else:
        z = s * np.nan
    out = pd.DataFrame({
        'date': daily['date'].dt.strftime('%Y-%m-%d'),
        'stock_code': daily['stock_code'].astype(str).str.zfill(6),
        'factor_value': z.round(6)
    }).dropna()
    records.append(out)

res = pd.concat(records, ignore_index=True).drop_duplicates(['date','stock_code'])
OUT.parent.mkdir(parents=True, exist_ok=True)
res.to_csv(OUT, index=False)
print(f'wrote {len(res)} rows to {OUT}')
print(res.head().to_string())
