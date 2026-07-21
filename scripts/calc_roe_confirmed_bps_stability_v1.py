import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

fund = pd.read_csv('data/csi1000_fundamental_cache.csv')
kline = pd.read_csv('data/csi1000_kline_raw.csv')

fund = fund.sort_values(['stock_code', 'report_date']).reset_index(drop=True)
fund['roe'] = pd.to_numeric(fund['roe'], errors='coerce')
fund['bps'] = pd.to_numeric(fund['bps'], errors='coerce')

for col in ['roe', 'bps']:
    lo, hi = fund[col].quantile(0.01), fund[col].quantile(0.99)
    fund[col] = fund[col].clip(lo, hi)

roe_pivot = fund.pivot_table(index='stock_code', columns='report_date', values='roe')
bps_pivot = fund.pivot_table(index='stock_code', columns='report_date', values='bps')

roe_yoy = roe_pivot.diff(periods=4, axis=1)
roe_qoq = roe_pivot.diff(periods=1, axis=1)

bps_yoy = bps_pivot.pct_change(periods=4, axis=1).clip(-2, 5)
bps_yoy_lag1 = bps_yoy.shift(1, axis=1)
bps_yoy_lag2 = bps_yoy.shift(2, axis=1)

report_dates = sorted(fund['report_date'].unique())
trade_dates = sorted(kline['date'].unique())

report_to_available = {
    '03-31': (0, '05-01'),
    '06-30': (0, '09-01'),
    '09-30': (0, '11-01'),
    '12-31': (1, '05-01'),
}

def robust_z(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan)
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad == 0:
        std = s.std()
        if pd.isna(std) or std == 0:
            return pd.Series(np.nan, index=s.index)
        z = (s - s.mean()) / std
    else:
        z = (s - med) / (1.4826 * mad)
    return z.clip(-3, 3)

factor_records = []
for rd in report_dates:
    mm_dd = rd[5:]
    if mm_dd not in report_to_available:
        continue
    year = int(rd[:4])
    year_offset, avail_mmdd = report_to_available[mm_dd]
    avail_date = f"{year + year_offset}-{avail_mmdd}"

    cols = [roe_yoy.get(rd), roe_qoq.get(rd), bps_yoy.get(rd), bps_yoy_lag1.get(rd), bps_yoy_lag2.get(rd)]
    if any(c is None for c in cols):
        continue

    df = pd.concat([
        roe_yoy[rd].rename('roe_yoy'),
        roe_qoq[rd].rename('roe_qoq'),
        bps_yoy[rd].rename('bps_yoy'),
        bps_yoy_lag1[rd].rename('bps_yoy_lag1'),
        bps_yoy_lag2[rd].rename('bps_yoy_lag2'),
    ], axis=1).dropna()

    if len(df) < 100:
        continue

    df['bps_accel'] = df['bps_yoy'] - df['bps_yoy_lag1']
    df['bps_stability_penalty'] = (df[['bps_yoy', 'bps_yoy_lag1', 'bps_yoy_lag2']].std(axis=1)).fillna(0)

    score = (
        0.55 * robust_z(df['roe_yoy']) +
        0.25 * robust_z(df['roe_qoq']) +
        0.15 * robust_z(df['bps_accel']) -
        0.25 * robust_z(df['bps_stability_penalty'])
    )
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 100:
        continue

    for sc, v in score.items():
        factor_records.append({
            'report_date': rd,
            'available_date': avail_date,
            'stock_code': sc,
            'raw_factor': v,
        })

factor_df = pd.DataFrame(factor_records)

all_factors_daily = []
for td in trade_dates:
    available = factor_df[factor_df['available_date'] <= td]
    if len(available) == 0:
        continue
    latest = available.sort_values('available_date').groupby('stock_code').tail(1)
    latest = latest[['stock_code', 'raw_factor']].copy()
    latest['date'] = td
    all_factors_daily.append(latest)

daily_factor = pd.concat(all_factors_daily, ignore_index=True)

kline_subset = kline[['date', 'stock_code', 'amount']].copy()
kline_subset['log_amount'] = np.log1p(pd.to_numeric(kline_subset['amount'], errors='coerce'))
kline_subset = kline_subset.sort_values(['stock_code', 'date'])
kline_subset['ma20_log_amount'] = kline_subset.groupby('stock_code')['log_amount'].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

daily_factor = daily_factor.merge(
    kline_subset[['date', 'stock_code', 'ma20_log_amount']],
    on=['date', 'stock_code'], how='left'
)

def neutralize(group: pd.DataFrame) -> pd.DataFrame:
    y = group['raw_factor']
    x = group['ma20_log_amount']
    mask = y.notna() & x.notna()
    group['factor_value'] = np.nan
    if mask.sum() < 50:
        return group
    y_clean = y[mask]
    x_clean = x[mask]
    xmat = np.column_stack([np.ones(len(x_clean)), x_clean.values])
    beta = np.linalg.lstsq(xmat, y_clean.values, rcond=None)[0]
    resid = y_clean.values - xmat @ beta
    std = np.std(resid)
    if std > 0:
        resid = resid / std
    group.loc[mask, 'factor_value'] = resid
    return group

daily_factor = daily_factor.groupby('date', group_keys=False).apply(neutralize)
out = daily_factor[['date', 'stock_code', 'factor_value']].dropna()
out = out.sort_values(['date', 'stock_code']).reset_index(drop=True)
out.to_csv('data/factor_roe_confirmed_bps_stability_v1.csv', index=False)
print(out.head())
print(out['factor_value'].describe())
print(f"saved rows={len(out)} dates={out['date'].min()}~{out['date'].max()}")
