"""
BPS YoY 增长率因子 — 向量化版 v2
Hypothesis: Low asset growth (BPS YoY proxy) → high future returns
    Cooper, Gulen & Schill (2008) asset growth anomaly
    q-factor investment factor (Hou-Xue-Zhang, 2015)
High BPS growth (aggressive investment) → subsequent over-valuation → lower returns
45-day reporting lag → cross-sectional OLS neutralization + MAD + z-score
"""
import pandas as pd
import numpy as np, warnings; warnings.filterwarnings('ignore')

KEYS = {'amihud_illiq_v2', 'shadow_pressure_v1', 'overnight_momentum_v1',
        'gap_momentum_v1', 'tail_risk_cvar_v1', 'turnover_level_v1',
        'tae_v1', 'amp_level_v2', 'ma_disp_v1', 'vol_cv_neg_v1',
        'turnover_decel_v1', 'informed_flow_v1', 'price_mom_5d_v1',
        'vol_ret_align_v1', 'vwap_dev_v1', 'vol_log60d_v4', 'vssignal_v1',
        'roe_persistence_neg_v1', 'range_efficiency_v1', 'corr_gspd_v1',
        'pv_corr_v1', 'pv_divergence_v1', 'amp_compress_v1', 'vol_ext_cm_v1'}

print("=== BPS YoY 增长率因子 ===")

# ── 数据加载 ──
print("[1] Loading...")
kline = pd.read_csv('data/csi1000_kline_raw.csv', parse_dates=['date'])
kline['stock_code'] = kline['stock_code'].astype(str).str.zfill(6)
fund = pd.read_csv('data/csi1000_fundamental_cache.csv', parse_dates=['report_date'])
fund['stock_code'] = fund['stock_code'].astype(str).str.zfill(6)
fund = fund.dropna(subset=['bps']).drop_duplicates(['stock_code', 'report_date']).sort_values('report_date').reset_index(drop=True)
fund['avail_date'] = (fund['report_date'] + pd.Timedelta(days=45)).dt.normalize()
all_dates = sorted(pd.to_datetime(kline['date'].dt.normalize().unique()))
all_stocks = sorted(kline['stock_code'].unique())

# ── 季度 BPS YoY (4个季度同比) ──
print("[2] Quarterly BPS YoY (4Q)...")
bps_pivot = fund.pivot_table(index='stock_code', columns='report_date', values='bps')
bps_4q = bps_pivot.shift(4, axis=1)
bps_yoy = bps_pivot / bps_4q - 1          # (bps_t / bps_{t-4}) - 1
bps_yoy = bps_yoy.where((bps_4q > 0) & bps_pivot.notna())

print(f"   Quarters: {bps_yoy.shape[1]}, Stocks: {bps_yoy.shape[0]}")

# bpst[date] → bps_yoy for all stocks  (rows = quarter_dates, cols = stock_code)
bpst = bps_yoy.T.copy()
bpst.index = pd.to_datetime(bpst.index).normalize()

# ── 展开到日频 ──
print("[3] Expanding quarterly → daily...")
# 用季度报告可用日（45天滞后）而不是 report_date，避免前视偏差
avail_map = (
    fund[['report_date', 'avail_date']]
    .drop_duplicates()
    .sort_values('report_date')
    .set_index('report_date')['avail_date']
)
bpst.index = pd.to_datetime(avail_map.loc[bpst.index].values).normalize()
q_dates = pd.DatetimeIndex(bpst.index)
q_idx = {date: q_dates.searchsorted(date, side='right') - 1 for date in all_dates}
valid_q_dates = [(d, qi) for d, qi in q_idx.items() if qi >= 0]

print(f"   Valid trading days with quarterly data: {len(valid_q_dates)}")

# 构建日频截面
k_by_date = kline[['date', 'stock_code', 'amount']].drop_duplicates(['date', 'stock_code']).set_index(
    ['date', 'stock_code'])

rows = []
for date, qi in valid_q_dates:
    vals = bpst.iloc[qi]
    k_d = k_by_date.loc[date]
    common = vals.index.intersection(k_d.index)
    good = common[vals[common].notna() & vals[common].between(-4.99, 4.99)]
    if len(good) < 10:
        continue
    for sc in good:
        rows.append({'date': date, 'stock_code': sc,
                     'bps_growth': float(vals[sc]),
                     'amount': float(k_d.loc[sc, 'amount'])})

print(f"   Raw daily rows: {len(rows)}")
raw = pd.DataFrame(rows)

# ── 中性化 + 标准化 ──
print("[4] Cross-sectional OLS neutralize...")
raw['log_amount'] = np.log(raw['amount'].clip(lower=1))

out_rows, n_cs = [], 0
for date, g in raw.groupby('date'):
    g = g.dropna(subset=['bps_growth', 'log_amount'])
    if len(g) < 10:
        continue
    X = np.column_stack([np.ones(len(g)), g['log_amount'].values])
    y = g['bps_growth'].values
    try:
        r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    except:
        continue
    med, mad = np.median(r), np.median(np.abs(r - np.median(r)))
    if mad < 1e-10:
        continue
    r = np.clip(r, med - 5.2*mad, med + 5.2*mad)
    m, s = r.mean(), r.std()
    if s < 1e-10:
        continue
    z = (r - m) / s
    for (_, row), zi in zip(g.iterrows(), z.flat):
        out_rows.append({'date': date, 'stock_code': row['stock_code'], 'factor_value': float(zi)})
    n_cs += 1

print(f"   Neutralized cross-sections: {n_cs}")

out = pd.DataFrame(out_rows)
print(f"   Final rows: {len(out)} | Stocks: {out['stock_code'].nunique()} | "
      f"Dates: {out['date'].min().date()} ~ {out['date'].max().date()}")

out[['date', 'stock_code', 'factor_value']].to_csv('data/factor_bps_growth_v1.csv', index=False)
print("Saved: data/factor_bps_growth_v1.csv")
print(out['factor_value'].describe())
print("=== Done ===")
