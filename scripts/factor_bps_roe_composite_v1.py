#!/usr/bin/env python3
"""
因子  bps_roe_composite_v1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
基本面复合因子: bps与roe_q Rank复合（截面Rank变换后相乘）
  理论: ROE×BPS = 盈利×净资产的综合质量信号，优于单独的ROE或BPS的选股力（BPS×ROE挑剔基本面，适用小盘股的成长溢价）

方向: 正向 (高=更好的基本面, 更高预期收益)

步骤:
  a) 用PM数据合并: 财报45天延迟后在 roll-over 时间点合并（严格避免前视偏差）
  b) cross-sectional Rank-transform BPS 和 ROE_Q
  c) factor_raw = rank_bps × rank_roe_q
  d) 成交额OLS中性化（取残差）
  e) MAD Winsorize (3σ)
  f) z-score截面标准化

执行时间: 约5分钟
"""

import pandas as pd
import numpy as np
import time, sys
from scipy.stats import spearmanr

LAG_DAYS = 45  # 财报发布后45天才能获知

def winsorize_mad(series, n_mad=3.0):
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return series
    scaled = 1.4826 * mad
    return series.clip(med - n_mad*scaled, med + n_mad*scaled)

def ols_resid(y, X):
    """OLS残差: y - X@beta, X含截距"""
    mask = np.isfinite(y) & np.all(np.isfinite(X),axis=1)
    if mask.sum() < X.shape[1]+2:
        return y
    ym = y[mask]; Xm = X[mask]
    try:
        beta = np.linalg.solve(Xm.T@Xm + 1e-10*np.eye(Xm.shape[1]), Xm.T@ym)
        r = np.full_like(y, np.nan); r[mask] = ym - Xm@beta
        return r
    except np.linalg.LinAlgError:
        return y

def rank_transform(series):
    return series.rank(pct=True, na_option='keep')

# ─── 主流程 ──────────────────────────────────────────────────────────────────
T0 = time.time()
print("[1/6] 加载数据 ...", flush=True)

ROOT = '/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab'

k = pd.read_csv(f'{ROOT}/data/csi1000_kline_raw.csv', parse_dates=['date']).rename(columns={'stock_code': 'stock_id'})
fund = pd.read_csv(f'{ROOT}/data/csi1000_fundamental_cache.csv', parse_dates=['report_date']).rename(
    columns={'stock_code': 'stock_id', 'roe': 'roe_q'}
)
print(f"  kline: {k.shape}, fund: {fund.shape}", flush=True)

# ── 2. Merge: 财报45天延迟, 每个"stock+date"找到最新可用fundamental ──────────
print("[2/6] 合并fundamental (45天延迟) ...", flush=True)

fund['available_date'] = fund['report_date'] + pd.Timedelta(days=LAG_DAYS)

k.sort_values(['stock_id','date'], inplace=True); k.reset_index(drop=True, inplace=True)
fund.sort_values(['stock_id','available_date'], inplace=True); fund.reset_index(drop=True, inplace=True)

merged = []
for sid, k_sub in k.groupby('stock_id'):
    f_sub = fund[fund['stock_id']==sid].copy()
    if f_sub.empty: continue
    dates = k_sub['date'].values
    avail = f_sub['available_date'].values
    idx = np.searchsorted(avail, dates, side='right') - 1
    m = idx >= 0
    if m.sum() == 0: continue
    matched = pd.DataFrame({
        'date': dates[m],
        'stock_id': sid,
        'close': k_sub['close'].values[m],
        'amount': k_sub['amount'].values[m],
        'bps': f_sub['bps'].values[idx[m]],
        'roe_q': f_sub['roe_q'].values[idx[m]],
    })
    merged.append(matched)

merged = pd.concat(merged, ignore_index=True)
print(f"  Merged shape: {merged.shape}", flush=True)
print(f"  Time: {time.time()-T0:.1f}s", flush=True)

# ── 3. Rank Transform + 合成因子 ─────────────────────────────────────────────
print("[3/6] Rank-Transform + 合成 ...", flush=True)

merged['rank_bps'] = merged.groupby('date')['bps'].transform(rank_transform)
merged['rank_roe'] = merged.groupby('date')['roe_q'].transform(rank_transform)
merged['factor_raw'] = merged['rank_bps'] * merged['rank_roe']

print(f"  factor_raw range: [{merged['factor_raw'].min():.3f}, {merged['factor_raw'].max():.3f}]", flush=True)
print(f"  factor_raw > 0 ratio: {(merged['factor_raw']>0).mean():.2%}", flush=True)

# ── 4. 成交额中性化 ─────────────────────────────────────────────────────────────
print("[4/6] 成交额OLS中性化 ...", flush=True)

merged['log_amount'] = np.log(merged['amount'].replace(0, np.nan))
merged['_resid'] = np.nan

for dt, grp in merged.groupby('date', sort=False):
    m = grp['factor_raw'].notna() & grp['log_amount'].notna()
    if m.sum()<100: continue
    y = grp.loc[m,'factor_raw'].values
    X = np.column_stack([np.ones(m.sum()), grp.loc[m,'log_amount'].values.astype(float)])
    r = ols_resid(y, X)
    merged.loc[grp.index[m], '_resid'] = r

merged['_resid'] = merged.groupby('date')['_resid'].transform(lambda x: winsorize_mad(x, 3.0))
merged['factor_val'] = merged.groupby('date')['_resid'].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-10)
)

print(f"  factor_val valid: {merged['factor_val'].notna().sum()}/{len(merged)}", flush=True)

# ── 5. 输出CSV ────────────────────────────────────────────────────────────────
print("[5/6] 保存因子文件 ...", flush=True)

OUT = f'{ROOT}/data/factor_bps_roe_composite_v1.csv'
out_df = merged[['date','stock_id','factor_val']].copy()
out_df.rename(columns={'stock_id': 'stock_code', 'factor_val': 'bps_roe_composite_v1'}, inplace=True)
out_df['date'] = out_df['date'].dt.strftime('%Y-%m-%d')
out_df.to_csv(OUT, index=False)
print(f"  保存: {OUT}", flush=True)

# ── 6. 快速IC检验 ─────────────────────────────────────────────────────────────
print("[6/6] 快速IC检验 ...", flush=True)

merged['fwd_ret_5d'] = merged.groupby('stock_id')['close'].transform(
    lambda x: x.shift(-5) / x - 1
)
v = merged[['factor_val','fwd_ret_5d']].dropna()
if len(v):
    ic, p = spearmanr(v['factor_val'], v['fwd_ret_5d'])
    print(f"\n  初步 5d IC = {ic:.4f}  (p={p:.4f})  n={len(v)}")
    print(f"  完成! 耗时 {time.time()-T0:.1f}s", flush=True)
else:
    print("  ⚠ 前进收益数据不足", flush=True)

print(f"\n  因子路径: {OUT}")
print(f"  请执行: python3 skills/alpha-factor-lab/scripts/factor_backtest.py --factor {OUT} --returns data/csi1000_returns.csv --groups 5 --forward-days 5 --cost 0.003 --output output/bps_roe_composite_v1/")
