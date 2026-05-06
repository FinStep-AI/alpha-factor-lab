#!/usr/bin/env python3
"""
因子: close_low_composite_v1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
多时间尺度收盘位置复合因子

构造:
  close_ratio = (high - close) / (high - low + 1e-8)
   (接近低点=高 close_ratio, 接近高点=低 close_ratio)

  factor_raw = 0.4*MA5(close_ratio) + 0.4*MA10(close_ratio) + 0.2*MA20(close_ratio)

 方向: 高因子值 = 持续收在日内低点 (接近 close_low_v1 信号)
 与 close_low_v1 (MA20单窗口) 角度互补: 多时间尺度合成更稳健

中性化: log_amount OLS
标准化: MAD Winsorize + z-score
"""

import pandas as pd, numpy as np, time
from scipy.stats import spearmanr

ROOT = '/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab'

def winsorize_mad(series, n_mad=3.0):
    med = series.median()
    mad = (series - med).abs().median()
    if mad==0 or np.isnan(mad): return series
    lo = med - n_mad*1.4826*mad; hi = med + n_mad*1.4826*mad
    return series.clip(lo, hi)

T0 = time.time()
print("[close_low_composite_v1] 因子构造", flush=True)

# ─── 数据 ────────────────────────────────────────────────────────────────────
k = pd.read_csv(f'{ROOT}/data/csi1000_kline_raw.csv', parse_dates=['date']).rename(
    columns={'stock_code':'stock_id'})
k.sort_values(['stock_id','date'], inplace=True); k.reset_index(drop=True, inplace=True)
g = k.groupby('stock_id')

print(f"数据: {k.shape} rows, {k['stock_id'].nunique()} stocks", flush=True)

# ─── 基础字段 ─────────────────────────────────────────────────────────────────
k['close_ratio'] = (k['high'] - k['close']) / (k['high'] - k['low'] + 1e-8)

# ─── 三窗口复合信号 ───────────────────────────────────────────────────────────
print("[1/4] 多窗口复合因子 ...", flush=True)
k['cr_ma5']  = g['close_ratio'].transform(lambda x: x.rolling(5,  min_periods=3).mean())
k['cr_ma10'] = g['close_ratio'].transform(lambda x: x.rolling(10, min_periods=5).mean())
k['cr_ma20'] = g['close_ratio'].transform(lambda x: x.rolling(20, min_periods=10).mean())

# 加权组合 MA5: 0.4 MA10: 0.4 MA20: 0.2  (短期信号权重更高)
k['factor_raw'] = 0.4 * k['cr_ma5'] + 0.4 * k['cr_ma10'] + 0.2 * k['cr_ma20']

print(f"  factor_raw range: {k['factor_raw'].min():.3f} ~ {k['factor_raw'].max():.3f}", flush=True)
print(f"  Time: {time.time()-T0:.1f}s", flush=True)

# ─── 成交额中性化 ─────────────────────────────────────────────────────────────
print("[2/4] 成交额OLS中性化 ...", flush=True)

k['log_amount'] = np.log(k['amount'].replace(0, np.nan))
k['_resid'] = np.nan

def ols_resid(y, X):
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if m.sum() < X.shape[1]+2: return y
    ym, Xm = y[m], X[m]
    try:
        beta = np.linalg.solve(Xm.T@Xm + 1e-10*np.eye(Xm.shape[1]), Xm.T@ym)
        r = np.full_like(y, np.nan); r[m] = ym - Xm@beta; return r
    except np.linalg.LinAlgError: return y

for dt, grp in k.groupby('date', sort=False):
    m = grp['factor_raw'].notna() & grp['log_amount'].notna()
    if m.sum() < 100: continue
    y = grp.loc[m,'factor_raw'].values
    X = np.column_stack([np.ones(m.sum()), grp.loc[m,'log_amount'].values.astype(float)])
    r = ols_resid(y, X)
    k.loc[grp.index[m], '_resid'] = r

k['_resid'] = k.groupby('date')['_resid'].transform(lambda x: winsorize_mad(x, 3.0))
k['_z'] = k.groupby('date')['_resid'].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-10)
)

print(f"  有效样本: {k['_z'].notna().sum()}/{len(k)}", flush=True)

# ─── 5日前瞻IC快速检验 ─────────────────────────────────────────────────────────
print("[3/4] 初步IC检验 (5日) ...", flush=True)

k['ret_5d'] = g['close'].transform(lambda x: x.shift(-5) / x - 1)
v = k[['_z','ret_5d']].dropna()

if len(v) > 50000:
    ic, p = spearmanr(v['_z'], v['ret_5d'])
    print(f"  IC(5d) = {ic:.4f}  p={p:.4f}  n={len(v):,}")

    # 單 Hedge 保守性: G5 vs G1 收益對比
    k['group'] = k.groupby('date')['_z'].transform(
        lambda x: pd.qcut(x.rank(method='first'), 5, labels=['G1','G2','G3','G4','G5'], duplicates='drop')
    )
    ann = {}
    for gname, grp2 in k.dropna(subset=['group']).groupby('group'):
        ann_avg = (grp2['ret_5d'].mean() * 252)
        ann[gname] = ann_avg
    print("  年化收益分層 (5日調倉):")
    for kk in ['G1','G2','G3','G4','G5']:
        print(f"    {kk}: {ann.get(kk, float('nan')):.1%}")

    ls_sr = (ann.get('G5',0) - ann.get('G1',0)) / k.groupby('date').apply(
        lambda x: x['ret_5d'].std()*np.sqrt(252) if len(x)>2 else np.nan
    ).mean()
    print(f"  多空 Sharpe (rough): {ls_sr:.2f}")
else:
    print("  樣本不足，跳過IC檢驗")

print(f"  耗时: {time.time()-T0:.1f}s", flush=True)

# ─── 保存因子 ─────────────────────────────────────────────────────────────────
print("[4/4] 保存因子数据 ...", flush=True)
OUT = f'{ROOT}/data/factor_close_low_composite_v1.csv'
out = k[['date', 'stock_id', '_z']].rename(columns={'stock_id':'stock_code', '_z':'close_low_composite_v1'})
out['date'] = out['date'].dt.strftime('%Y-%m-%d')
out.to_csv(OUT, index=False)
print(f"  保存路径: {OUT}", flush=True)

print(f"\n总耗时: {time.time()-T0:.1f}s  ✅", flush=True)
print(f"因子名称: close_low_composite_v1")
print(f"请执行回测: python3 skills/alpha-factor-lab/scripts/factor_backtest.py --factor {OUT} --returns data/csi1000_returns.csv --groups 5 --forward-days 5 --cost 0.003 --output output/close_low_composite_v1/")
