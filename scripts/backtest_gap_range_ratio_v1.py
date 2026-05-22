#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: gap_range_ratio_v1 — 跳空缺口占日内振幅比 (60日平滑)
============================================================
构造逻辑:
  每日 gap_ratio = |open / prev_close - 1|
                    ─────────────────────────────────────
                    (high - low) / prev_close + ε

  分子 : 隔夜跳空幅度  %
  分母 : 当日日内振幅  %    (加 ε 防止 inf)
  比值越大 → 跳空缺口占日内振幅越高 → 当日开盘断层主导价格区间
             机构/大户通过开盘竞价形成价格断层，后续已无惯性修复

  最终因子 = 60 日滚动均值(日度 gap_ratio)，截面OLS对数市值中性化

回测配置:
  forward_days  = 20 (任务要求)
  rebalance     = 20
  cost          = 0.002 (单边趋势加载时上调 0.003)
  市值代理       = log( amount 20日均值 )
"""

import json
import sys
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ─── 参数 ───────────────────────────────────────────────────────────────
WINDOW         = 60          # 缺口比滚动平滑窗口（天）
FORWARD_DAYS   = 20          # 前瞻收益窗口
REBALANCE_FREQ = 20          # 调仓频率
N_GROUPS       = 5
COST           = 0.002       # 单边成本
WINSORIZE_PCT  = 0.05        # 5% 缩尾
FACTOR_ID      = "gap_range_ratio_v1"
FACTOR_NAME_CN = "跳空缺口-日内振幅比 v1"
FACTOR_NAME_EN = "Gap-to-Range Ratio (60d smoothed) v1"
DIRECTION      = 1           # 正向：高比值 → 更高收益
DATA_PATH      = Path(__file__).resolve().parent.parent \
                 / "data" / "csi1000_kline_raw.csv"
OUTPUT_DIR     = Path(__file__).resolve().parent.parent \
                 / "output" / FACTOR_ID
REPORT_PATH    = OUTPUT_DIR / "backtest_report.json"

# ─── 入口 ────────────────────────────────────────────────────────────────
t0 = time.time()
print(f"[1] 加载数据  {DATA_PATH.name} ...", flush=True)
df = pd.read_csv(DATA_PATH, usecols=["date","stock_code","open","close","high","low","amount"])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code","date"]).reset_index(drop=True)
print(f"    shape={df.shape}  {time.time()-t0:.1f}s", flush=True)

# ─── 2. 构造因子 ────────────────────────────────────────────────────────
print(f"[2] 构造 {FACTOR_ID} (LB={WINDOW}d)...", flush=True)
t1 = time.time()

prev_close = df.groupby("stock_code")["close"].shift(1)

gap       = np.abs(df["open"].values / prev_close.values - 1.0)
intra_rng = (df["high"].values - df["low"].values) / prev_close.values + 1e-6

raw_daily = gap / intra_rng                                                          # 日度缺口比
df["raw_daily_gap_ratio"] = raw_daily

# 60日滚动均值
df["factor_raw"] = df.groupby("stock_code")["raw_daily_gap_ratio"].transform(
    lambda s: s.rolling(WINDOW, min_periods=max(WINDOW//2, 10)).mean()
)

# 上市满 60 日约束：前 WINDOW 天 factor_raw 为 NaN 不予使用
# 市值代理: amount 20日均值 → log
df["log_amount"] = np.log(
    df.groupby("stock_code")["amount"]
      .transform(lambda x: x.rolling(20, min_periods=5).mean())
      .clip(lower=1.0)
)

nonnull = df["factor_raw"].notna().sum()
print(f"    raw mean={df['factor_raw'].dropna().mean():.4f}  "
      f"std={df['factor_raw'].dropna().std():.4f}  "
      f"coverage={nonnull/len(df):.2%}  {time.time()-t1:.1f}s", flush=True)

# ─── 3. pivots ──────────────────────────────────────────────────────────
print("[3] 构建 pivot 矩阵 ...", flush=True)
dates  = sorted(df["date"].unique())
stocks = sorted(df["stock_code"].unique())
dmap   = {d:i for i,d in enumerate(dates)}
smap   = {s:i for i,s in enumerate(stocks)}

# 因子矩阵
F = np.full((len(dates), len(stocks)), np.nan)
for _, row in df.dropna(subset=["factor_raw"]).iterrows():
    F[dmap[row["date"]], smap[row["stock_code"]]] = row["factor_raw"]
factor_df = pd.DataFrame(F, index=dates, columns=stocks)

# 收益矩阵
close_piv = df.pivot_table(index="date", columns="stock_code", values="close", dropna=False)
ret_df    = close_piv.pct_change()

# log 市值矩阵
log_amt_piv = df.pivot_table(index="date", columns="stock_code", values="log_amount", dropna=False)

# ─── 4. 市值中性化 + 缩尾 ─────────────────────────────────────────────
print("[4] 截面OLS中性化 + 缩尾 ...", flush=True)
t2 = time.time()
factor_neutral = factor_df.copy()
for date in dates:
    f = factor_df.loc[date].dropna()
    m = log_amt_piv.loc[date].reindex(f.index).dropna()
    common = f.index.intersection(m.index)
    if len(common) < 30:
        continue
    f_c = f[common].values; m_c = m[common].values
    X   = np.column_stack([np.ones(len(m_c)), m_c])
    try:
        beta = np.linalg.lstsq(X, f_c, rcond=None)[0]
        factor_neutral.loc[date, common] = f_c - X @ beta
    except Exception:
        pass

# 缩尾
for date in dates:
    row = factor_neutral.loc[date].dropna()
    if len(row) < 10:
        continue
    lo  = row.quantile(WINSORIZE_PCT)
    hi  = row.quantile(1 - WINSORIZE_PCT)
    factor_neutral.loc[date] = factor_neutral.loc[date].clip(lo, hi)

print(f"    中性化后 mean={factor_neutral.stack().mean():.4f}  "
      f"std={factor_neutral.stack().std():.4f}  {time.time()-t2:.1f}s", flush=True)

# ─── 5. 回测 ────────────────────────────────────────────────────────────
print("[5] 回测引擎 (fwd=20d, rb=20d, cost=%.3f)..." % COST, flush=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent \
                       / "skills" / "alpha-factor-lab" / "scripts"))
from factor_backtest import compute_group_returns, compute_ic_dynamic, \
    compute_metrics, save_backtest_data

common_dates = sorted(
    factor_neutral.dropna(how="all").index
    .intersection(ret_df.dropna(how="all").index)
)
common_stocks = sorted(factor_neutral.columns.intersection(ret_df.columns))
F_al = factor_neutral.loc[common_dates, common_stocks]
R_al = ret_df.loc[common_dates, common_stocks]

print(f"    矩阵: {len(common_dates)}日 × {len(common_stocks)}股", flush=True)

t3 = time.time()
ic_series  = compute_ic_dynamic(F_al, R_al, FORWARD_DAYS, "pearson")
ric_series = compute_ic_dynamic(F_al, R_al, FORWARD_DAYS, "spearman")
gr, to, hi = compute_group_returns(F_al, R_al, N_GROUPS, REBALANCE_FREQ, COST)
metrics    = compute_metrics(gr, ic_series, ric_series, to, N_GROUPS, holdings_info=hi)
print(f"    回测完成 {time.time()-t3:.1f}s", flush=True)

ic_mean = metrics.get("ic_mean", 0) or 0
ic_t    = metrics.get("ic_t_stat", 0) or 0
ls_sh   = metrics.get("long_short_sharpe", 0) or 0
ls_ann  = metrics.get("long_short_ann_return", 0) or 0
ls_mdd  = metrics.get("long_short_mdd", 0) or 0
mono    = metrics.get("monotonicity", 0) or 0

# ─── 6. 输出文件 ─────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_backtest_data(gr, ic_series, ric_series, str(OUTPUT_DIR))

def n2n(o):
    if isinstance(o,(np.bool_,)): return bool(o)
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,(np.floating,)):
        v=float(o); return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(o,float) and (np.isnan(o) or np.isinf(o)): return None
    if isinstance(o,dict):  return {k:n2n(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [n2n(v) for v in o]
    return o

report = {
    "factor_id":        FACTOR_ID,
    "factor_name":      FACTOR_NAME_CN,
    "factor_name_en":   FACTOR_NAME_EN,
    "direction":        DIRECTION,
    "window":           WINDOW,
    "period":           f"{common_dates[0].strftime('%Y-%m-%d')}  ~  "
                         f"{common_dates[-1].strftime('%Y-%m-%d')}",
    "n_dates":          len(common_dates),
    "n_stocks":         len(common_stocks),
    "forward_days":     FORWARD_DAYS,
    "rebalance_freq":   REBALANCE_FREQ,
    "cost":             COST,
    "winsorize":        WINSORIZE_PCT,
    "neutralization":   "OLS(log_amount_20d)",
    "metrics":          metrics,
}
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(n2n(report), f, indent=2, ensure_ascii=False)

# ─── 7. 打印摘要 ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  {FACTOR_NAME_CN}  —  {FACTOR_NAME_EN}")
print(f"{'='*60}")
print(f"  周期    : {report['period']}")
print(f"  IC均值  : {ic_mean:.4f}   (t={ic_t:.2f},  IR={metrics.get('ir',0) or 0:.3f}, "
      f"p={metrics.get('ic_p_value',0) or 0:.4f})")
print(f"  Rank IC : {metrics.get('rank_ic_mean',0) or 0:.4f}")
print(f"  多空Sharpe : {ls_sh:.3f}   多空年化 : {ls_ann:.2%}")
print(f"  多空MDD    : {ls_mdd:.2%}   Calmar  : {metrics.get('calmar_ratio',0) or 0:.2f}")
print(f"  换手率 : {metrics.get('turnover_mean',0) or 0:.3f}  单调性 : {mono:.3f}")
grp = metrics.get("group_returns_annualized", [None]*N_GROUPS)
for i, r_a in enumerate(grp, 1):
    g_sh = metrics.get("group_sharpe", [None]*N_GROUPS)
    sg   = g_sh[i-1] if i-1 < len(g_sh) else None
    if r_a is not None:
        print(f"    G{i}: {r_a:>10.2%}  sh={sg:.2f}" if sg else
              f"    G{i}: {r_a:>10.2%}")
print(f"{'='*60}")

is_valid = abs(ic_mean) > 0.015 and ic_t > 2 and abs(ls_sh) > 0.5
print(f"\n  ➤ 因子方 {'有效 ✓' if is_valid else '无效 ✗'}")
print(f"     |IC|={abs(ic_mean):.4f} > 0.015 : {'✓' if abs(ic_mean)>0.015 else '✗'}")
print(f"     t_value={ic_t:.2f} > 2      : {'✓' if ic_t>2 else '✗'}")
print(f"     LS_sharpe={abs(ls_sh):.3f} > 0.5: {'✓' if abs(ls_sh)>0.5 else '✗'}")
print(f"\n  输出:")
print(f"    {OUTPUT_DIR/'cumulative_returns.json'}")
print(f"    {OUTPUT_DIR/'ic_series.json'}")
print(f"    {REPORT_PATH}")
print(f"  总耗时: {time.time()-t0:.1f}s\n", flush=True)

sys.exit(0 if is_valid else 1)
