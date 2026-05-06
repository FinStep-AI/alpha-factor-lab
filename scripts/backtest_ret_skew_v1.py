#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""
因子: ret_skew_v1 — 日收益率偏度
"""

import json, sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ─────────── 参数 ───────────
WINDOW = 20
FORWARD_DAYS = 20
REBALANCE_FREQ = 20
N_GROUPS = 5
COST = 0.002
WINSORIZE_PCT = 0.05
FACTOR_ID = "ret_skew_v1"
BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "data" / "csi1000_kline_raw.csv"
OUTPUT_DIR = BASE / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

sys.path.insert(0, str(BASE / "skills" / "alpha-factor-lab" / "scripts"))

# ─────────── 1. 数据加载 ───────────
print("[1] 加载数据...", flush=True)
t0 = time.time()
df = pd.read_csv(DATA_PATH, usecols=["date", "stock_code", "close", "amount", "volume", "high", "low"])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
print(f"   加载完成 {time.time()-t0:.1f}s, shape={df.shape}", flush=True)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
volume_piv = df.pivot_table(index="date", columns="stock_code", values="volume")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")
ret_piv = close_piv.pct_change()
log_mktcap = np.log(amount_piv.rolling(20).mean().clip(lower=1))
dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股", flush=True)

# ─────────── 2. 因子构造 (向量化) ───────────
print(f"[2] 构造 ret_skew 因子 (window={WINDOW})...", flush=True)
t0 = time.time()

# 使用 rolling apply
def rolling_skewness(x):
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return np.nan
    return sp_stats.skew(x, bias=False)

# 逐股计算 (日频滚动可以向量化)
factor_values = []
for i, stock in enumerate(stocks):
    ret_s = ret_piv[stock].values
    skews = np.full(len(dates), np.nan)
    # 简单滑动窗口
    for j in range(WINDOW, len(dates)):
        window = ret_s[j-WINDOW:j]
        if np.isnan(window).sum() < 2:
            valid = window[~np.isnan(window)]
            if len(valid) >= 10:
                skews[j] = sp_stats.skew(valid, bias=False)
    factor_values.append(skews)
    if (i+1) % 100 == 0:
        print(f"   ... 进度 {i+1}/{len(stocks)}", flush=True)

factor_matrix = pd.DataFrame(np.array(factor_values).T, index=dates, columns=stocks)
print(f"   因子计算完成 {time.time()-t0:.1f}s, 非空率={factor_matrix.notna().mean().mean():.2%}", flush=True)

# 方向回顾
print(f"[2b] 方向回顾检验...", flush=True)
test_ic_pos = 0; test_ic_neg = 0; count = 0
test_start = WINDOW + FORWARD_DAYS
for idx in range(test_start, min(test_start+80, len(dates))):
    dt = dates[idx]
    f = factor_matrix.iloc[idx].dropna()
    m = log_mktcap.iloc[idx].reindex(f.index).dropna()
    common = f.index.intersection(m.index)
    if len(common) < 30:
        continue
    X = np.column_stack([np.ones(len(m[common])), m[common].values])
    try:
        beta = np.linalg.lstsq(X, f[common].values, rcond=None)[0]
        f_resid = f[common].values - X @ beta
        fwd_idx = idx + FORWARD_DAYS
        if fwd_idx >= len(dates):
            continue
        ret_fwd = ret_piv.iloc[fwd_idx][common].values
        v = ~(np.isnan(f_resid) | np.isnan(ret_fwd))
        if v.sum() >= 10 and np.std(f_resid[v]) > 0:
            ic_pos = np.corrcoef(f_resid[v], ret_fwd[v])[0,1]
            ic_neg = np.corrcoef(-f_resid[v], ret_fwd[v])[0,1]
            if not np.isnan(ic_pos):
                test_ic_pos += ic_pos
                test_ic_neg += ic_neg
                count += 1
    except: pass

if count > 0:
    test_ic_pos /= count; test_ic_neg /= count
    direction = "positive" if abs(test_ic_pos) > abs(test_ic_neg) else "negative"
    print(f"   正向IC: {test_ic_pos:.4f}, 反向IC: {test_ic_neg:.4f}", flush=True)
    print(f"   → 选择: {'正向(高偏度→高收益)' if direction=='positive' else '反向(低偏度→高收益)'}", flush=True)
    factor_matrix = factor_matrix if direction == "positive" else -factor_matrix
else:
    print(f"   方向预览失败，保持原始方向", flush=True)

# ─────────── 3. 缩尾 ───────────
print(f"[3] 缩尾 ({WINSORIZE_PCT*100:.0f}%)...", flush=True)
t0 = time.time()
for date in dates:
    row = factor_matrix.loc[date].dropna()
    if len(row) < 10: continue
    lo = row.quantile(WINSORIZE_PCT); hi = row.quantile(1-WINSORIZE_PCT)
    factor_matrix.loc[date] = factor_matrix.loc[date].clip(lo, hi)
print(f"   完成 {time.time()-t0:.1f}s", flush=True)

# ─────────── 4. 市值中性化 ───────────
print(f"[4] 市值中性化...", flush=True)
t0 = time.time()
factor_neutral = factor_matrix.copy()
for date in dates:
    f = factor_matrix.loc[date].dropna()
    m = log_mktcap.loc[date].reindex(f.index).dropna()
    common = f.index.intersection(m.index)
    if len(common) < 30: continue
    f_c = f[common].values; m_c = m[common].values
    X = np.column_stack([np.ones(len(m_c)), m_c])
    try:
        beta = np.linalg.lstsq(X, f_c, rcond=None)[0]
        factor_neutral.loc[date, common] = f_c - X @ beta
    except: pass
print(f"   完成 {time.time()-t0:.1f}s", flush=True)

# 因子分布
non_null = factor_neutral.stack().dropna()
print(f"   均值={non_null.mean():.4f}, std={non_null.std():.4f}, skew={sp_stats.skew(non_null, bias=False):.4f}", flush=True)

# ─────────── 5. 回测 ───────────
print(f"[5] 分层回测...", flush=True)
t0 = time.time()
common_dates = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
common_stocks = sorted(factor_neutral.columns.intersection(ret_piv.columns))
factor_aligned = factor_neutral.loc[common_dates, common_stocks]
return_aligned = ret_piv.loc[common_dates, common_stocks]
print(f"   对齐: {len(common_dates)}日 x {len(common_stocks)}股", flush=True)

from factor_backtest import compute_group_returns, compute_ic_dynamic, compute_metrics, save_backtest_data

ic_series = compute_ic_dynamic(factor_aligned, return_aligned, FORWARD_DAYS, "pearson")
rank_ic_series = compute_ic_dynamic(factor_aligned, return_aligned, FORWARD_DAYS, "spearman")
group_returns, turnovers, holdings_info = compute_group_returns(
    factor_aligned, return_aligned, N_GROUPS, REBALANCE_FREQ, COST)
metrics = compute_metrics(group_returns, ic_series, rank_ic_series, turnovers, N_GROUPS, holdings_info=holdings_info)
print(f"   回测完成 {time.time()-t0:.1f}s", flush=True)

# ─────────── 6. 输出 ───────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_backtest_data(group_returns, ic_series, rank_ic_series, str(OUTPUT_DIR))

def nan_to_none(o):
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o); return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)): return None
    if isinstance(o, dict): return {k: nan_to_none(v) for k,v in o.items()}
    if isinstance(o, (list, tuple)): return [nan_to_none(v) for v in o]
    return o

report = {
    "factor_id": FACTOR_ID,
    "factor_name": "日收益率偏度 v1",
    "factor_name_en": "Daily Return Skewness v1",
    "category": "波动率/高阶矩",
    "period": f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}",
    "n_dates": len(common_dates), "n_stocks": len(common_stocks),
    "n_groups": N_GROUPS, "rebalance_freq": REBALANCE_FREQ, "forward_days": FORWARD_DAYS,
    "cost": COST, "metrics": metrics,
}
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(nan_to_none(report), f, indent=2, ensure_ascii=False)

# ─────────── 7. 打印摘要 ───────────
print(f"\n{'='*60}")
print(f"  Return Skewness v1 回测结果")
print(f"{'='*60}")
ic_mean = metrics.get('ic_mean', 0) or 0; ic_t = abs(metrics.get('ic_t_stat', 0) or 0)
ls_sh = metrics.get("long_short_sharpe", 0) or 0; ls_md = metrics.get("long_short_mdd", 0) or 0
mono = metrics.get("monotonicity", 0) or 0; to = metrics.get("turnover_mean", 0) or 0
print(f"  IC均值: {ic_mean:.4f} (t={ic_t:.2f})")
print(f"  Rank IC: {metrics.get('rank_ic_mean', 0):.4f}")
print(f"  IR:      {metrics.get('ir', 0):.4f}")
print(f"  多空Sharpe: {ls_sh:.4f}\n  多空MDD:    {ls_md:.2%}")
print(f"  单调性: {mono:.4f}  换手率: {to:.2%}")
print(f"  分层年化收益:")
for i,r in enumerate(metrics.get("group_returns_annualized", []), 1):
    print(f"    G{i}: {r:.2%}" if r is not None else f"    G{i}: N/A")
print(f"{'='*60}")
is_valid = abs(ic_mean) > 0.015 and ic_t > 2 and abs(ls_sh) > 0.5
print(f"\n  因子{'有效 ✓' if is_valid else '无效 ✗'} (|IC|>0.015: {'✓' if abs(ic_mean)>0.015 else '✗'}, t>2: {'✓' if ic_t>2 else '✗'}, |Sharpe|>0.5: {'✓' if abs(ls_sh)>0.5 else '✗'})", flush=True)
print(f"\n输出文件:", flush=True)
print(f"  {OUTPUT_DIR / 'cumulative_returns.json'}", flush=True)
print(f"  {OUTPUT_DIR / 'ic_series.json'}", flush=True)
sys.exit(0 if is_valid else 1)
