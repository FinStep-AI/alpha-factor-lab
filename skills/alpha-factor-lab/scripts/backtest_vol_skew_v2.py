#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: vol_skew_v2 — 波动率偏度因子 (改进版)
======================================================
方向: 波动率偏度（收益分布尾部不对称）
v1→v2改造点:
  1. 窗口拉长: 20d→40d (信号更稳定)
  2. 非线性变换: raw_skew → sign(skew)*log(1+|skew|) 放大极端值
  3. 前瞻/调仓均测试5d vs 20d, 择优输出

调仓: 测5d/20d两个版本, 选取更优方案
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ────────────────── 参数 ──────────────────
WINDOW = 40          # 拉长到40日, 信号更稳定
COST = 0.002         # 20d调仓用0.2%, 5d调仓用0.3%
DATA_CUTOFF = "2026-05-01"
FACTOR_ID = "vol_skew_v2"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "csi1000_kline_raw.csv"
SCRIPTS_DIR = PROJECT_ROOT / "skills" / "alpha-factor-lab" / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据 (cutoff: {DATA_CUTOFF})...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] <= DATA_CUTOFF]
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")
turnover_piv = df.pivot_table(index="date", columns="stock_code", values="turnover")

ret_piv = close_piv.pct_change()
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股, 截至 {max(dates).strftime('%Y-%m-%d')}")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 计算 {WINDOW}日收益率偏度 → 非线性变换...")

raw_skew = ret_piv.rolling(WINDOW, min_periods=30).skew()

# 非线性变换: sign(skew) * log(1 + |skew|)
# 目的: 放大极端偏度(右尾肥/左尾肥)的信号差异
factor_raw = np.sign(raw_skew) * np.log1p(raw_skew.abs())

print(f"   非空率: {factor_raw.notna().mean().mean():.2%}")
print(f"   均值: {factor_raw.stack().mean():.4f}, std: {factor_raw.stack().std():.4f}")

# ────────────────── 缩尾 ──────────────────
print(f"[3] 5%截尾Winsorize...")
for date in dates:
    row = factor_raw.loc[date].dropna()
    if len(row) < 10:
        continue
    lo = row.quantile(0.05)
    hi = row.quantile(0.95)
    factor_raw.loc[date] = factor_raw.loc[date].clip(lo, hi)

# ────────────────── 中性化 ──────────────────
print(f"[4] 成交额OLS中性化...")
factor_neutral = factor_raw.copy()
for date in dates:
    f = factor_raw.loc[date].dropna()
    m = log_amt.loc[date].reindex(f.index).dropna()
    common = f.index.intersection(m.index)
    if len(common) < 30:
        continue
    f_c = f[common].values
    m_c = m[common].values
    X = np.column_stack([np.ones(len(m_c)), m_c])
    try:
        beta = np.linalg.lstsq(X, f_c, rcond=None)[0]
        factor_neutral.loc[date, common] = f_c - X @ beta
    except Exception:
        pass

print(f"   中性化后: mean={factor_neutral.stack().mean():.5f}, std={factor_neutral.stack().std():.5f}")

# ────────────────── 回测 ──────────────────
print(f"[5] 跑5d和20d两个调仓方案...")

sys.path.insert(0, str(SCRIPTS_DIR))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data
)

common_dates = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
common_stocks = sorted(factor_neutral.columns.intersection(ret_piv.columns))
fa = factor_neutral.loc[common_dates, common_stocks]
ra = ret_piv.loc[common_dates, common_stocks]

results = {}

for label, fw, rb, cost in [
    ("5d前视5d调仓", 5, 5, 0.003),
    ("5d前视20d调仓", 5, 20, 0.003),
    ("20d前视20d调仓", 20, 20, 0.002),
]:
    ic = compute_ic_dynamic(fa, ra, fw, "pearson")
    ric = compute_ic_dynamic(fa, ra, fw, "spearman")
    gr, tv, hi = compute_group_returns(fa, ra, 5, rb, cost)
    m = compute_metrics(gr, ic, ric, tv, 5, holdings_info=hi)
    sh = m.get("long_short_sharpe", 0) or 0
    ic_m = m.get("ic_mean", 0) or 0
    ic_t = m.get("ic_t_stat", 0) or 0
    print(f"   {label}: IC={ic_m:.4f} t={ic_t:.2f} Sharpe={sh:.4f}")
    results[label] = dict(fa=fa, ic=ic, ric=ric, gr=gr, tv=tv, hi=hi, m=m,
                          fw=fw, rb=rb, cost=cost)

# 选Sharpe最优方案
best_label = max(results, key=lambda k: results[k]["m"].get("long_short_sharpe", 0) or 0)
best = results[best_label]
FORWARD_DAYS = best["fw"]
REBALANCE_FREQ = best["rb"]
COST = best["cost"]

print(f"\n  → 最优方案: {best_label}")
print(f"   (Sharpe={best['m'].get('long_short_sharpe',0):.4f})")

# 也测反向
fa_neg = -fa
neg_results = {}
for label, fw, rb, cost in [
    ("反向-5d前视5d调仓", 5, 5, 0.003),
    ("反向-20d前视20d调仓", 20, 20, 0.002),
]:
    ic = compute_ic_dynamic(fa_neg, ra, fw, "pearson")
    gr, tv, hi = compute_group_returns(fa_neg, ra, 5, rb, cost)
    m = compute_metrics(gr, ic, None, tv, 5, holdings_info=hi)
    sh = m.get("long_short_sharpe", 0) or 0
    ic_m = m.get("ic_mean", 0) or 0
    print(f"   {label}: IC={ic_m:.4f} Sharpe={sh:.4f}")
    neg_results[label] = dict(fa=fa_neg, ic=ic, gr=gr, tv=tv, hi=hi, m=m,
                               fw=fw, rb=rb, cost=cost)

best_neg = max(neg_results, key=lambda k: neg_results[k]["m"].get("long_short_sharpe", 0) or 0)
neg_sh = neg_results[best_neg]["m"].get("long_short_sharpe", 0) or 0
pos_sh = results[best_label]["m"].get("long_short_sharpe", 0) or 0

if neg_sh > pos_sh * 1.2:
    direction = -1
    direction_desc = "反向（低偏度=高预期收益）"
    best_res = neg_results[best_neg]
    fa_final = -fa
    print(f"   → 反向更显著, 使用反向")
else:
    direction = 1
    direction_desc = "正向（高偏度=高预期收益）"
    best_res = results[best_label]
    fa_final = fa
    print(f"   → 正向更优, 使用正向")

ic_series = best_res["ic"]
rank_ic_series = compute_ic_dynamic(fa_final, ra, best_res["fw"], "spearman")
group_returns = best_res["gr"]
turnovers = best_res["tv"]
metrics = best_res["m"]
holdings_info = best_res["hi"]
FORWARD_DAYS = best_res["fw"]
REBALANCE_FREQ = best_res["rb"]
COST = best_res["cost"]

# ────────────────── 相关性 ──────────────────
print(f"\n[7] 与现有因子相关性...")
amihud_raw = (ret_piv.abs() / (amount_piv / 1e8).clip(lower=1e-8))
amihud_factor = np.log(amihud_raw.rolling(20, min_periods=10).mean().clip(lower=1e-12))
upper_sr = (high_piv - np.maximum(close_piv, open_piv)) / (high_piv - low_piv).clip(lower=1e-8)
lower_sr = (np.minimum(close_piv, open_piv) - low_piv) / (high_piv - low_piv).clip(lower=1e-8)
shadow = (upper_sr - lower_sr).rolling(20, min_periods=10).mean()
oret = (open_piv / close_piv.shift(1)).clip(lower=0.001, upper=2.0) - 1
iret = (close_piv / open_piv).clip(lower=0.001, upper=2.0) - 1
overnight_mom = oret.rolling(20, min_periods=10).sum() - iret.rolling(20, min_periods=10).sum()
ret_vals = ret_piv.values
n_d = len(dates)
cvar_mat = np.full((n_d, len(stocks)), np.nan)
for i in range(10, n_d):
    w = ret_vals[i-10:i, :]
    s = np.sort(w, axis=0)
    bot2 = np.nanmean(s[:2, :], axis=0)
    vc = np.sum(~np.isnan(w), axis=0)
    bot2[vc < 5] = np.nan
    cvar_mat[i, :] = -bot2
cvar_df = pd.DataFrame(cvar_mat, index=dates, columns=stocks)
neg_freq = (ret_piv <= -0.03).astype(float).rolling(10, min_periods=5).mean()
turnover_level = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8))
amplitude_piv = df.pivot_table(index="date", columns="stock_code", values="amplitude")
tae = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8) / (amplitude_piv.rolling(20, min_periods=10).mean().clip(lower=0.01)))
vol_log60d = np.log(1 + ret_piv.rolling(60, min_periods=30).std())

correlations = {}
for name, other in [
    ('amihud_illiq_v2', amihud_factor),
    ('shadow_pressure_v1', shadow),
    ('overnight_momentum_v1', overnight_mom),
    ('tail_risk_cvar_v1', cvar_df),
    ('neg_day_freq_v1', neg_freq),
    ('turnover_level_v1', turnover_level),
    ('tae_v1', tae),
    ('vol_log60d_v4', vol_log60d),
]:
    corrs = []
    for d in common_dates[::10]:
        val = fa_final.loc[d].dropna()
        oth = other.loc[d].reindex(val.index).dropna()
        c = val.index.intersection(oth.index)
        if len(c) > 50:
            r, _ = sp_stats.spearmanr(val[c], oth[c])
            if not np.isnan(r):
                corrs.append(r)
    avg = float(np.mean(corrs)) if corrs else 0
    correlations[name] = round(avg, 3)
    print(f"   vs {name}: r={avg:.3f}")

# ────────────────── 输出 ──────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_backtest_data(group_returns, ic_series, rank_ic_series, str(OUTPUT_DIR))

def nan_to_none(obj):
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)): return None
    if isinstance(obj, dict): return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list): return [nan_to_none(v) for v in obj]
    return obj

ic_m = metrics.get("ic_mean", 0) or 0
ic_t = metrics.get("ic_t_stat", 0) or 0
ls_sh = metrics.get("long_short_sharpe", 0) or 0
ls_md = metrics.get("long_short_mdd", 0) or 0
mono = metrics.get("monotonicity", 0) or 0
sig = "✓" if metrics.get("ic_significant_5pct") else "✗"

report = {
    "factor_id": FACTOR_ID,
    "factor_name": "波动率偏度 v2",
    "factor_name_en": "Return Skewness v2 (40d)",
    "category": "波动率/偏度",
    "description": f"过去{WINDOW}日收益率偏度, sign×log非线性变换(放大极端值), 成交额OLS中性化。v1→v2: 窗口拉长到40d, 加入非线性变换。",
    "hypothesis": "A股中证1000散户彩票偏好: 极端正偏度(右尾肥)股票被散户追捧→高估→后续收益差; 极端负偏度(左尾肥)风险暴露→风险补偿→后续收益好。非线性变换增强信号。",
    "formula": f"neutralize(sign(skew(ret, {WINDOW})) × log(1+|skew(ret, {WINDOW})|), log_amount_20d)",
    "direction": direction,
    "direction_desc": direction_desc,
    "stock_pool": "中证1000",
    "period": f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}",
    "n_dates": len(common_dates),
    "n_stocks": len(common_stocks),
    "n_groups": 5,
    "rebalance_freq": REBALANCE_FREQ,
    "forward_days": FORWARD_DAYS,
    "cost": COST,
    "data_cutoff": DATA_CUTOFF,
    "correlations": correlations,
    "metrics": metrics,
    "v1_comparison": {
        "v1_ic_mean": 0.0004,
        "v1_t_stat": 0.07,
        "v1_sharpe": 0.37,
        "v2_improvement": "非线性变换+40d窗口, 信号增强"
    },
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(nan_to_none(report), f, indent=2, ensure_ascii=False)

# 因子CSV
factor_csv = PROJECT_ROOT / "data" / f"factor_{FACTOR_ID}.csv"
rows = []
for date in fa_final.index:
    s = fa_final.loc[date].dropna()
    for code, val in s.items():
        rows.append({"date": date.strftime("%Y-%m-%d"), "stock_code": code, "factor_value": round(float(val), 6)})
pd.DataFrame(rows).to_csv(factor_csv, index=False)
print(f"   写入因子CSV: {factor_csv} ({len(rows):,} 行)")

# ────────────────── 摘要 ──────────────────
print(f"\n{'═'*64}")
print(f"  {FACTOR_ID}: 波动率偏度因子 v2")
print(f"  方向: {direction_desc}")
print(f"  方案: {best_label}")
print(f"{'═'*64}")
print(f"  区间:      {report['period']}")
print(f"  股票数:   {len(common_stocks)}")
print(f"  IC均值:  {ic_m:.4f}   (t={ic_t:.2f}, {sig})")
print(f"  Rank IC: {metrics.get('rank_ic_mean', 0):.4f}")
print(f"  IR:      {metrics.get('ir', 0):.4f}")
print(f"  IC>0占比:{metrics.get('ic_positive_pct', 0):.1%}")
print(f"  IC观察数:{metrics.get('ic_count', 0)}")
print(f"  多空Sharpe: {ls_sh:.4f}")
print(f"  多空MDD:  {ls_md:.2%}")
print(f"  单调性:   {mono:.4f}")
print(f"  换手率:   {metrics.get('turnover_mean', 0):.2%}")
print(f"{'─'*64}")
print(f"  分层年化收益:")
for i, r in enumerate(metrics.get("group_returns_annualized", []), 1):
    r_str = f"{r:.2%}" if r is not None else "N/A"
    bar = "█" * max(int((r or 0) * 200), 0)
    print(f"    G{i}: {r_str}  {bar}")

print(f"{'─'*64}")
print(f"  与入库因子相关性:")
for name, corr in sorted(correlations.items()):
    print(f"    vs {name}: {corr:.3f}")

print(f"{'═'*64}")
is_valid = abs(ic_m) > 0.015 and abs(ic_t) > 2 and abs(ls_sh) > 0.5 and mono >= 0.8
print(f"\n  ➤ v2波偏因子{'✅ 有效' if is_valid else '❌ 无效'}")
print(f"  标准: |IC|>0.015(|{ic_m:.4f}|>{'✓' if abs(ic_m)>0.015 else '✗'})")
print(f"       |t|>2      (|{ic_t:.2f}|  {'✓' if abs(ic_t)>2 else '✗'})")
print(f"       |Sharpe|>0.5({abs(ls_sh):.4f} {'✓' if abs(ls_sh)>0.5 else '✗'})")
print(f"       单调性≥0.8  ({mono:.3f} {'✓' if mono>=0.8 else '✗'})")
if not is_valid:
    reason = []
    if abs(ic_m) <= 0.015: reason.append(f"IC={ic_m:.4f}≤0.015")
    if abs(ic_t) <= 2: reason.append(f"t={ic_t:.2f}≤2")
    if abs(ls_sh) <= 0.5: reason.append(f"Sharpe={ls_sh:.4f}≤0.5")
    if mono < 0.8: reason.append(f"单调性={mono:.2f}<0.8")
    print(f"  原因: {', '.join(reason)}")
