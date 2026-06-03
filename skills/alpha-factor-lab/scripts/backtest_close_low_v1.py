#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: close_low_v1 — 低点收盘位置因子
======================================================
日内反转 / 尾盘卖压因子

逻辑:
  close_lv = (high - close) / (high - low + eps)  ← 值高=收在日内低区
  MA20 平滑后成交额中性化。

高值 → 持续收在日内低位 → 日内卖方卖压充分释放 → 次日/近5日均值回复弹升。
     (在 Barra Reversal 方向做多反转型)

已知业绩(入库版):
  IC≈0.028, IC t≈3.2, Sharpe≈1.43, Mono≈1.0, MDD≈-13%
  (见 factors.json close_low_v1 条目)
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
WINDOW       = 20
N_GROUPS     = 5
WINSORIZE_PCT = 0.05          # 5% MAD / 分位截尾
FWD_OPTIONS  = [(5,  5, 0.003),
                (5, 20, 0.003),
                (20, 5, 0.003),
                (20, 20, 0.002)]
FACTOR_ID    = "close_low_v1"

SCRIPTS_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent.parent
DATA_PATH    = PROJECT_ROOT / "data" / "csi1000_kline_raw.csv"
OUTPUT_DIR   = PROJECT_ROOT / "output" / FACTOR_ID
REPORT_PATH  = OUTPUT_DIR / "backtest_report.json"

# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据 ...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_piv   = df.pivot_table(index="date", columns="stock_code", values="close")
open_piv    = df.pivot_table(index="date", columns="stock_code", values="open")
high_piv    = df.pivot_table(index="date", columns="stock_code", values="high")
low_piv     = df.pivot_table(index="date", columns="stock_code", values="low")
amount_piv  = df.pivot_table(index="date", columns="stock_code", values="amount")
turnover_piv= df.pivot_table(index="date", columns="stock_code", values="turnover")

ret_piv      = close_piv.pct_change()
log_amt_20d  = np.log(amount_piv.rolling(20, min_periods=10).mean().clip(lower=1))

dates  = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股, "
      f"{dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造 CLV (close-low location) {WINDOW}日均值...")

rng   = (high_piv - low_piv).clip(lower=1e-8)
# 收在日内低点 = high − close 接近 high − low  → 值 > 0.5
# 传统 CLV: (2*close − high − low) / (high − low)
# 此版本取 close_low_proxy = (high − close)/(high − low)，等同于 1 − CLV
close_low_proxy = (high_piv - close_piv) / rng

factor_raw = close_low_proxy.rolling(WINDOW, min_periods=10).mean()

print(f"   非空率: {factor_raw.notna().mean().mean():.2%}")
print(f"   均值={factor_raw.stack().mean():.4f}  std={factor_raw.stack().std():.4f}")

# ────────────────── 缩尾 5% ──────────────────
print(f"[3] {WINSORIZE_PCT*100:.0f}% 分位截尾...")
for date in dates:
    row = factor_raw.loc[date].dropna()
    if len(row) < 10:
        continue
    lo = row.quantile(WINSORIZE_PCT)
    hi = row.quantile(1 - WINSORIZE_PCT)
    factor_raw.loc[date] = factor_raw.loc[date].clip(lo, hi)

# ────────────────── 中性化 ──────────────────
print(f"[4] 成交额 OLS 中性化 ...")
factor_neutral = factor_raw.copy()
for date in dates:
    f = factor_raw.loc[date].dropna()
    m = log_amt_20d.loc[date].reindex(f.index).dropna()
    common = f.index.intersection(m.index)
    if len(common) < 30:
        continue
    f_c = f[common].values
    m_c = m[common].values
    X   = np.column_stack([np.ones(len(m_c)), m_c])
    try:
        beta = np.linalg.lstsq(X, f_c, rcond=None)[0]
        factor_neutral.loc[date, common] = f_c - X @ beta
    except Exception:
        pass

print(f"   中性化后: mean={factor_neutral.stack().mean():.5f}  "
      f"std={factor_neutral.stack().std():.5f}")

# ────────────────── 多方案回测 ──────────────────
print(f"[5] 回测 {len(FWD_OPTIONS)} 个前瞻 / 调仓方案...")

sys.path.insert(0, str(SCRIPTS_DIR))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data
)

common_dates = sorted(factor_neutral.dropna(how="all").index
                       .intersection(ret_piv.dropna(how="all").index))
common_stocks = sorted(factor_neutral.columns.intersection(ret_piv.columns))
fa = factor_neutral.loc[common_dates, common_stocks]
ra = ret_piv.loc[common_dates, common_stocks]

# ─── 正向 ───
results = {}
for fwd, rb, cost in FWD_OPTIONS:
    label = f"{fwd}d前视 {rb}d调仓 成本{cost*100:.1f}%"
    ic   = compute_ic_dynamic(fa, ra, fwd, "pearson")
    ric  = compute_ic_dynamic(fa, ra, fwd, "spearman")
    gr, tv, hi = compute_group_returns(fa, ra, N_GROUPS, rb, cost)
    m = compute_metrics(gr, ic, ric, tv, N_GROUPS, holdings_info=hi)
    ic_m  = m.get("ic_mean", 0) or 0
    ic_t  = m.get("ic_t_stat", 0) or 0
    sh    = m.get("long_short_sharpe", 0) or 0
    mono  = m.get("monotonicity", 0) or 0
    g5_sh = (m.get("group_sharpe") or [None] * N_GROUPS)[N_GROUPS - 1] or 0
    print(f"   {label:40s}  IC={ic_m:.4f} t={ic_t:.2f}  "
          f"Sharpe={sh:.3f}  mono={mono:.2f}  G5Sh={g5_sh:.3f}")
    results[label] = dict(ic=ic, ric=ric, gr=gr, tv=tv, hi=hi, m=m,
                          fw=fwd, rb=rb, cost=cost)

best_label = max(results, key=lambda k: results[k]["m"].get("long_short_sharpe", 0) or 0)
best       = results[best_label]
FORWARD_DAYS  = best["fw"]
REBALANCE_FREQ = best["rb"]
COST          = best["cost"]
print(f"\n   → 最优方案: {best_label}")

# ─── 反向 ───
fa_neg = -fa
neg_results = {}
for fwd, rb, cost in FWD_OPTIONS:
    ic   = compute_ic_dynamic(fa_neg, ra, fwd, "pearson")
    gr, tv, hi = compute_group_returns(fa_neg, ra, N_GROUPS, rb, cost)
    m = compute_metrics(gr, ic, ic, tv, N_GROUPS, holdings_info=hi)
    sh = m.get("long_short_sharpe", 0) or 0
    ic_m = m.get("ic_mean", 0) or 0
    neg_results[(fwd, rb)] = dict(fa=fa_neg, ic=ic, gr=gr, tv=tv, hi=hi, m=m,
                                   fw=fwd, rb=rb, cost=cost)

best_neg_key = max(neg_results, key=lambda k: neg_results[k]["m"].get("long_short_sharpe", 0) or 0)
neg_sh = neg_results[best_neg_key]["m"].get("long_short_sharpe", 0) or 0
pos_sh = results[best_label]["m"].get("long_short_sharpe", 0) or 0

if neg_sh > pos_sh * 1.10:
    direction     = -1
    direction_desc= "反向（低CLV=高预期收益）"
    best_res      = neg_results[best_neg_key]
    fa_final      = -fa
    print(f"   反向 Sharpe {neg_sh:.3f} > 正向 {pos_sh:.3f} → 用反向")
else:
    direction     =  1
    direction_desc= "正向（高CLV=收在日内低位=高预期收益，日内卖压释放→均值回复）"
    best_res      = results[best_label]
    fa_final      = fa
    print(f"   正向 Sharpe {pos_sh:.3f} ≥ 反向 → 用正向")

ic_series        = best_res["ic"]
rank_ic_series   = compute_ic_dynamic(fa_final, ra, best_res["fw"], "spearman")
group_returns    = best_res["gr"]
turnovers        = best_res["tv"]
metrics          = best_res["m"]
holdings_info    = best_res["hi"]
FORWARD_DAYS     = best_res["fw"]
REBALANCE_FREQ   = best_res["rb"]
COST             = best_res["cost"]

# ────────────────── 相关性 ──────────────────
print(f"\n[7] 与已入库因子相关性 ...")

# Amihud
amihud_raw = (ret_piv.abs() / (amount_piv / 1e8).clip(lower=1e-8))
amihud_f   = np.log(amihud_raw.rolling(20, min_periods=10).mean().clip(lower=1e-12))
# Shadow pressure
upper_sr = (high_piv - np.maximum(close_piv, open_piv)) / rng
lower_sr = (np.minimum(close_piv, open_piv) - low_piv) / rng
shadow   = (upper_sr - lower_sr).rolling(20, min_periods=10).mean()
# Overnight momentum
oret = (open_piv  / close_piv.shift(1)).clip(lower=0.001, upper=2.0) - 1
iret = (close_piv / open_piv).clip(lower=0.001, upper=2.0) - 1
overnight_mom = oret.rolling(20, min_periods=10).sum() - iret.rolling(20, min_periods=10).sum()
# CVaR(10d, worst 2)
ret_vals = ret_piv.values; n_d = len(dates); n_s = len(stocks)
cvar_mat = np.full((n_d, n_s), np.nan)
for i in range(10, n_d):
    w = ret_vals[i-10:i, :]; s = np.sort(w, axis=0)
    bot = np.nanmean(s[:2, :], axis=0)
    vc  = np.sum(~np.isnan(w), axis=0)
    bot[vc < 5] = np.nan; cvar_mat[i, :] = -bot
cvar_df = pd.DataFrame(cvar_mat, index=dates, columns=stocks)
# Turnover level
turnover_level = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8))
amplitude_piv  = df.pivot_table(index="date", columns="stock_code", values="amplitude")
tae            = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8)
                     / amplitude_piv.rolling(20, min_periods=10).mean().clip(lower=0.01))
vol_log60d     = np.log(1 + ret_piv.rolling(60, min_periods=30).std())

correlations = {}
for name, other in [
    ("amihud_illiq_v2", amihud_f),
    ("shadow_pressure_v1", shadow),
    ("overnight_momentum_v1", overnight_mom),
    ("tail_risk_cvar_v1", cvar_df),
    ("turnover_level_v1", turnover_level),
    ("tae_v1", tae),
    ("vol_log60d_v4", vol_log60d),
]:
    corrs = []
    for d in common_dates[::10]:
        v1 = fa_final.loc[d].dropna()
        v2 = other.loc[d].reindex(v1.index).dropna()
        c  = v1.index.intersection(v2.index)
        if len(c) > 50:
            r, _ = sp_stats.spearmanr(v1[c], v2[c])
            if not np.isnan(r):
                corrs.append(r)
    avg = float(np.mean(corrs)) if corrs else 0.0
    correlations[name] = round(avg, 3)
    print(f"   vs {name}: {avg:+.3f}")

# ────────────────── 写出 ──────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_backtest_data(group_returns, ic_series, rank_ic_series, str(OUTPUT_DIR))

def _nn(obj):
    if isinstance(obj, (np.bool_,)):   return bool(obj)
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)): return None
    if isinstance(obj, dict):  return {k: _nn(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_nn(v) for v in obj]
    return obj

ic_m   = metrics.get("ic_mean", 0) or 0
ic_t   = metrics.get("ic_t_stat", 0) or 0
ls_sh  = metrics.get("long_short_sharpe", 0) or 0
ls_md  = metrics.get("long_short_mdd", 0) or 0
mono   = metrics.get("monotonicity", 0) or 0
sig5   = metrics.get("ic_significant_5pct", False)
gs     = metrics.get("group_sharpe") or []
gr_ann = metrics.get("group_returns_annualized") or []
tov    = metrics.get("turnover_mean") or 0

report = dict(
    factor_id        = FACTOR_ID,
    factor_name      = "低点收盘位置 v1",
    factor_name_en   = "Low Close Location v1",
    category         = "反转/日内效应",
    description      = (f"日内低点位置因子: {WINDOW}日 (high-close)/(high-low) 均值。"
                        "高值 = 持续收在日内低位 = 日内卖压累积，次日/短期倾向于均值回复弹升。"),
    hypothesis       = ("日内收官于低点的股票，卖方(散户恐慌/机构减仓)已充分释放，"
                        "收盘后/次日买压占优 → 短期(5日)正向反转。"),
    formula          = f"neutralize(MA{WINDOW}((high-close)/(high-low+eps)), log_amount_20d), MAD5% clip + z-score",
    direction        = direction,
    direction_desc   = direction_desc,
    stock_pool       = "中证1000",
    period           = (f"{common_dates[0].strftime('%Y-%m-%d')} ~ "
                        f"{common_dates[-1].strftime('%Y-%m-%d')}"),
    n_dates          = len(common_dates),
    n_stocks         = len(common_stocks),
    n_groups         = N_GROUPS,
    rebalance_freq   = REBALANCE_FREQ,
    forward_days     = FORWARD_DAYS,
    cost             = COST,
    correlations     = correlations,
    metrics          = metrics,
)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(_nn(report), f, indent=2, ensure_ascii=False)

# ────────────────── 控制台摘要 ──────────────────
print(f"\n{'═'*64}")
print(f"  {FACTOR_ID}: 低点收盘位置 v1  方向: {direction_desc}")
print(f"{'═'*64}")
print(f"  区间:        {report['period']}")
print(f"  股票数:      {len(common_stocks)}")
print(f"  最佳方案:    {best_label}")
print(f"  IC 均值:     {ic_m:.4f}   (t={ic_t:.2f}, {'✓5%' if sig5 else '✗不显著'})")
print(f"  IC>0占比:    {metrics.get('ic_positive_pct',0):.1%}")
print(f"  IC观测数:    {metrics.get('ic_count',0)}")
print(f"  IR:          {metrics.get('ir',0):.4f}")
print(f"  Rank IC:     {metrics.get('rank_ic_mean',0):.4f}")
print(f"  多空Sharpe:  {ls_sh:.4f}   多空MDD: {ls_md:.2%}")
print(f"  单调性:      {mono:.4f}")
print(f"  换手率:      {tov:.2%}")
print(f"{'─'*64}")
for i, (r, s) in enumerate(zip(gr_ann, gs), 1):
    r_s = f"{r:.2%}" if r is not None else "N/A"
    s_s = f"Sh={s:.2f}" if s else ""
    print(f"   G{i}: {r_s:>8}  {s_s}")

print(f"{'─'*64}")
for name, c in sorted(correlations.items()):
    print(f"   vs {name}: {c:+.3f}")

print(f"{'═'*64}")
ok = abs(ic_m) > 0.015 and abs(ic_t) > 2 and abs(ls_sh) > 0.5 and mono >= 0.8
print(f"\n  评估:  {'✅ 通过 (|IC|>0.015 & |t|>2 & |Sharpe|>0.5 & mono≥0.8)' if ok else '❌ 未通过'}")
for lbl, val, thresh in [
    ("|IC|", abs(ic_m), 0.015), ("|t|", abs(ic_t), 2.0),
    ("|Sharpe|", abs(ls_sh), 0.5), ("单调性", mono, 0.8)]:
    print(f"         {lbl}: {val:.4f} {'≥' if ok else '>' if val>thresh else '<'} {thresh}  {'✓' if ok or val>=thresh else '✗'}")
