#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: gap_efficiency_v2 — 跳空缺口效率因子 v2
============================================
v1失败原因: IC=0.0126 < 0.015阈值, t=1.51 < 2
v2改进:
  1. 惰加mean为sum: 累积信号 vs 均值信号, 对A股小盘股可能更有效
  2. 窗口加加40d: 更长的时间尺度过滤噪音
  3. forward_days=20: 与overnight动量一致, 测试中频alpha

来源: gap_efficiency_v1的升级版
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
WINDOW = 40
FORWARD_DAYS = 20
REBALANCE_FREQ = 20
N_GROUPS = 5
COST = 0.002
WINSORIZE_PCT = 0.05
DATA_CUTOFF = "2026-04-15"
FACTOR_ID = "gap_efficiency_v2"

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "csi1000_kline_raw.csv"
SCRIPTS_DIR = BASE_DIR / "skills" / "alpha-factor-lab" / "scripts"
OUTPUT_DIR = BASE_DIR / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据 (cutoff: {DATA_CUTOFF})...")
df = pd.read_csv(DATA_PATH, encoding="utf-8")
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] <= DATA_CUTOFF]
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close = df.pivot_table(index="date", columns="stock_code", values="close")
amount = df.pivot_table(index="date", columns="stock_code", values="amount")
open_p = df.pivot_table(index="date", columns="stock_code", values="open")
high_p = df.pivot_table(index="date", columns="stock_code", values="high")
low_p = df.pivot_table(index="date", columns="stock_code", values="low")
turnover = df.pivot_table(index="date", columns="stock_code", values="turnover")

ret = close.pct_change()
log_amt = np.log(amount.rolling(20).mean().clip(lower=1))

dates = close.index.tolist()
stocks = close.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造跳空效率因子 v2 (window={WINDOW}d, sum instead of mean)...")

overnight_ret = close / open_p - 1          # 隔夜收益率
daily_range = (high_p - low_p) / close.shift(1).clip(lower=0.01)  # 日内总范围

# v2改进: 用sum放缓reinterpret
# gap_efficiency = rolling sum( |overnight_ret| / max(range, threshold) )
gap_intensity = overnight_ret.abs() / daily_range.clip(lower=1e-6)
factor_raw = gap_intensity.rolling(WINDOW, min_periods=int(WINDOW*0.5)).sum()
factor_raw = np.log(factor_raw.clip(lower=1e-8))

print(f"   非空率: {factor_raw.notna().mean().mean():.2%}")
print(f"   均值: {factor_raw.stack().mean():.4f}, std: {factor_raw.stack().std():.4f}")

# ────────────────── 缩尾 ──────────────────
print(f"[3] 缩尾 ({WINSORIZE_PCT*100:.0f}%)...")
for date in dates:
    row = factor_raw.loc[date].dropna()
    if len(row) < 10:
        continue
    lo = row.quantile(WINSORIZE_PCT)
    hi = row.quantile(1 - WINSORIZE_PCT)
    factor_raw.loc[date] = factor_raw.loc[date].clip(lo, hi)

# ────────────────── 中性化 ──────────────────
print(f"[4] 成交额中性化 (OLS)...")
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
    except:
        pass

# ────────────────── 回测 ──────────────────
common_dates = sorted(factor_neutral.dropna(how="all").index.intersection(ret.dropna(how="all").index))
common_stocks = sorted(factor_neutral.columns.intersection(ret.columns))
fa = factor_neutral.loc[common_dates, common_stocks]
ra = ret.loc[common_dates, common_stocks]

sys.path.insert(0, str(SCRIPTS_DIR))
from factor_backtest import compute_group_returns, compute_ic_dynamic, compute_metrics, save_backtest_data

# ────────────────── 方向探索 ──────────────────
print(f"[5] 方向探索...")

ic_pos = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "pearson")
gr_pos, tv_pos, hi_pos = compute_group_returns(fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_pos = compute_metrics(gr_pos, ic_pos, ic_pos, tv_pos, N_GROUPS, holdings_info=hi_pos)

ic_neg = compute_ic_dynamic(-fa, ra, FORWARD_DAYS, "pearson")
gr_neg, tv_neg, hi_neg = compute_group_returns(-fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_neg = compute_metrics(gr_neg, ic_neg, ic_neg, tv_neg, N_GROUPS, holdings_info=hi_neg)

pos_sh = m_pos.get("long_short_sharpe", 0) or 0
neg_sh = m_neg.get("long_short_sharpe", 0) or 0
pos_ic = m_pos.get("ic_mean", 0) or 0
neg_ic = m_neg.get("ic_mean", 0) or 0

print(f"   正向: IC={pos_ic:.4f}, Sharpe={pos_sh:.4f}")
print(f"   反向: IC={neg_ic:.4f}, Sharpe={neg_sh:.4f}")

if neg_sh > pos_sh:
    direction = -1
    fa_use = -fa
    direction_desc = "反向（低累计缺口效率=高预期收益）"
else:
    direction = 1
    fa_use = fa
    direction_desc = "正向（高累计缺口效率=高预期收益）"
print(f"   → 使用{['反向', '正向'][direction > 0]}")

# ────────────────── 最终回测 ──────────────────
print(f"[6] 最终回测 (direction={direction})...")

ic_series = compute_ic_dynamic(fa_use, ra, FORWARD_DAYS, "pearson")
rank_ic_series = compute_ic_dynamic(fa_use, ra, FORWARD_DAYS, "spearman")

group_returns, turnovers, holdings_info = compute_group_returns(
    fa_use, ra, N_GROUPS, REBALANCE_FREQ, COST
)
metrics = compute_metrics(
    group_returns, ic_series, rank_ic_series, turnovers, N_GROUPS,
    holdings_info=holdings_info
)

# ────────────────── 相关性 ──────────────────
print(f"[7] 相关性...")
amihud_raw = ret.abs() / (amount / 1e8).clip(lower=1e-8)
amihud_f = np.log(amihud_raw.rolling(20, min_periods=10).mean().clip(lower=1e-12))

upper_sr = (high_p - np.maximum(close, open_p)) / (high_p - low_p).clip(lower=1e-8)
lower_sr = (np.minimum(close, open_p) - low_p) / (high_p - low_p).clip(lower=1e-8)
shadow_f = (upper_sr - lower_sr).rolling(20, min_periods=10).mean()

oret = open_p / close.shift(1) - 1
oiret = close / open_p - 1
overnight_f = oret.rolling(20, min_periods=10).sum() - oiret.rolling(20, min_periods=10).sum()

cvar_mat = np.full((len(dates), len(stocks)), np.nan)
ret_vals = ret.values
n_d = len(dates)
for i in range(10, n_d):
    w = ret_vals[i-10:i, :]
    s = np.sort(w, axis=0)
    bot2 = np.nanmean(s[:2, :], axis=0)
    vc = np.sum(~np.isnan(w), axis=0)
    bot2[vc < 5] = np.nan
    cvar_mat[i, :] = -bot2
cvar_f = pd.DataFrame(cvar_mat, index=dates, columns=stocks)

neg_freq = (ret <= -0.03).astype(float).rolling(10, min_periods=5).mean()
turnover_level = np.log(turnover.rolling(20, min_periods=10).mean().clip(lower=1e-8))

correlations = {}
for name, other in [
    ('amihud_illiq_v2', amihud_f),
    ('shadow_pressure_v1', shadow_f),
    ('overnight_momentum_v1', overnight_f),
    ('tail_risk_cvar_v1', cvar_f),
    ('neg_day_freq_v1', neg_freq),
    ('turnover_level_v1', turnover_level),
]:
    corrs = []
    for d in common_dates[::10]:
        f1 = fa_use.loc[d].dropna()
        f2 = other.loc[d].reindex(f1.index).dropna()
        c = f1.index.intersection(f2.index)
        if len(c) > 50:
            r, _ = sp_stats.spearmanr(f1[c], f2[c])
            if not np.isnan(r):
                corrs.append(r)
    avg = float(np.mean(corrs)) if corrs else 0
    correlations[name] = round(avg, 3)
    print(f"   vs {name}: r={avg:.3f}")

correlations[FACTOR_ID] = 1.0

# ────────────────── 输出 ──────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_backtest_data(group_returns, ic_series, rank_ic_series, str(OUTPUT_DIR))

def nan_to_none(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    return obj

report = {
    "factor_id": FACTOR_ID,
    "factor_name": "跳空缺口效率 v2",
    "factor_name_en": "Gap Efficiency v2",
    "category": "量价/信息密度",
    "description": f"40日 sum( |隔夜收益| / 日内范围) 取对数, 成交额中性化。累积信号(vs v1均值)。高值=隔夜变动持续高于日内波动=信息效率高。",
    "hypothesis": "隔夜效率高的股票, 少量交易驱动大隔夜变动 → 信息密度高 → 动量延续 → 后续收益好(v1验证了单调性正确方向)",
    "formula": f"neutralize(log(sum(|close/open - 1| / ((H-L)/prev_close), {WINDOW}d)), log_amount_20d)",
    "direction": direction,
    "direction_desc": direction_desc,
    "stock_pool": "中证1000",
    "period": f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}",
    "n_dates": len(common_dates),
    "n_stocks": len(common_stocks),
    "n_groups": N_GROUPS,
    "rebalance_freq": REBALANCE_FREQ,
    "forward_days": FORWARD_DAYS,
    "cost": COST,
    "data_cutoff": DATA_CUTOFF,
    "upgrade_notes": "v2: 重叠signal改为累积(sum), 窗口加加到40d, 20d前瞻中频alpha。",
    "correlations": correlations,
    "metrics": metrics,
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(nan_to_none(report), f, indent=2, ensure_ascii=False)

# ────────────────── 摘要 ──────────────────
ic_m = metrics.get("ic_mean", 0) or 0
ic_t = metrics.get("ic_t_stat", 0) or 0
ls_sh = metrics.get("long_short_sharpe", 0) or 0
ls_md = metrics.get("long_short_mdd", 0) or 0
mono = metrics.get("monotonicity", 0) or 0
sig = "✓" if metrics.get("ic_significant_5pct") else "✗"

print(f"\n{'='*60}")
print(f"  {FACTOR_ID}: 跳空效率 v2")
print(f"  方向: {direction_desc}")
print(f"{'='*60}")
print(f"  区间:     {report['period']}")
print(f"  股票:     {len(common_stocks)}")
print(f"  IC均值:     {ic_m:.4f}  (t={ic_t:.2f}, {sig})")
print(f"  IR:         {metrics.get('ir', 0):.4f}")
print(f"  多空Sharpe: {ls_sh:.4f}")
print(f"  多空MDD:    {ls_md:.2%}")
print(f"  单调性:     {mono:.4f}")
print(f"  换手率:     {metrics.get('turnover_mean', 0):.2%}")
print(f"{'─'*60}")
for i, r in enumerate(metrics.get("group_returns_annualized", []), 1):
    r_str = f"{r:.2%}" if r is not None and not np.isnan(r) else "N/A"
    bar = "█" * max(int((r or 0) * 100), 0)
    print(f"    G{i}: {r_str}  {bar}")
print(f"{'='*60}")
is_valid = abs(ic_m) > 0.015 and abs(ic_t) > 2 and abs(ls_sh) > 0.5
print(f"\n  ➤ 因子{'有效 ✓' if is_valid else '无效 ✗'}")
if is_valid:
    print(f"  → 达标! 写入factors.json并git提交")
else:
    print(f"  → 未达标")
    if abs(ic_m) <= 0.015:
        print(f"    ✗ |IC|={abs(ic_m):.4f} <= 0.015")
    if abs(ic_t) <= 2:
        print(f"    ✗ |t|={abs(ic_t):.2f} <= 2")
    if abs(ls_sh) <= 0.5:
        print(f"    ✗ |Sharpe|={abs(ls_sh):.4f} <= 0.5")
