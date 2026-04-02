#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: breakout_intensity_v1 — 突破强度因子
==========================================
方向: 趋势/动量 (频率维度)

构造:
  1. 对每只股票每天, 判断close > rolling_max(close, 5d shift 1) (向上突破)
     或 close < rolling_min(close, 5d shift 1) (向下突破)
  2. 20日窗口: up_break_ratio - down_break_ratio
  3. 成交额中性化(OLS)
  4. 5%缩尾

逻辑:
  传统动量看累计涨幅, 本因子看"突破频率":
  - 频繁向上突破(close超过近期高点) = 持续创新高 = 趋势清晰/机构持续买入
  - 频繁向下突破 = 持续创新低 = 持续抛压
  - 二者之差 = 净突破方向
  
  与传统动量的区别:
  - 动量因子看幅度: 涨了多少
  - 本因子看频率: 多少次突破了近期高/低点
  - 一只股票可能涨幅不大但频繁刷新近期高点(窄幅上涨通道) → 本因子捕捉这种特征
  
  学术基础:
  - Donchian Channel Breakout (Richard Donchian, 1960s) — 趋势跟踪经典方法
  - George & Hwang (2004) "52-Week High and Momentum Investing" — 价格相对于历史高点的位置预测收益
  - Rachev et al. (2007) — 突破频率与趋势持续性的关系
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
LOOKBACK = 5          # 高低点回看窗口
WINDOW = 20           # 突破频率统计窗口
FORWARD_DAYS = 5      # 前瞻收益
REBALANCE_FREQ = 5    # 调仓频率
N_GROUPS = 5
COST = 0.003
WINSORIZE_PCT = 0.05
DATA_CUTOFF = "2026-03-24"
FACTOR_ID = "breakout_intensity_v1"

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "csi1000_kline_raw.csv"
SCRIPTS_DIR = BASE_DIR / "skills" / "alpha-factor-lab" / "scripts"
OUTPUT_DIR = BASE_DIR / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据 (cutoff: {DATA_CUTOFF})...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] <= DATA_CUTOFF]
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")
open_piv = df.pivot_table(index="date", columns="stock_code", values="open")
high_piv = df.pivot_table(index="date", columns="stock_code", values="high")
low_piv = df.pivot_table(index="date", columns="stock_code", values="low")

ret_piv = close_piv.pct_change()
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造突破强度因子 (lookback={LOOKBACK}d, window={WINDOW}d)...")

# Rolling max/min of close over past LOOKBACK days (shifted by 1 to avoid look-ahead)
rolling_high = close_piv.shift(1).rolling(LOOKBACK, min_periods=3).max()
rolling_low = close_piv.shift(1).rolling(LOOKBACK, min_periods=3).min()

# Up breakout: close > rolling_high (today's close exceeds recent N-day high)
up_break = (close_piv > rolling_high).astype(float)
# Down breakout: close < rolling_low
down_break = (close_piv < rolling_low).astype(float)

# Net breakout ratio over WINDOW days
up_ratio = up_break.rolling(WINDOW, min_periods=10).mean()
down_ratio = down_break.rolling(WINDOW, min_periods=10).mean()
factor_raw = up_ratio - down_ratio

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
print(f"[5] 回测: {N_GROUPS}组, {REBALANCE_FREQ}d调仓, {COST*100:.1f}%成本...")

common_dates = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
common_stocks = sorted(factor_neutral.columns.intersection(ret_piv.columns))
fa = factor_neutral.loc[common_dates, common_stocks]
ra = ret_piv.loc[common_dates, common_stocks]

sys.path.insert(0, str(SCRIPTS_DIR))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data
)

# 方向确认
print(f"[6] 方向确认...")
ic_pos = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "pearson")
gr_pos, turn_pos, hi_pos = compute_group_returns(fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_pos = compute_metrics(gr_pos, ic_pos, ic_pos, turn_pos, N_GROUPS, holdings_info=hi_pos)

ic_neg = compute_ic_dynamic(-fa, ra, FORWARD_DAYS, "pearson")
gr_neg, turn_neg, hi_neg = compute_group_returns(-fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_neg = compute_metrics(gr_neg, ic_neg, ic_neg, turn_neg, N_GROUPS, holdings_info=hi_neg)

pos_sh = m_pos.get("long_short_sharpe", 0) or 0
neg_sh = m_neg.get("long_short_sharpe", 0) or 0
print(f"   正向Sharpe={pos_sh:.4f} (高净突破=高收益，趋势跟踪)")
print(f"   反向Sharpe={neg_sh:.4f} (低净突破=高收益，反转)")

if neg_sh > pos_sh:
    print(f"   → 反向更好 (反转逻辑)")
    direction = -1
    fa_use = -fa
    ic_series = ic_neg
    rank_ic_series = compute_ic_dynamic(-fa, ra, FORWARD_DAYS, "spearman")
    group_returns = gr_neg
    turnovers = turn_neg
    holdings_info = hi_neg
    metrics = m_neg
    direction_desc = "反向（低/负净突破=高预期收益，突破反转）"
else:
    print(f"   → 正向更好 (趋势跟踪逻辑)")
    direction = 1
    fa_use = fa
    ic_series = ic_pos
    rank_ic_series = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "spearman")
    group_returns = gr_pos
    turnovers = turn_pos
    holdings_info = hi_pos
    metrics = m_pos
    direction_desc = "正向（高净突破=高预期收益，趋势跟踪）"

# ────────────────── 也测20d前瞻 ──────────────────
print(f"[6b] 额外测试20日前瞻...")
ic_20d = compute_ic_dynamic(fa_use, ra, 20, "pearson")
gr_20d, turn_20d, hi_20d = compute_group_returns(fa_use, ra, N_GROUPS, 20, COST)
m_20d = compute_metrics(gr_20d, ic_20d, ic_20d, turn_20d, N_GROUPS, holdings_info=hi_20d)
sh_20d = m_20d.get("long_short_sharpe", 0) or 0
ic_20d_m = m_20d.get("ic_mean", 0) or 0
print(f"   20d前瞻: IC={ic_20d_m:.4f}, Sharpe={sh_20d:.4f}")

# ────────────────── 相关性 ──────────────────
print(f"[7] 与现有因子相关性...")

turnover_piv = df.pivot_table(index="date", columns="stock_code", values="turnover")

amihud_raw = (ret_piv.abs() / (amount_piv / 1e8).clip(lower=1e-8))
amihud_factor = np.log(amihud_raw.rolling(20, min_periods=10).mean().clip(lower=1e-12))

upper_sr = (high_piv - np.maximum(close_piv, open_piv)) / (high_piv - low_piv).clip(lower=1e-8)
lower_sr = (np.minimum(close_piv, open_piv) - low_piv) / (high_piv - low_piv).clip(lower=1e-8)
shadow = (upper_sr - lower_sr).rolling(20, min_periods=10).mean()

oret = open_piv / close_piv.shift(1) - 1
iret = close_piv / open_piv - 1
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

amplitude_piv = (high_piv - low_piv) / close_piv.shift(1).clip(lower=0.01)
tae = np.log((turnover_piv.rolling(20, min_periods=10).mean()) / 
             (amplitude_piv.rolling(20, min_periods=10).mean().clip(lower=0.01)))
amp_level = np.log(amplitude_piv.rolling(60, min_periods=30).mean().clip(lower=1e-8))

# 20日简单动量(作为参照)
mom_20d = close_piv.pct_change(20)

correlations = {}
for name, other in [
    ('amihud_illiq_v2', amihud_factor), 
    ('shadow_pressure_v1', shadow),
    ('overnight_momentum_v1', overnight_mom), 
    ('tail_risk_cvar_v1', cvar_df),
    ('neg_day_freq_v1', neg_freq),
    ('turnover_level_v1', turnover_level),
    ('tae_v1', tae),
    ('amp_level_v2', amp_level),
    ('momentum_20d', mom_20d),
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
    "factor_name": "突破强度 v1",
    "factor_name_en": "Breakout Intensity v1",
    "category": "趋势/动量(频率维度)",
    "description": f"过去{WINDOW}日中close突破前{LOOKBACK}日高点的频率 - 突破前{LOOKBACK}日低点的频率。衡量净突破方向，即趋势的频率特征而非幅度特征。",
    "hypothesis": "频繁向上突破近期高点=趋势清晰/机构持续买入，频繁向下突破=持续抛压。净突破频率是动量的频率维度补充。",
    "formula": f"neutralize(MA{WINDOW}(close>rolling_max(close,{LOOKBACK}d)) - MA{WINDOW}(close<rolling_min(close,{LOOKBACK}d)), log_amount_20d)",
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
    "correlations": correlations,
    "metrics": metrics,
    "alt_20d": {
        "ic_mean": m_20d.get("ic_mean"),
        "ic_t_stat": m_20d.get("ic_t_stat"),
        "long_short_sharpe": m_20d.get("long_short_sharpe"),
    }
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
print(f"  {FACTOR_ID}: 突破强度因子")
print(f"  方向: {direction_desc}")
print(f"{'='*60}")
print(f"  区间:     {report['period']}")
print(f"  股票:     {len(common_stocks)}")
print(f"  IC均值:     {ic_m:.4f}  (t={ic_t:.2f}, {sig})")
print(f"  Rank IC:    {metrics.get('rank_ic_mean', 0):.4f}")
print(f"  IR:         {metrics.get('ir', 0):.4f}")
print(f"  IC>0占比:   {metrics.get('ic_positive_pct', 0):.1%}")
print(f"  IC观测数:   {metrics.get('ic_count', 0)}")
print(f"  多空Sharpe: {ls_sh:.4f}")
print(f"  多空MDD:    {ls_md:.2%}")
print(f"  单调性:     {mono:.4f}")
print(f"  换手率:     {metrics.get('turnover_mean', 0):.2%}")
print(f"{'─'*60}")
print(f"  分层年化收益:")
for i, r in enumerate(metrics.get("group_returns_annualized", []), 1):
    r_str = f"{r:.2%}" if r is not None else "N/A"
    bar = "█" * max(int((r or 0) * 100), 0)
    print(f"    G{i}: {r_str}  {bar}")

for key in sorted(group_returns.keys(), key=lambda x: str(x)):
    cum = (1 + group_returns[key]).cumprod()
    print(f"  {key} NAV: {cum.iloc[-1]:.4f}")

print(f"{'='*60}")
is_valid = abs(ic_m) > 0.015 and abs(ic_t) > 2 and abs(ls_sh) > 0.5
print(f"\n  ➤ 因子{'有效 ✓' if is_valid else '无效 ✗'}")
print(f"  评估标准: |IC|>0.015, |t|>2, |Sharpe|>0.5")

if is_valid:
    print(f"\n  → 因子达标! 准备写入factors.json并git提交")
else:
    print(f"\n  → 因子未达标, 记录失败原因")
    if abs(ic_m) <= 0.015:
        print(f"    ✗ |IC|={abs(ic_m):.4f} <= 0.015")
    if abs(ic_t) <= 2:
        print(f"    ✗ |t|={abs(ic_t):.2f} <= 2")
    if abs(ls_sh) <= 0.5:
        print(f"    ✗ |Sharpe|={abs(ls_sh):.4f} <= 0.5")
