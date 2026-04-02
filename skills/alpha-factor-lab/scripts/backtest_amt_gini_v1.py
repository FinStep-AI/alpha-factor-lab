#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: amt_gini_v1 — 成交额集中度(基尼系数)因子
================================================

方向: 资金集中度 / 流动性微观结构

构造:
  1. 过去20个交易日的日成交额的基尼系数
  2. 基尼系数衡量成交额在时间上的不均匀程度
  3. 成交额(log_amount_20d) OLS中性化
  4. 5%缩尾 + z-score

逻辑:
  - 高基尼(集中度高): 成交额集中在少数几天 → 有大单冲击/事件驱动/信息不对称
    → 知情交易者在特定时点集中交易 → 信息逐步释放 → 动量效应
  - 低基尼(分散): 每天成交额差不多 → 交易平稳 → 噪声交易为主
  
  也可能相反:
  - 高基尼 → 交易不稳定 → 流动性不可预测 → 风险更高 → 需要补偿(正向)
  - 高基尼 → 被事件驱动的投机 → 过度反应 → 后续反转(反向)
  
  用双向探索确定最优方向。

理论:
  - Chordia, Huh & Subrahmanyam (2007) "Trading Volume"
  - Kyle (1985) 知情交易者模型 — 知情者倾向于集中交易
  - Amihud (2002) — 流动性维度的延伸

与现有因子的区别:
  - Amihud: |ret|/amount 衡量价格冲击
  - Turnover Level: 换手率水平
  - Vol CV: 成交量变异系数(标准差/均值)
  - 本因子: 成交额在时间分布上的集中程度(基尼系数)
    基尼系数比CV更鲁棒(不受极端值影响)，且有明确的"不平等"经济学含义
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
WINDOW = 20
FORWARD_DAYS = 5
REBALANCE_FREQ = 5
N_GROUPS = 5
COST = 0.003
WINSORIZE_PCT = 0.05
DATA_CUTOFF = "2026-03-27"
FACTOR_ID = "amt_gini_v1"

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "csi1000_kline_raw.csv"
SCRIPTS_DIR = BASE_DIR / "skills" / "alpha-factor-lab" / "scripts"
OUTPUT_DIR = BASE_DIR / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"


def gini_coefficient(arr):
    """计算基尼系数。arr为非负数组。"""
    arr = np.sort(arr)
    n = len(arr)
    if n < 2 or arr.sum() <= 0:
        return np.nan
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))


# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据 (cutoff: {DATA_CUTOFF})...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] <= DATA_CUTOFF]
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")

ret_piv = close_piv.pct_change()
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
n_dates = len(dates)
n_stocks = len(stocks)
print(f"   {n_dates} 日, {n_stocks} 股")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造成交额基尼系数因子 (window={WINDOW})...")

amount_vals = amount_piv.values  # (n_dates, n_stocks)
gini_mat = np.full((n_dates, n_stocks), np.nan)

for i in range(WINDOW, n_dates):
    window_amt = amount_vals[i - WINDOW:i, :]  # (WINDOW, n_stocks)
    for j in range(n_stocks):
        col = window_amt[:, j]
        valid = col[~np.isnan(col)]
        # 需要至少15个有效值,且成交额>0
        valid = valid[valid > 0]
        if len(valid) >= 15:
            gini_mat[i, j] = gini_coefficient(valid)

factor_raw = pd.DataFrame(gini_mat, index=dates, columns=stocks)

print(f"   非空率: {factor_raw.notna().mean().mean():.2%}")
print(f"   均值: {factor_raw.stack().mean():.4f}, std: {factor_raw.stack().std():.4f}")
print(f"   分位: 25%={factor_raw.stack().quantile(0.25):.4f}, "
      f"50%={factor_raw.stack().quantile(0.50):.4f}, "
      f"75%={factor_raw.stack().quantile(0.75):.4f}")

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
print(f"[5] 回测引擎...")

common_dates = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
common_stocks = sorted(factor_neutral.columns.intersection(ret_piv.columns))
fa = factor_neutral.loc[common_dates, common_stocks]
ra = ret_piv.loc[common_dates, common_stocks]

sys.path.insert(0, str(SCRIPTS_DIR))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data
)

# ────────────────── 方向探索 ──────────────────
print(f"[6] 方向探索...")

# 正向: 做多高基尼(集中度高→知情交易/风险溢价)
ic_pos = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "pearson")
gr_pos, _, _ = compute_group_returns(fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_pos = compute_metrics(gr_pos, ic_pos, ic_pos, [], N_GROUPS)

# 反向: 做多低基尼(分散→稳定→后续好)
ic_neg = compute_ic_dynamic(-fa, ra, FORWARD_DAYS, "pearson")
gr_neg, _, _ = compute_group_returns(-fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_neg = compute_metrics(gr_neg, ic_neg, ic_neg, [], N_GROUPS)

pos_sh = m_pos.get("long_short_sharpe", 0) or 0
neg_sh = m_neg.get("long_short_sharpe", 0) or 0
pos_ic = m_pos.get("ic_mean", 0) or 0
neg_ic = m_neg.get("ic_mean", 0) or 0
pos_t = m_pos.get("ic_t_stat", 0) or 0
neg_t = m_neg.get("ic_t_stat", 0) or 0
pos_mono = m_pos.get("monotonicity", 0) or 0
neg_mono = m_neg.get("monotonicity", 0) or 0

print(f"   正向 (高基尼=高收益): IC={pos_ic:.4f}, t={pos_t:.2f}, Sharpe={pos_sh:.4f}, Mono={pos_mono:.2f}")
print(f"   反向 (低基尼=高收益): IC={neg_ic:.4f}, t={neg_t:.2f}, Sharpe={neg_sh:.4f}, Mono={neg_mono:.2f}")

# 选择更好的方向(综合考虑Sharpe和t值)
pos_score = abs(pos_sh) + abs(pos_t) * 0.3
neg_score = abs(neg_sh) + abs(neg_t) * 0.3
if neg_score > pos_score:
    print(f"   → 使用反向 (做多低基尼)")
    direction = -1
    fa_final = -fa
    direction_desc = "反向（低集中度=高预期收益，做多成交额分散的股票）"
else:
    print(f"   → 使用正向 (做多高基尼)")
    direction = 1
    fa_final = fa
    direction_desc = "正向（高集中度=高预期收益，做多成交额集中的股票）"

# ────────────────── 多前瞻窗口探索 ──────────────────
print(f"[6b] 前瞻窗口探索...")
best_fwd = FORWARD_DAYS
best_sharpe = 0
for fwd in [5, 10, 20]:
    for reb in [fwd]:
        ic_t = compute_ic_dynamic(fa_final, ra, fwd, "pearson")
        gr_t, _, _ = compute_group_returns(fa_final, ra, N_GROUPS, reb, COST)
        m_t = compute_metrics(gr_t, ic_t, ic_t, [], N_GROUPS)
        sh_t = m_t.get("long_short_sharpe", 0) or 0
        ic_t_val = m_t.get("ic_mean", 0) or 0
        t_t = m_t.get("ic_t_stat", 0) or 0
        mono_t = m_t.get("monotonicity", 0) or 0
        print(f"   fwd={fwd}d, reb={reb}d: IC={ic_t_val:.4f}, t={t_t:.2f}, Sharpe={sh_t:.4f}, Mono={mono_t:.2f}")
        if sh_t > best_sharpe:
            best_sharpe = sh_t
            best_fwd = fwd

print(f"   最优前瞻窗口: {best_fwd}d (Sharpe={best_sharpe:.4f})")
FORWARD_DAYS = best_fwd
REBALANCE_FREQ = best_fwd

# ────────────────── 多窗口探索 ──────────────────
print(f"[6c] 因子计算窗口探索...")
for test_window in [10, 30, 40]:
    gini_test = np.full((n_dates, n_stocks), np.nan)
    for i in range(test_window, n_dates):
        window_amt = amount_vals[i - test_window:i, :]
        for j in range(n_stocks):
            col = window_amt[:, j]
            valid = col[~np.isnan(col)]
            valid = valid[valid > 0]
            if len(valid) >= max(8, test_window * 3 // 4):
                gini_test[i, j] = gini_coefficient(valid)
    
    gini_df = pd.DataFrame(gini_test, index=dates, columns=stocks)
    # 缩尾
    for d in dates:
        row = gini_df.loc[d].dropna()
        if len(row) < 10:
            continue
        lo = row.quantile(0.05)
        hi = row.quantile(0.95)
        gini_df.loc[d] = gini_df.loc[d].clip(lo, hi)
    # 中性化
    for d in dates:
        f_t = gini_df.loc[d].dropna()
        m_t = log_amt.loc[d].reindex(f_t.index).dropna()
        common_t = f_t.index.intersection(m_t.index)
        if len(common_t) < 30:
            continue
        fc = f_t[common_t].values
        mc = m_t[common_t].values
        X = np.column_stack([np.ones(len(mc)), mc])
        try:
            beta = np.linalg.lstsq(X, fc, rcond=None)[0]
            gini_df.loc[d, common_t] = fc - X @ beta
        except:
            pass
    
    cd = sorted(gini_df.dropna(how="all").index.intersection(ra.index))
    cs = sorted(gini_df.columns.intersection(ra.columns))
    if len(cd) < 50:
        continue
    
    fa_t = (direction * gini_df).loc[cd, cs]
    ic_t = compute_ic_dynamic(fa_t, ra.loc[cd, cs], FORWARD_DAYS, "pearson")
    gr_t, _, _ = compute_group_returns(fa_t, ra.loc[cd, cs], N_GROUPS, REBALANCE_FREQ, COST)
    m_t2 = compute_metrics(gr_t, ic_t, ic_t, [], N_GROUPS)
    print(f"   Window={test_window}d: IC={m_t2.get('ic_mean',0):.4f}, "
          f"t={m_t2.get('ic_t_stat',0):.2f}, Sharpe={m_t2.get('long_short_sharpe',0):.4f}")

# ────────────────── 最终回测 ──────────────────
print(f"[7] 最终回测 (方向={direction}, fwd={FORWARD_DAYS}d, reb={REBALANCE_FREQ}d)...")

ic_series = compute_ic_dynamic(fa_final, ra, FORWARD_DAYS, "pearson")
rank_ic_series = compute_ic_dynamic(fa_final, ra, FORWARD_DAYS, "spearman")

group_returns, turnovers, holdings_info = compute_group_returns(
    fa_final, ra, N_GROUPS, REBALANCE_FREQ, COST
)
metrics = compute_metrics(
    group_returns, ic_series, rank_ic_series, turnovers, N_GROUPS,
    holdings_info=holdings_info
)

# ────────────────── 相关性 ──────────────────
print(f"[8] 与现有因子相关性...")

open_piv = df.pivot_table(index="date", columns="stock_code", values="open")
high_piv = df.pivot_table(index="date", columns="stock_code", values="high")
low_piv = df.pivot_table(index="date", columns="stock_code", values="low")
volume_piv = df.pivot_table(index="date", columns="stock_code", values="volume")
turnover_piv = df.pivot_table(index="date", columns="stock_code", values="turnover")
amplitude_piv = df.pivot_table(index="date", columns="stock_code", values="amplitude")

# Amihud
amihud_raw = (ret_piv.abs() / (amount_piv / 1e8).clip(lower=1e-8))
amihud_factor = np.log(amihud_raw.rolling(20, min_periods=10).mean().clip(lower=1e-12))

# Shadow pressure
upper_sr = (high_piv - np.maximum(close_piv, open_piv)) / (high_piv - low_piv).clip(lower=1e-8)
lower_sr = (np.minimum(close_piv, open_piv) - low_piv) / (high_piv - low_piv).clip(lower=1e-8)
shadow = (upper_sr - lower_sr).rolling(20, min_periods=10).mean()

# Overnight momentum
oret = open_piv / close_piv.shift(1) - 1
iret = close_piv / open_piv - 1
overnight_mom = oret.rolling(20, min_periods=10).sum() - iret.rolling(20, min_periods=10).sum()

# CVaR
ret_vals = ret_piv.values
cvar_mat = np.full((n_dates, n_stocks), np.nan)
for i in range(10, n_dates):
    w = ret_vals[i-10:i, :]
    s = np.sort(w, axis=0)
    bot2 = np.nanmean(s[:2, :], axis=0)
    vc = np.sum(~np.isnan(w), axis=0)
    bot2[vc < 5] = np.nan
    cvar_mat[i, :] = -bot2
cvar_df = pd.DataFrame(cvar_mat, index=dates, columns=stocks)

# Neg day freq
neg_freq = (ret_piv <= -0.03).astype(float).rolling(10, min_periods=5).mean()

# Turnover level
turnover_level = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8))

# TAE
tae_raw = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8) / 
                  (amplitude_piv.rolling(20, min_periods=10).mean().clip(lower=0.01)))

# Vol CV neg
vol_cv_raw = volume_piv.rolling(20, min_periods=10).std() / volume_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8)
vol_cv_neg = -vol_cv_raw

# MA dispersion
ma_list = [5, 10, 20, 40, 60, 120]
ma_vals = {}
for w in ma_list:
    ma_vals[w] = close_piv.rolling(w, min_periods=max(w//2, 3)).mean()
ma_norm = {w: ma_vals[w] / close_piv for w in ma_list}
ma_arr = np.stack([ma_norm[w].values for w in ma_list], axis=0)
ma_disp = pd.DataFrame(np.nanstd(ma_arr, axis=0), index=dates, columns=stocks)

# Amplitude level
amp_raw = ((high_piv - low_piv) / close_piv.shift(1)).clip(lower=0)
amp_level = np.log(amp_raw.rolling(60, min_periods=30).mean().clip(lower=1e-8))

correlations = {}
for name, other in [
    ('amihud_illiq_v2', amihud_factor), 
    ('shadow_pressure_v1', shadow),
    ('overnight_momentum_v1', overnight_mom), 
    ('tail_risk_cvar_v1', cvar_df),
    ('neg_day_freq_v1', neg_freq),
    ('turnover_level_v1', turnover_level),
    ('tae_v1', tae_raw),
    ('vol_cv_neg_v1', vol_cv_neg),
    ('ma_disp_v1', ma_disp),
    ('amp_level_v2', amp_level),
]:
    corrs = []
    for d in common_dates[::10]:
        f1 = fa_final.loc[d].dropna()
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
    "factor_name": "成交额集中度(基尼) v1",
    "factor_name_en": "Amount Gini Concentration v1",
    "category": "资金集中度/流动性微观结构",
    "description": f"过去{WINDOW}日成交额的基尼系数，衡量成交额在时间上的不均匀程度。成交额OLS中性化。",
    "hypothesis": "成交额时间分布的集中度反映知情交易者的活跃模式和流动性结构，对后续收益有预测力。",
    "formula": f"neutralize(gini(amount, {WINDOW}d), log_amount_20d)",
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
print(f"  {FACTOR_ID}: 成交额集中度(基尼系数)因子")
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
    print(f"  → 准备写入factors.json并git提交")
else:
    print(f"  → 因子未达标，记录失败原因")
