#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: realized_skew_v1 — 已实现偏度因子
==========================================

方向: 波动率偏度 / 横截面定价异象

构造:
  1. 过去20个交易日的日收益率三阶中心矩 / std^3 (标准化偏度)
  2. 市值(成交额)中性化(OLS)
  3. 5%缩尾
  4. 反向使用: 做多低偏度(负偏=左尾肥), 做空高偏度(正偏=彩票型)

逻辑:
  偏度反映收益分布的不对称性:
  - 正偏度(右尾肥): "彩票型"股票, 小概率大涨, 散户偏好→被高估→未来收益低
  - 负偏度(左尾肥): 经常小涨偶尔大跌, 风险暴露大→要求风险补偿→未来收益高
  
  A股中证1000小盘股散户比例高, 彩票偏好效应更显著:
  → 高偏度 → 散户追捧 → 高估 → 后续收益差
  → 低偏度 → 被忽视 → 低估 → 后续收益好

理论:
  - Boyer, Mitton & Vorkink (2010) "Expected Idiosyncratic Skewness" RFS
  - Bali, Cakici & Whitelaw (2011) "Maxing Out" JFE (MAX effect)
  - 张峥 & 刘力 (2006) "A股个人投资者彩票偏好与IPO溢价"
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
DATA_CUTOFF = "2026-03-13"
FACTOR_ID = "realized_skew_v1"

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

ret_piv = close_piv.pct_change()
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造已实现偏度因子 (window={WINDOW})...")

# 方法: 滚动窗口计算标准化偏度
# skew = E[(r - μ)^3] / σ^3
# 使用 scipy.stats.skew 的 fisher=True (即减去正态偏度0)

n_dates = len(dates)
n_stocks = len(stocks)
ret_vals = ret_piv.values  # (n_dates, n_stocks)

skew_mat = np.full((n_dates, n_stocks), np.nan)

for i in range(WINDOW, n_dates):
    window_rets = ret_vals[i - WINDOW:i, :]  # (WINDOW, n_stocks)
    # 每列计算偏度
    for j in range(n_stocks):
        col = window_rets[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) >= 10:  # 至少10个有效观测
            std = np.std(valid, ddof=1)
            if std > 1e-8:
                mean = np.mean(valid)
                skew_mat[i, j] = np.mean((valid - mean) ** 3) / (std ** 3)

factor_raw = pd.DataFrame(skew_mat, index=dates, columns=stocks)

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

# ────────────────── 方向探索 ──────────────────
print(f"[6] 方向探索...")

# 正向: 做多高偏度
ic_pos = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "pearson")
gr_pos, _, _ = compute_group_returns(fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_pos = compute_metrics(gr_pos, ic_pos, ic_pos, [], N_GROUPS)

# 反向: 做多低偏度 (即 -skew)
ic_neg = compute_ic_dynamic(-fa, ra, FORWARD_DAYS, "pearson")
gr_neg, _, _ = compute_group_returns(-fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_neg = compute_metrics(gr_neg, ic_neg, ic_neg, [], N_GROUPS)

pos_sh = m_pos.get("long_short_sharpe", 0) or 0
neg_sh = m_neg.get("long_short_sharpe", 0) or 0
pos_ic = m_pos.get("ic_mean", 0) or 0
neg_ic = m_neg.get("ic_mean", 0) or 0

print(f"   正向 (高偏度=高收益): IC={pos_ic:.4f}, Sharpe={pos_sh:.4f}")
print(f"   反向 (低偏度=高收益): IC={neg_ic:.4f}, Sharpe={neg_sh:.4f}")

# 选择更好的方向
if neg_sh > pos_sh:
    print(f"   → 使用反向 (做多低偏度, 做空高偏度)")
    direction = -1
    fa_final = -fa
    direction_desc = "反向（低偏度=高预期收益，做空彩票型股票）"
else:
    print(f"   → 使用正向 (高偏度=高收益)")
    direction = 1
    fa_final = fa
    direction_desc = "正向（高偏度=高预期收益）"

# ────────────────── 最终回测 ──────────────────
print(f"[7] 最终回测 (方向={direction})...")

ic_series = compute_ic_dynamic(fa_final, ra, FORWARD_DAYS, "pearson")
rank_ic_series = compute_ic_dynamic(fa_final, ra, FORWARD_DAYS, "spearman")

group_returns, turnovers, holdings_info = compute_group_returns(
    fa_final, ra, N_GROUPS, REBALANCE_FREQ, COST
)
metrics = compute_metrics(
    group_returns, ic_series, rank_ic_series, turnovers, N_GROUPS,
    holdings_info=holdings_info
)

# ────────────────── 不同窗口对比 ──────────────────
print(f"[8] 窗口敏感性分析...")
for test_window in [10, 30, 40]:
    skew_test = np.full((n_dates, n_stocks), np.nan)
    for i in range(test_window, n_dates):
        window_rets = ret_vals[i - test_window:i, :]
        for j in range(n_stocks):
            col = window_rets[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) >= max(10, test_window // 2):
                std = np.std(valid, ddof=1)
                if std > 1e-8:
                    mean = np.mean(valid)
                    skew_test[i, j] = np.mean((valid - mean) ** 3) / (std ** 3)
    
    skew_df = pd.DataFrame(skew_test, index=dates, columns=stocks)
    # 简单缩尾
    for d in dates:
        row = skew_df.loc[d].dropna()
        if len(row) < 10:
            continue
        lo = row.quantile(0.05)
        hi = row.quantile(0.95)
        skew_df.loc[d] = skew_df.loc[d].clip(lo, hi)
    
    # 中性化
    for d in dates:
        f_test = skew_df.loc[d].dropna()
        m_test = log_amt.loc[d].reindex(f_test.index).dropna()
        common_t = f_test.index.intersection(m_test.index)
        if len(common_t) < 30:
            continue
        fc = f_test[common_t].values
        mc = m_test[common_t].values
        X = np.column_stack([np.ones(len(mc)), mc])
        try:
            beta = np.linalg.lstsq(X, fc, rcond=None)[0]
            skew_df.loc[d, common_t] = fc - X @ beta
        except:
            pass
    
    cd = sorted(skew_df.dropna(how="all").index.intersection(ra.index))
    cs = sorted(skew_df.columns.intersection(ra.columns))
    if len(cd) < 50:
        continue
    
    fa_t = (direction * skew_df).loc[cd, cs]
    ic_t = compute_ic_dynamic(fa_t, ra.loc[cd, cs], FORWARD_DAYS, "pearson")
    gr_t, _, _ = compute_group_returns(fa_t, ra.loc[cd, cs], N_GROUPS, REBALANCE_FREQ, COST)
    m_t = compute_metrics(gr_t, ic_t, ic_t, [], N_GROUPS)
    
    print(f"   Window={test_window}d: IC={m_t.get('ic_mean',0):.4f}, "
          f"t={m_t.get('ic_t_stat',0):.2f}, Sharpe={m_t.get('long_short_sharpe',0):.4f}")

# ────────────────── 相关性 ──────────────────
print(f"[9] 与现有因子相关性...")

# 构造已有因子
open_piv = df.pivot_table(index="date", columns="stock_code", values="open")
high_piv = df.pivot_table(index="date", columns="stock_code", values="high")
low_piv = df.pivot_table(index="date", columns="stock_code", values="low")
turnover_piv = df.pivot_table(index="date", columns="stock_code", values="turnover")

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
amplitude_piv = df.pivot_table(index="date", columns="stock_code", values="amplitude")
tae_raw = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8) / 
                  (amplitude_piv.rolling(20, min_periods=10).mean().clip(lower=0.01)))

correlations = {}
for name, other in [
    ('amihud_illiq_v2', amihud_factor), 
    ('shadow_pressure_v1', shadow),
    ('overnight_momentum_v1', overnight_mom), 
    ('tail_risk_cvar_v1', cvar_df),
    ('neg_day_freq_v1', neg_freq),
    ('turnover_level_v1', turnover_level),
    ('tae_v1', tae_raw),
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
    "factor_name": "已实现偏度 v1",
    "factor_name_en": "Realized Skewness v1",
    "category": "波动率偏度",
    "description": f"过去{WINDOW}日日收益率的标准化偏度(三阶中心矩/σ³)，成交额中性化。反映收益分布不对称性——正偏度=彩票型(被高估)，负偏度=左尾风险(被低估)。",
    "hypothesis": "高偏度(正偏=彩票型)股票被散户追捧而高估，低偏度(负偏=左尾)股票要求更高风险补偿。A股中证1000散户比例高，彩票偏好效应更显著。",
    "formula": f"neutralize(skew(daily_ret, {WINDOW}d), log_amount_20d)",
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
print(f"  {FACTOR_ID}: 已实现偏度因子")
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
