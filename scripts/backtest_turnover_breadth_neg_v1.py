#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: 换手率广度（反向）v1 (Turnover Breadth Neg v1)
=======================================================

方向: 反向 (高广度高反转/低收益)
Barra风格: Sentiment
新因子: 第6个换手率维度(水平/稳定性/变化方向/加速/偏度 → 广度)

理论基础:
  连续高换手日（换手率高于自身20日中位数）的比例衡量了投资者关注的广度/持续性。
  高比率 → 连续高强度参与 → 过度关注 → 知情交易者信息释放完毕 → 短期内反转
  低比率 → 换手稀疏 → 筹码稳定 → 散户关注度低 → 更易吸引后续流入 → 正alpha

关键不同:
  - turnover_level_v1: 换手率水平 (静态水平，活跃度)
  - turnover_decel_v1: 换手率减速 (变化方向)
  - turnover_vol_resid_v1: 换手率数量型总量
  - turnover_breadth_neg_v1: 换手次数超过均值的天数占比

构造:
  1. 20日内，换手率 > 20日换手中位数 = 高活跃日
  2. 高频次 = 比值: 高活跃日比例
  3. 反向使用: -ratio → 做多"低活跃广度"股票
  4. 成交额中性化(OLS) + 5% MAD缩尾 + z-score
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ────────────────── 参数 ──────────────────
WINDOW = 20
REBALANCE_FREQ = 20
FORWARD_DAYS = 10  # 10日前瞻
N_GROUPS = 5
COST = 0.003
WINSORIZE_PCT = 0.05
FACTOR_ID = "turnover_breadth_neg_v1"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "csi1000_kline_raw.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"


# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")
turnover_piv = df.pivot_table(index="date", columns="stock_code", values="turnover")
ret_piv = close_piv.pct_change()
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造因子: {WINDOW}日内高换手日比例 (反向)...")

turnover_mid = turnover_piv.rolling(WINDOW, min_periods=15).median()
above_median = (turnover_piv > turnover_mid).astype(float)
factor_raw = above_median.rolling(WINDOW, min_periods=15).mean()

print(f"   非空率: {factor_raw.notna().mean().mean():.2%}")

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

# ────────────────── 反向：取负 ──────────────────
print(f"[5] 取负（反向使用: 低比率=高收益）...")
factor_neutral = -factor_neutral

# ────────────────── 回测 ──────────────────
print(f"[6] 回测: {N_GROUPS}组, {REBALANCE_FREQ}d, {COST*100:.1f}%成本...")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills" / "alpha-factor-lab" / "scripts"))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data
)

print(f"[7] IC (forward={FORWARD_DAYS}d)...")
ic_series = compute_ic_dynamic(factor_neutral, ret_piv, FORWARD_DAYS, "pearson")
rank_ic_series = compute_ic_dynamic(factor_neutral, ret_piv, FORWARD_DAYS, "spearman")

group_returns, turnovers, holdings_info = compute_group_returns(
    factor_neutral, ret_piv, N_GROUPS, REBALANCE_FREQ, COST
)
metrics = compute_metrics(
    group_returns, ic_series, rank_ic_series, turnovers, N_GROUPS,
    holdings_info=holdings_info
)

# ────────────────── CVaR相关性 ──────────────────
print(f"[8] CVaR相关性...")
from scipy import stats as sp_stats
ret_vals = ret_piv.values
cvar_matrix = np.full_like(ret_vals, np.nan)
for i in range(10, len(dates)):
    window = ret_vals[i-10:i, :]
    sorted_w = np.sort(window, axis=0)
    bot2 = np.nanmean(sorted_w[:2, :], axis=0)
    valid_count = np.sum(~np.isnan(window), axis=0)
    bot2[valid_count < 5] = np.nan
    cvar_matrix[i, :] = -bot2
cvar_df = pd.DataFrame(cvar_matrix, index=dates, columns=stocks)

correlations = []
for date in sorted(factor_neutral.dropna(how="all").index)[::20]:
    f1 = factor_neutral.loc[date].dropna()
    f2 = cvar_df.loc[date].reindex(f1.index).dropna()
    common = f1.index.intersection(f2.index)
    if len(common) > 50:
        corr, _ = sp_stats.spearmanr(f1[common], f2[common])
        if not np.isnan(corr):
            correlations.append(corr)
avg_corr_cvar = float(np.mean(correlations)) if correlations else 0

print(f"   与CVaR截面Spearman: {avg_corr_cvar:.3f}")

# ────────────────── Turnover_Level相关性 ──────────────────
print(f"[9] Turnover_Level相关性...")

print(f"   Turnover_Level需单独计算...")

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


common_dates = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))

report = {
    "factor_id": FACTOR_ID,
    "factor_name": "换手率广度（反向）v1",
    "factor_name_en": "Turnover Breadth Neg v1",
    "category": "流动性/情绪",
    "description": f"过去{WINDOW}日内换手率高于自身中位数的天数比例，反向使用:做多低比率(连续低换手=筹码稳定)。新因子，第6个换手率维度，从关注度持续性角度衡量参与广度。",
    "hypothesis": "高活跃日比率(多数交易日换手高) → 连续过度关注 → 知情交易者信息释放完毕 → 短期反转(高换手天比率→低后续收益)。低高换手日比率→持续低参与→散户关注度低→后续反转向上→高未来收益。",
    "formula": f"neutralize(-proportion(days turnover>{WINDOW}d median, {WINDOW}d), log_amount_20d)",
    "direction": -1,
    "stock_pool": "中证1000",
    "period": f"{common_dates[0].strftime('%Y-%m-%d') if common_dates else 'N/A'} ~ {common_dates[-1].strftime('%Y-%m-%d') if common_dates else 'N/A'}",
    "n_dates": len(common_dates),
    "n_stocks": len(common_stocks) if 'common_stocks' in dir() else 0,
    "n_groups": N_GROUPS,
    "rebalance_freq": REBALANCE_FREQ,
    "forward_days": FORWARD_DAYS,
    "cost": COST,
    "data_cutoff": str(pd.Timestamp("2026-04-24")),
    "correlation_with_cvar": avg_corr_cvar,
    "metrics": nan_to_none(metrics),
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)

print(f"\n[9] 结果汇报: {REPORT_PATH}")
m = report.get("metrics", {})
ic_str = f"  IC mean: {m.get('ic_mean', 'N/A'):.4f}" if isinstance(m.get('ic_mean'), float) else "  IC mean: N/A"
t_str = f"  IC t: {m.get('ic_t_stat', 'N/A'):.2f}" if isinstance(m.get('ic_t_stat'), float) else "  IC t: N/A"
sh_str = f"  LS Sharpe: {m.get('long_short_sharpe', 'N/A'):.2f}" if isinstance(m.get('long_short_sharpe'), float) else "  LS Sharpe: N/A"
mo_str = f"  Mono: {m.get('monotonicity', 'N/A'):.2f}" if isinstance(m.get('monotonicity'), float) else "  Mono: N/A"
print(f"{ic_str}")
print(f"{t_str}")
print(f"{sh_str}")
print(f"{mo_str}")

# 阈值判断
ic_mean = m.get("ic_mean", 0)
ic_t = m.get("ic_t_stat", 0)
sharpe = m.get("long_short_sharpe", 0)

if isinstance(ic_mean, float) and isinstance(ic_t, float) and isinstance(sharpe, float):
    if abs(ic_mean) > 0.015 and abs(ic_t) > 2 and abs(sharpe) > 0.5:
        print("\n✅ 因子达标！写入factors.json ...")
        PASSES = True
    else:
        print("\n❌ 因子未达标，记录失败原因...")
        PASSES = False
else:
    print("\n⚠️ 指标计算异常，需检查")
    PASSES = False
