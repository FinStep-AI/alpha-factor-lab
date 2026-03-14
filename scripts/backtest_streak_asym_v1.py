#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: streak_asym_v1 — 连涨连跌不对称性
构造:
  1. 计算过去20日每天的涨跌方向(+1/-1/0)
  2. 统计连涨天数序列和连跌天数序列
  3. 因子 = mean(连涨streak长度) - mean(连跌streak长度)
     正值 = 涨的时候持续时间长、跌的时候快速结束 → 买方力量持久
  4. 市值中性化(OLS回归取残差)
  5. 5%缩尾
逻辑: 行为金融学中的"处置效应"——散户倾向于快速止损但让盈利奔跑不够；
      如果一只股票连涨天数>连跌天数,说明卖方力量弱(持筹稳定),趋势延续概率高。
      在中证1000小盘股中,散户占比高,处置效应更显著。
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
FORWARD_DAYS = 20
REBALANCE_FREQ = 20
N_GROUPS = 5
COST = 0.002
WINSORIZE_PCT = 0.05
FACTOR_ID = "streak_asym_v1"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "csi1000_kline_raw.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

# pivot
close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")

# 日收益率
ret_piv = close_piv.pct_change()

# 对数市值 (用 close * volume 近似, 因为没有流通股本数据; 用 amount 近似)
log_mktcap = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造 streak_asym 因子 (window={WINDOW})...")


def calc_streak_asymmetry(returns_series):
    """
    计算连涨/连跌天数不对称性。
    输入: 长度为 WINDOW 的收益率序列
    输出: mean(上涨streak长度) - mean(下跌streak长度)
    """
    signs = np.sign(returns_series)
    
    up_streaks = []
    down_streaks = []
    current_streak = 0
    current_sign = 0
    
    for s in signs:
        if s == current_sign and s != 0:
            current_streak += 1
        else:
            if current_sign > 0 and current_streak > 0:
                up_streaks.append(current_streak)
            elif current_sign < 0 and current_streak > 0:
                down_streaks.append(current_streak)
            current_streak = 1 if s != 0 else 0
            current_sign = s
    
    # 最后一个streak
    if current_sign > 0 and current_streak > 0:
        up_streaks.append(current_streak)
    elif current_sign < 0 and current_streak > 0:
        down_streaks.append(current_streak)
    
    mean_up = np.mean(up_streaks) if up_streaks else 0
    mean_down = np.mean(down_streaks) if down_streaks else 0
    
    return mean_up - mean_down


factor_matrix = pd.DataFrame(index=dates, columns=stocks, dtype=float)

for stock in stocks:
    ret_s = ret_piv[stock].values
    for i in range(WINDOW, len(dates)):
        window_ret = ret_s[i - WINDOW:i]
        if np.sum(np.isnan(window_ret)) > WINDOW * 0.3:
            continue
        valid_ret = window_ret[~np.isnan(window_ret)]
        if len(valid_ret) < 10:
            continue
        factor_matrix.iloc[i][stock] = calc_streak_asymmetry(valid_ret)

factor_matrix = factor_matrix.astype(float)
print(f"   因子非空率: {factor_matrix.notna().mean().mean():.2%}")

# ────────────────── 缩尾 ──────────────────
print(f"[3] 缩尾 ({WINSORIZE_PCT*100:.0f}%)...")
for date in dates:
    row = factor_matrix.loc[date].dropna()
    if len(row) < 10:
        continue
    lo = row.quantile(WINSORIZE_PCT)
    hi = row.quantile(1 - WINSORIZE_PCT)
    factor_matrix.loc[date] = factor_matrix.loc[date].clip(lo, hi)

# ────────────────── 市值中性化 ──────────────────
print(f"[4] 市值中性化 (OLS)...")
factor_neutral = factor_matrix.copy()
for date in dates:
    f = factor_matrix.loc[date].dropna()
    m = log_mktcap.loc[date].reindex(f.index).dropna()
    common = f.index.intersection(m.index)
    if len(common) < 30:
        continue
    f_c = f[common].values
    m_c = m[common].values
    X = np.column_stack([np.ones(len(m_c)), m_c])
    try:
        beta = np.linalg.lstsq(X, f_c, rcond=None)[0]
        residual = f_c - X @ beta
        factor_neutral.loc[date, common] = residual
    except:
        pass

print(f"   中性化后非空率: {factor_neutral.notna().mean().mean():.2%}")

# ────────────────── 回测引擎 ──────────────────
print(f"[5] 分层回测: {N_GROUPS}组, 频率{REBALANCE_FREQ}天, 成本{COST*100:.1f}%...")

# 准备因子和收益矩阵
common_dates = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
common_stocks = sorted(factor_neutral.columns.intersection(ret_piv.columns))
factor_aligned = factor_neutral.loc[common_dates, common_stocks]
return_aligned = ret_piv.loc[common_dates, common_stocks]

# 使用回测引擎
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills" / "alpha-factor-lab" / "scripts"))
from factor_backtest import (
    compute_group_returns,
    compute_ic_dynamic,
    compute_metrics,
    save_backtest_data,
    newey_west_t_stat,
)

# IC
print(f"[6] 计算 IC (forward={FORWARD_DAYS}d)...")
ic_series = compute_ic_dynamic(factor_aligned, return_aligned, FORWARD_DAYS, "pearson")
rank_ic_series = compute_ic_dynamic(factor_aligned, return_aligned, FORWARD_DAYS, "spearman")

# 分层
group_returns, turnovers, holdings_info = compute_group_returns(
    factor_aligned, return_aligned, N_GROUPS, REBALANCE_FREQ, COST
)

# 指标
metrics = compute_metrics(
    group_returns, ic_series, rank_ic_series, turnovers, N_GROUPS,
    holdings_info=holdings_info
)

# ────────────────── 输出 ──────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 保存中间数据
save_backtest_data(group_returns, ic_series, rank_ic_series, str(OUTPUT_DIR))

# 报告
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
    "factor_name": "连涨连跌不对称性 v1",
    "factor_name_en": "Streak Asymmetry v1",
    "category": "行为/动量",
    "period": f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}",
    "n_dates": len(common_dates),
    "n_stocks": len(common_stocks),
    "n_groups": N_GROUPS,
    "rebalance_freq": REBALANCE_FREQ,
    "forward_days": FORWARD_DAYS,
    "cost": COST,
    "metrics": metrics,
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(nan_to_none(report), f, indent=2, ensure_ascii=False)

# ────────────────── 打印摘要 ──────────────────
print(f"\n{'='*60}")
print(f"  Streak Asymmetry v1 回测结果")
print(f"{'='*60}")
print(f"  区间:     {report['period']}")
print(f"  股票:     {len(common_stocks)}")
ic_sig = "✓" if metrics.get("ic_significant_5pct") else "✗"
print(f"  IC均值:   {metrics.get('ic_mean', 0):.4f}  (t={metrics.get('ic_t_stat', 0):.2f}, {ic_sig})")
print(f"  Rank IC:  {metrics.get('rank_ic_mean', 0):.4f}")
print(f"  IR:       {metrics.get('ir', 0):.4f}")
ls_sh = metrics.get("long_short_sharpe", 0)
ls_md = metrics.get("long_short_mdd", 0)
print(f"  多空Sharpe: {ls_sh:.4f}")
print(f"  多空MDD:    {ls_md:.2%}")
print(f"  单调性:     {metrics.get('monotonicity', 0):.4f}")
print(f"  换手率:     {metrics.get('turnover_mean', 0):.2%}")
print(f"{'─'*60}")
print(f"  分层年化收益:")
for i, r in enumerate(metrics.get("group_returns_annualized", []), 1):
    r_str = f"{r:.2%}" if r is not None else "N/A"
    print(f"    G{i}: {r_str}")
print(f"{'='*60}")

# 判断是否有效
ic_mean = abs(metrics.get("ic_mean", 0) or 0)
ic_t = abs(metrics.get("ic_t_stat", 0) or 0)
ls_sharpe = abs(ls_sh or 0)
is_valid = ic_mean > 0.015 and ic_t > 2 and ls_sharpe > 0.5
print(f"\n  ➤ 因子{'有效 ✓' if is_valid else '无效 ✗'} (|IC|>{0.015}: {'✓' if ic_mean>0.015 else '✗'}, t>{2}: {'✓' if ic_t>2 else '✗'}, Sharpe>{0.5}: {'✓' if ls_sharpe>0.5 else '✗'})")
