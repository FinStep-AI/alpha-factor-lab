#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: amt_trend_v1 — 成交额趋势(斜率)
构造:
  1. 对每只股票的 log(daily_amount) 做20日滚动线性回归 (y = a + b*t)
  2. 因子值 = 斜率 b (标准化后)
  3. 反向使用: 做多成交额萎缩(b<0)的股票
     逻辑: 成交额持续增长=散户涌入/热点关注→后续反转
           成交额持续萎缩=关注度降低→卖压耗尽→未来反弹
  4. 同时测试正向(做多b>0),看哪个方向有效
  5. 市值中性化(OLS) + 5%缩尾

学术背景:
- Chordia & Subrahmanyam (2004) "Order Imbalance, Liquidity, and Market Returns"
- 成交额趋势 vs 换手率衰减的区别: 换手率衰减看短/长比值(5d/20d),本因子看斜率(单调趋势方向)
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
FACTOR_ID = "amt_trend_v1"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "csi1000_kline_raw.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")

ret_piv = close_piv.pct_change()
log_amount = np.log(amount_piv.clip(lower=1))
log_mktcap = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ────────────────── 因子构造 (向量化) ──────────────────
print(f"[2] 构造 amt_trend 因子 (window={WINDOW})...")

# 向量化: 对每个日期窗口内的 log_amount 做线性回归取斜率
# slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
# x = [0, 1, ..., WINDOW-1]

x = np.arange(WINDOW, dtype=float)
n = WINDOW
sum_x = x.sum()
sum_x2 = (x ** 2).sum()
denom = n * sum_x2 - sum_x ** 2

factor_matrix = pd.DataFrame(index=dates, columns=stocks, dtype=float)

log_amt_vals = log_amount.values  # (T, S)

for i in range(WINDOW, len(dates)):
    window_data = log_amt_vals[i - WINDOW:i, :]  # (WINDOW, S)
    valid_count = np.sum(~np.isnan(window_data), axis=0)
    
    # 至少需要15个有效观测
    mask = valid_count >= 15
    
    # 简化：用全窗口数据（NaN用列均值填充）
    col_means = np.nanmean(window_data, axis=0)
    filled = np.where(np.isnan(window_data), col_means[np.newaxis, :], window_data)
    
    sum_y = filled.sum(axis=0)
    sum_xy = (x[:, np.newaxis] * filled).sum(axis=0)
    
    slopes = (n * sum_xy - sum_x * sum_y) / denom
    slopes[~mask] = np.nan
    factor_matrix.iloc[i] = slopes

factor_matrix = factor_matrix.astype(float)
print(f"   因子非空率: {factor_matrix.notna().mean().mean():.2%}")

# 先测试两个方向
# 反向: 做多成交额萎缩
factor_neg = -factor_matrix.copy()
# 正向: 做多成交额增长
factor_pos = factor_matrix.copy()

# ────────────────── 处理函数 ──────────────────
def process_and_backtest(factor_raw, direction_name, factor_id_suffix):
    """缩尾 + 中性化 + 回测"""
    factor = factor_raw.copy()
    
    # 缩尾
    for date in dates:
        row = factor.loc[date].dropna()
        if len(row) < 10:
            continue
        lo = row.quantile(WINSORIZE_PCT)
        hi = row.quantile(1 - WINSORIZE_PCT)
        factor.loc[date] = factor.loc[date].clip(lo, hi)
    
    # 中性化
    factor_neutral = factor.copy()
    for date in dates:
        f = factor.loc[date].dropna()
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
    
    # 回测
    common_dates = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
    common_stocks = sorted(factor_neutral.columns.intersection(ret_piv.columns))
    factor_aligned = factor_neutral.loc[common_dates, common_stocks]
    return_aligned = ret_piv.loc[common_dates, common_stocks]
    
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills" / "alpha-factor-lab" / "scripts"))
    from factor_backtest import (
        compute_group_returns,
        compute_ic_dynamic,
        compute_metrics,
        save_backtest_data,
    )
    
    ic_series = compute_ic_dynamic(factor_aligned, return_aligned, FORWARD_DAYS, "pearson")
    rank_ic_series = compute_ic_dynamic(factor_aligned, return_aligned, FORWARD_DAYS, "spearman")
    
    group_returns, turnovers, holdings_info = compute_group_returns(
        factor_aligned, return_aligned, N_GROUPS, REBALANCE_FREQ, COST
    )
    
    metrics = compute_metrics(
        group_returns, ic_series, rank_ic_series, turnovers, N_GROUPS,
        holdings_info=holdings_info
    )
    
    # 输出
    out_dir = Path(__file__).resolve().parent.parent / "output" / f"{FACTOR_ID}_{factor_id_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_backtest_data(group_returns, ic_series, rank_ic_series, str(out_dir))
    
    # 打印
    ic_sig = "✓" if metrics.get("ic_significant_5pct") else "✗"
    ls_sh = metrics.get("long_short_sharpe", 0)
    ls_md = metrics.get("long_short_mdd", 0)
    print(f"\n  --- {direction_name} ---")
    print(f"  IC均值:   {metrics.get('ic_mean', 0):.4f}  (t={metrics.get('ic_t_stat', 0):.2f}, {ic_sig})")
    print(f"  Rank IC:  {metrics.get('rank_ic_mean', 0):.4f}")
    print(f"  多空Sharpe: {ls_sh:.4f}")
    print(f"  多空MDD:    {ls_md:.2%}")
    print(f"  单调性:     {metrics.get('monotonicity', 0):.4f}")
    grp_rets = metrics.get("group_returns_annualized", [])
    for i, r in enumerate(grp_rets, 1):
        r_str = f"{r:.2%}" if r is not None else "N/A"
        print(f"    G{i}: {r_str}")
    
    ic_mean = abs(metrics.get("ic_mean", 0) or 0)
    ic_t = abs(metrics.get("ic_t_stat", 0) or 0)
    ls_sharpe = abs(ls_sh or 0)
    is_valid = ic_mean > 0.015 and ic_t > 2 and ls_sharpe > 0.5
    print(f"  ➤ {'有效 ✓' if is_valid else '无效 ✗'}")
    
    return metrics, common_dates, common_stocks, group_returns, ic_series, rank_ic_series, is_valid

print(f"[3-6] 回测两个方向...")
metrics_neg, cd_n, cs_n, gr_n, ic_n, ric_n, valid_neg = process_and_backtest(factor_neg, "反向(做多缩量)", "neg")
metrics_pos, cd_p, cs_p, gr_p, ic_p, ric_p, valid_pos = process_and_backtest(factor_pos, "正向(做多放量)", "pos")

# 选择更好的方向保存最终报告
if valid_neg or valid_pos:
    # 选IC更好的
    if abs(metrics_neg.get("ic_mean",0) or 0) >= abs(metrics_pos.get("ic_mean",0) or 0):
        best = "neg"
        best_metrics = metrics_neg
        best_gr, best_ic, best_ric = gr_n, ic_n, ric_n
        best_cd, best_cs = cd_n, cs_n
        best_dir = -1
    else:
        best = "pos"
        best_metrics = metrics_pos
        best_gr, best_ic, best_ric = gr_p, ic_p, ric_p
        best_cd, best_cs = cd_p, cs_p
        best_dir = 1
    
    # 保存到主输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
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
    
    save_backtest_data(best_gr, best_ic, best_ric, str(OUTPUT_DIR))
    
    report = {
        "factor_id": FACTOR_ID,
        "factor_name": f"成交额趋势({'反向' if best_dir==-1 else '正向'}) v1",
        "direction": best_dir,
        "best_direction": best,
        "period": f"{best_cd[0].strftime('%Y-%m-%d')} ~ {best_cd[-1].strftime('%Y-%m-%d')}",
        "n_dates": len(best_cd),
        "n_stocks": len(best_cs),
        "n_groups": N_GROUPS,
        "rebalance_freq": REBALANCE_FREQ,
        "forward_days": FORWARD_DAYS,
        "cost": COST,
        "metrics": best_metrics,
    }
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(nan_to_none(report), f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 最佳方向: {best}, 报告已保存: {REPORT_PATH}")
else:
    print(f"\n❌ 两个方向均未达标")
