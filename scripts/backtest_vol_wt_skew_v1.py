#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: vol_wt_skew_v1 — 成交量加权收益偏度
构造:
  1. 计算过去20日的成交量加权三阶矩(偏度)
     vol_wt_skew = Σ(w_i * (r_i - r̄_w)^3) / (vol_wt_std^3)
     其中 w_i = volume_i / Σvolume, r̄_w = Σ(w_i * r_i)
  2. 反向使用: 做多负偏度(高量日倾向下跌,但后续均值回复)
     做空正偏度(高量日倾向上涨,但已反映信息)
  3. 市值中性化(OLS)
  4. 5%缩尾

逻辑: 
- 正偏度(放量上涨) = 好消息已在大成交量中被price in,后续动力不足
- 负偏度(放量下跌) = 恐慌性抛售制造机会,后续均值回复
- 与普通收益偏度不同: 用成交量加权可以区分"大资金方向"vs"散户噪音"
- 类似 "smart money" 信号: 大量流入日的收益方向暗示知情交易者的判断
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
FACTOR_ID = "vol_wt_skew_v1"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "csi1000_kline_raw.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
volume_piv = df.pivot_table(index="date", columns="stock_code", values="volume")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")

ret_piv = close_piv.pct_change()
log_mktcap = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造 vol_wt_skew 因子 (window={WINDOW})...")


def calc_vol_weighted_skewness(returns, volumes):
    """
    成交量加权偏度。
    w_i = vol_i / sum(vol)
    r_bar_w = sum(w_i * r_i)
    std_w = sqrt(sum(w_i * (r_i - r_bar_w)^2))
    skew_w = sum(w_i * ((r_i - r_bar_w)/std_w)^3)
    """
    valid = ~(np.isnan(returns) | np.isnan(volumes))
    r = returns[valid]
    v = volumes[valid]
    
    if len(r) < 10 or v.sum() <= 0:
        return np.nan
    
    w = v / v.sum()
    r_bar = np.sum(w * r)
    var_w = np.sum(w * (r - r_bar) ** 2)
    
    if var_w < 1e-16:
        return np.nan
    
    std_w = np.sqrt(var_w)
    skew_w = np.sum(w * ((r - r_bar) / std_w) ** 3)
    
    return skew_w


factor_matrix = pd.DataFrame(index=dates, columns=stocks, dtype=float)

for stock in stocks:
    ret_s = ret_piv[stock].values
    vol_s = volume_piv[stock].values
    for i in range(WINDOW, len(dates)):
        window_ret = ret_s[i - WINDOW:i]
        window_vol = vol_s[i - WINDOW:i]
        val = calc_vol_weighted_skewness(window_ret, window_vol)
        if not np.isnan(val):
            factor_matrix.iloc[i][stock] = val

factor_matrix = factor_matrix.astype(float)
print(f"   因子非空率: {factor_matrix.notna().mean().mean():.2%}")

# 反向使用 (做多负偏度)
print(f"[2b] 反向化 (做多负偏度 → 乘以-1)...")
factor_matrix = -factor_matrix

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

# ────────────────── 回测 ──────────────────
print(f"[5] 分层回测: {N_GROUPS}组, 频率{REBALANCE_FREQ}天, 成本{COST*100:.1f}%...")

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

print(f"[6] 计算 IC (forward={FORWARD_DAYS}d)...")
ic_series = compute_ic_dynamic(factor_aligned, return_aligned, FORWARD_DAYS, "pearson")
rank_ic_series = compute_ic_dynamic(factor_aligned, return_aligned, FORWARD_DAYS, "spearman")

group_returns, turnovers, holdings_info = compute_group_returns(
    factor_aligned, return_aligned, N_GROUPS, REBALANCE_FREQ, COST
)

metrics = compute_metrics(
    group_returns, ic_series, rank_ic_series, turnovers, N_GROUPS,
    holdings_info=holdings_info
)

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
    "factor_name": "成交量加权收益偏度(反向) v1",
    "factor_name_en": "Volume-Weighted Return Skewness (Neg) v1",
    "category": "量价/微观结构",
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
print(f"  Vol-Weighted Return Skewness (Neg) v1 回测结果")
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

ic_mean = abs(metrics.get("ic_mean", 0) or 0)
ic_t = abs(metrics.get("ic_t_stat", 0) or 0)
ls_sharpe = abs(ls_sh or 0)
is_valid = ic_mean > 0.015 and ic_t > 2 and ls_sharpe > 0.5
print(f"\n  ➤ 因子{'有效 ✓' if is_valid else '无效 ✗'} (|IC|>{0.015}: {'✓' if ic_mean>0.015 else '✗'}, t>{2}: {'✓' if ic_t>2 else '✗'}, Sharpe>{0.5}: {'✓' if ls_sharpe>0.5 else '✗'})")
