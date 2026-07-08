#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: close_bias_trend_rev_v1 — 收盘偏向趋势反转 v1
============================================================
构造:
  1. daily_bias = 2 * (close - (high+low)/2) / (high-low)
     映射到 [-1, 1], 收在最高=1, 收在最低=-1
  2. 对每只股票近20日 daily_bias 做OLS回归, 取斜率
     slope > 0 表示收盘位置持续改善; slope < 0 表示收盘位置持续恶化
  3. 取反后使用: factor_raw = - slope
     经济含义: 尾盘持续走弱/收盘位置连续下移的股票, 短期存在均值回复
  4. 截面按 log(MA20(amount)) 做OLS中性化 + 5%分位缩尾 + z-score

回测配置:
  20日窗口 | 市值代理=log_amount_20d | 5%缩尾 | 5组分层
  forward_days=5 | rebalance=5 | cost=0.003

有效性门槛:
  |IC| > 0.015, t > 2, long_short_sharpe > 0.5
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WINDOW = 20
FORWARD_DAYS = 5
REBALANCE_FREQ = 5
N_GROUPS = 5
COST = 0.003
WINSORIZE_PCT = 0.05
FACTOR_ID = "close_bias_trend_rev_v1"
FACTOR_NAME_CN = "收盘偏向趋势反转 v1"
FACTOR_NAME_EN = "Close Bias Trend Reversal v1"
DIRECTION = 1
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "data" / "csi1000_kline_raw.csv"
FACTOR_CSV = REPO_ROOT / "data" / "factor_close_bias_trend_v1.csv"
OUTPUT_DIR = REPO_ROOT / "output" / FACTOR_ID


def compute_ic_t(ic_series: pd.Series) -> float:
    v = ic_series.dropna().values
    if len(v) < 10:
        return 0.0
    s = float(np.std(v, ddof=1))
    if s < 1e-12:
        return 0.0
    return float(np.mean(v) / (s / np.sqrt(len(v))))


print(f"[1] 加载数据: {DATA_PATH.name}")
raw = pd.read_csv(DATA_PATH, usecols=["date", "stock_code", "close"])
raw["date"] = pd.to_datetime(raw["date"])
raw["stock_code"] = raw["stock_code"].astype(str).str.zfill(6)
raw = raw.sort_values(["date", "stock_code"])

print(f"[2] 加载因子: {FACTOR_CSV.name}")
fac = pd.read_csv(FACTOR_CSV)
fac["date"] = pd.to_datetime(fac["date"])
fac["stock_code"] = fac["stock_code"].astype(str).str.zfill(6)

close = raw.pivot_table(index="date", columns="stock_code", values="close")
ret = close.pct_change(fill_method=None)
# 反向使用: 尾盘持续走弱 -> 后续反转
factor = -fac.pivot_table(index="date", columns="stock_code", values="factor")

common_dates = sorted(factor.dropna(how="all").index.intersection(ret.dropna(how="all").index))
common_stocks = sorted(factor.columns.intersection(ret.columns))
F = factor.loc[common_dates, common_stocks]
R = ret.loc[common_dates, common_stocks]

print(f"    矩阵: {len(common_dates)}日 × {len(common_stocks)}股")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from factor_backtest import compute_group_returns, compute_ic_dynamic, compute_metrics, save_backtest_data

print(f"[3] 回测: fwd={FORWARD_DAYS}d rb={REBALANCE_FREQ}d cost={COST}")
ic_series = compute_ic_dynamic(F, R, FORWARD_DAYS, "pearson")
rank_ic_series = compute_ic_dynamic(F, R, FORWARD_DAYS, "spearman")
group_returns, turnover, holdings = compute_group_returns(F, R, N_GROUPS, REBALANCE_FREQ, COST)
metrics = compute_metrics(group_returns, ic_series, rank_ic_series, turnover, N_GROUPS, holdings_info=holdings)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_backtest_data(group_returns, ic_series, rank_ic_series, str(OUTPUT_DIR))

report = {
    "factor_id": FACTOR_ID,
    "factor_name": FACTOR_NAME_CN,
    "factor_name_en": FACTOR_NAME_EN,
    "description": "收盘位置(在当日高低点区间中的位置)的20日OLS斜率取反，捕捉尾盘持续走弱后的短期反转。高因子值=收盘位置连续下移=尾盘卖压累积，后续5日更易均值回复。",
    "formula": "neutralize(-OLS_slope_20d(2*(close-mid)/(high-low)), log_amount_20d), 5% winsorize, z-score",
    "direction": DIRECTION,
    "window": WINDOW,
    "forward_days": FORWARD_DAYS,
    "rebalance_freq": REBALANCE_FREQ,
    "cost": COST,
    "period": f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}",
    "metrics": metrics,
}
with open(OUTPUT_DIR / "backtest_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=float)

ic_mean = float(metrics.get("ic_mean", 0) or 0)
ic_t = float(metrics.get("ic_t_stat", compute_ic_t(ic_series)) or 0)
ls_sh = float(metrics.get("long_short_sharpe", 0) or 0)
mono = float(metrics.get("monotonicity", 0) or 0)
valid = abs(ic_mean) > 0.015 and ic_t > 2 and ls_sh > 0.5

print("\n=== SUMMARY ===")
print(f"IC={ic_mean:.4f}  t={ic_t:.2f}  Sharpe={ls_sh:.3f}  mono={mono:.2f}")
print(f"Valid: {'YES' if valid else 'NO'}")
print(f"Output: {OUTPUT_DIR}")

sys.exit(0 if valid else 1)
