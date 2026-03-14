#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: limit_proximity_v1 — 涨跌停接近度
构造:
  1. daily_max_pct = (high - prev_close) / prev_close  (日内最高涨幅)
  2. daily_min_pct = (low - prev_close) / prev_close   (日内最低跌幅)
  3. 涨停接近度 = mean(daily_max_pct >= 9%, 20d) (频率)
  4. 跌停接近度 = mean(daily_min_pct <= -9%, 20d) (频率)
  5. 组合因子1: -跌停接近度 (回避接近跌停的, 类似CVaR)
  6. 组合因子2: 涨停接近度 - 跌停接近度 (不对称性)
  7. 组合因子3: -(涨停+跌停接近度) (回避所有极端,低波动偏好)
  8. 市值中性化 + 5%缩尾

逻辑:
- 接近跌停 = 面临极端下行风险, 类似CVaR的逻辑
- 接近涨停 = lottery-like, 散户追涨
- 方向需要实证测试

注意: A股中证1000涨跌停板为10%(科创板/创业板20%)
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WINDOW = 10  # 用10日窗口(类似CVaR)
WINSORIZE_PCT = 0.05
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "csi1000_kline_raw.csv"
BASE_OUTPUT = Path(__file__).resolve().parent.parent / "output"

print(f"[1] 加载数据...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
high_piv = df.pivot_table(index="date", columns="stock_code", values="high")
low_piv = df.pivot_table(index="date", columns="stock_code", values="low")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")

ret_piv = close_piv.pct_change()
log_amount_20d = np.log(amount_piv.rolling(20).mean().clip(lower=1))

prev_close = close_piv.shift(1)
daily_max_pct = (high_piv - prev_close) / prev_close.clip(lower=0.01)
daily_min_pct = (low_piv - prev_close) / prev_close.clip(lower=0.01)

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# 接近涨停: 日内最高涨幅 >= 8% (给一些buffer,不一定正好10%)
near_limit_up = (daily_max_pct >= 0.08).astype(float)
# 接近跌停: 日内最低跌幅 <= -8%
near_limit_down = (daily_min_pct <= -0.08).astype(float)

# 也用更宽松的阈值
near_limit_up_5 = (daily_max_pct >= 0.05).astype(float)
near_limit_down_5 = (daily_min_pct <= -0.05).astype(float)

# 滚动窗口 (10日和20日)
print(f"[2] 构造因子...")

# 因子候选: 
# A: 跌停接近频率(反向=做多低跌停接近)
f_down_10d = near_limit_down.rolling(10, min_periods=5).mean()
f_down_20d = near_limit_down.rolling(20, min_periods=10).mean()

# B: 涨停接近频率(可能正向or反向)
f_up_10d = near_limit_up.rolling(10, min_periods=5).mean()

# C: 极端日频率(涨停+跌停,反向=做多低极端)
f_extreme_10d = (near_limit_up.astype(float) + near_limit_down.astype(float)).rolling(10, min_periods=5).mean()

# D: 用5%阈值的跌停接近频率
f_down5_10d = near_limit_down_5.rolling(10, min_periods=5).mean()
f_down5_20d = near_limit_down_5.rolling(20, min_periods=10).mean()

# E: 极端负收益天数(类CVaR的离散版)
extreme_neg_ret = (ret_piv <= ret_piv.quantile(0.05, axis=1).values[:, np.newaxis]).astype(float)
# 不对, 让我用固定阈值
extreme_neg_day = (ret_piv <= -0.03).astype(float)
f_neg3pct_10d = extreme_neg_day.rolling(10, min_periods=5).mean()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills" / "alpha-factor-lab" / "scripts"))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data
)


def pipeline(factor_raw, direction, fwd, rebal, cost, factor_id, label):
    factor = factor_raw.copy()
    if direction == -1:
        factor = -factor
    
    for date in dates:
        row = factor.loc[date].dropna()
        if len(row) < 10:
            continue
        lo = row.quantile(WINSORIZE_PCT)
        hi = row.quantile(1 - WINSORIZE_PCT)
        factor.loc[date] = factor.loc[date].clip(lo, hi)
    
    for date in dates:
        f = factor.loc[date].dropna()
        m = log_amount_20d.loc[date].reindex(f.index).dropna()
        common = f.index.intersection(m.index)
        if len(common) < 30:
            continue
        f_c = f[common].values
        m_c = m[common].values
        X = np.column_stack([np.ones(len(m_c)), m_c])
        try:
            beta = np.linalg.lstsq(X, f_c, rcond=None)[0]
            factor.loc[date, common] = f_c - X @ beta
        except:
            pass
    
    cd = sorted(factor.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
    cs = sorted(factor.columns.intersection(ret_piv.columns))
    fa = factor.loc[cd, cs]
    ra = ret_piv.loc[cd, cs]
    
    ic = compute_ic_dynamic(fa, ra, fwd, "pearson")
    ric = compute_ic_dynamic(fa, ra, fwd, "spearman")
    gr, to, hi = compute_group_returns(fa, ra, 5, rebal, cost)
    metrics = compute_metrics(gr, ic, ric, to, 5, holdings_info=hi)
    
    ic_m = metrics.get("ic_mean", 0) or 0
    ic_t = metrics.get("ic_t_stat", 0) or 0
    ls_sh = metrics.get("long_short_sharpe", 0) or 0
    mono = metrics.get("monotonicity", 0) or 0
    sig = "✓" if metrics.get("ic_significant_5pct") else "✗"
    
    is_valid = abs(ic_m) > 0.015 and abs(ic_t) > 2 and abs(ls_sh) > 0.5
    flag = "✅" if is_valid else "  "
    print(f"{flag} [{label}] IC={ic_m:.4f}(t={ic_t:.2f},{sig}) Sh={ls_sh:.2f} Mono={mono:.2f}")
    
    if is_valid:
        grp = metrics.get("group_returns_annualized", [])
        for i, r in enumerate(grp, 1):
            print(f"      G{i}: {r:.2%}" if r is not None else f"      G{i}: N/A")
        
        out = BASE_OUTPUT / factor_id
        out.mkdir(parents=True, exist_ok=True)
        save_backtest_data(gr, ic, ric, str(out))
    
    return metrics, is_valid


print(f"\n[3] 回测...")
print(f"{'='*90}")

factors_to_test = [
    (f_down_10d, "down8_10d", "跌停接近8%_10d"),
    (f_down_20d, "down8_20d", "跌停接近8%_20d"),
    (f_down5_10d, "down5_10d", "跌方5%_10d"),
    (f_down5_20d, "down5_20d", "跌方5%_20d"),
    (f_neg3pct_10d, "neg3_10d", "负3%天数_10d"),
    (f_extreme_10d, "extreme_10d", "极端日_10d"),
    (f_up_10d, "up8_10d", "涨停接近_10d"),
]

any_valid = False
for fraw, fbase, fname in factors_to_test:
    for d, dname in [(-1, "neg"), (1, "pos")]:
        for fwd, reb, cost in [(5, 5, 0.003)]:
            fid = f"limit_{fbase}_{dname}_f{fwd}"
            lab = f"{fname} {dname} f{fwd}"
            m, v = pipeline(fraw, d, fwd, reb, cost, fid, lab)
            if v:
                any_valid = True

if not any_valid:
    print(f"\n❌ 涨跌停接近度因子所有配置均未达标")
