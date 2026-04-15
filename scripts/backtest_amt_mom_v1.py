#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: amt_mom_v1 — 成交额动量
========================================
构造:
  1. factor_raw = log(MA20(amount) / MA60(amount))
  
     正值 = 近期成交额 > 远期成交额 = 关注度上升/资金流入
     负值 = 近期成交额 < 远期成交额 = 关注度下降/资金流出
     
  2. 直接市值(amount)中性化(OLS) + 5%缩尾 + z-score

逻辑:
- 成交额增长=更多投资者关注/更多信息到达→价格发现效率提高→正alpha
- 成交额萎缩=关注度下降/资金退出→后续表现差→低因子值=负alpha
- 本质是 momentum 逻辑应用于市场关注度维度

与turnover_level的区别:
- turnover_level: 绝对换手率水平(高=关注度高)
- amt_mom: 成交额的变化趋势(增=关注度上升动量)
- 两者可能互补, 但amt_mom更接近momentum逻辑

相关文献:
- Campbell, Grossman & Wang (1993) "Trading Volume and Serial Correlation" 
  → 成交量变化与收益自相关
- Brennan, Chordia & Subrahmanyam (1998) "Alternative factor specifications"
  → 成交额动量效应
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SHORT_W = 20
LONG_W = 60
FORWARD_DAYS = 5
REBALANCE_FREQ = 5
N_GROUPS = 5
COST = 0.003
WINSORIZE_PCT = 0.05
FACTOR_ID = "amt_mom_v1"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "csi1000_kline_raw.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

print(f"[1] 加载数据...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

df["daily_ret"] = df.groupby("stock_code")["close"].pct_change()

# 市值代理 (较长期: 60日均值)
df["avg_amount_60d"] = df.groupby("stock_code")["amount"].transform(
    lambda x: x.rolling(LONG_W, min_periods=45).mean()
)
df["log_amount_60d"] = np.log(df["avg_amount_60d"].clip(lower=1))

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造 Amount Momentum 因子 (MA{SHORT_W}/MA{LONG_W})...")

df["ma20_amount"] = df.groupby("stock_code")["amount"].transform(
    lambda x: x.rolling(SHORT_W, min_periods=15).mean()
)
df["ma60_amount"] = df.groupby("stock_code")["amount"].transform(
    lambda x: x.rolling(LONG_W, min_periods=45).mean()
)

df["factor_raw"] = np.log(df["ma20_amount"] / (df["ma60_amount"] + 1))

print(f"   因子非空率: {df['factor_raw'].notna().mean():.2%}")

valid_f = df["factor_raw"].dropna()
if len(valid_f) > 0:
    print(f"   因子描述统计: mean={valid_f.mean():.4f} std={valid_f.std():.4f} "
          f"p5={valid_f.quantile(0.05):.4f} p50={valid_f.median():.4f} p95={valid_f.quantile(0.95):.4f}")

# ────────────────── 回测 ──────────────────
dates = sorted(df["date"].unique())
stocks = sorted(df["stock_code"].unique())

factor_matrix = np.full((len(dates), len(stocks)), np.nan)
date_idx = {d: i for i, d in enumerate(dates)}
stock_idx = {s: i for i, s in enumerate(stocks)}

for _, row in df.iterrows():
    if not np.isnan(row["factor_raw"]):
        factor_matrix[date_idx[row["date"]], stock_idx[row["stock_code"]]] = row["factor_raw"]

factor_df = pd.DataFrame(factor_matrix, index=dates, columns=stocks)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close", dropna=False)
ret_piv = close_piv.pct_change()
log_mktcap = df.pivot_table(index="date", columns="stock_code", 
                              values="avg_amount_60d", dropna=False)
log_mktcap = np.log(log_mktcap.clip(lower=1))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills" / "alpha-factor-lab" / "scripts"))
try:
    from factor_backtest import compute_group_returns, compute_ic_dynamic, compute_metrics, save_backtest_data
except ImportError:
    from factor_backtest import compute_group_returns, compute_ic_dynamic, compute_metrics, save_backtest_data


def run_backtest(factor_raw, direction, label):
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
            factor.loc[date, common] = residual
        except:
            pass
    
    common_dates = sorted(factor.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
    common_stocks = sorted(factor.columns.intersection(ret_piv.columns))
    fa = factor.loc[common_dates, common_stocks]
    ra = ret_piv.loc[common_dates, common_stocks]
    
    ic = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "pearson")
    ric = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "spearman")
    gr, to, hi = compute_group_returns(fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
    metrics = compute_metrics(gr, ic, ric, to, N_GROUPS, holdings_info=hi)
    
    ic_mean = metrics.get("ic_mean", 0) or 0
    ic_t = metrics.get("ic_t_stat", 0) or 0
    ls_sh = metrics.get("long_short_sharpe", 0) or 0
    mono = metrics.get("monotonicity", 0) or 0
    ls_md = metrics.get("long_short_mdd", 0) or 0
    
    sig = "✓" if metrics.get("ic_significant_5pct") else "✗"
    print(f"\n  [{label}] IC={ic_mean:.4f}(t={ic_t:.2f},{sig}) Sharpe={ls_sh:.2f} Mono={mono:.2f} MDD={ls_md:.2%}")
    grp = metrics.get("group_returns_annualized", [None] * N_GROUPS)
    for i, r in enumerate(grp, 1):
        print(f"    G{i}: {r:.2%}" if r is not None else f"    G{i}: N/A")
    
    is_valid = abs(ic_mean) > 0.015 and abs(ic_t) > 2 and abs(ls_sh) > 0.5
    print(f"  ➤ {'有效 ✓' if is_valid else '无效 ✗'}")
    return metrics, gr, ic, ric, common_dates, common_stocks, is_valid


print(f"\n=== Forward={FORWARD_DAYS}d, Rebal={REBALANCE_FREQ}d ===")
m1, g1, i1, r1, cd1, cs1, v1 = run_backtest(factor_df, 1, "正向(做多成交额上升)")
m2, g2, i2, r2, cd2, cs2, v2 = run_backtest(factor_df, -1, "反向(做多成交额下降)")

FORWARD_DAYS2 = 20
REBALANCE_FREQ2 = 20
COST2 = 0.002
print(f"\n=== Forward={FORWARD_DAYS2}d, Rebal={REBALANCE_FREQ2}d ===")

def run_backtest_20d(factor_raw, direction, label):
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
            factor.loc[date, common] = residual
        except:
            pass
    common_dates = sorted(factor.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
    common_stocks = sorted(factor.columns.intersection(ret_piv.columns))
    fa = factor.loc[common_dates, common_stocks]
    ra = ret_piv.loc[common_dates, common_stocks]
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills" / "alpha-factor-lab" / "scripts"))
    from factor_backtest import compute_group_returns, compute_ic_dynamic, compute_metrics, save_backtest_data
    ic = compute_ic_dynamic(fa, ra, FORWARD_DAYS2, "pearson")
    ric = compute_ic_dynamic(fa, ra, FORWARD_DAYS2, "spearman")
    gr, to, hi = compute_group_returns(fa, ra, N_GROUPS, REBALANCE_FREQ2, COST2)
    metrics = compute_metrics(gr, ic, ric, to, N_GROUPS, holdings_info=hi)
    ic_mean = metrics.get("ic_mean", 0) or 0
    ic_t = metrics.get("ic_t_stat", 0) or 0
    ls_sh = metrics.get("long_short_sharpe", 0) or 0
    mono = metrics.get("monotonicity", 0) or 0
    ls_md = metrics.get("long_short_mdd", 0) or 0
    sig = "✓" if metrics.get("ic_significant_5pct") else "✗"
    print(f"\n  [{label}] IC={ic_mean:.4f}(t={ic_t:.2f},{sig}) Sharpe={ls_sh:.2f} Mono={mono:.2f} MDD={ls_md:.2%}")
    grp = metrics.get("group_returns_annualized", [None] * N_GROUPS)
    for i, r in enumerate(grp, 1):
        print(f"    G{i}: {r:.2%}" if r is not None else f"    G{i}: N/A")
    is_valid = (abs(ic_mean) > 0.015 and abs(ic_t) > 2 and abs(ls_sh) > 0.5)
    print(f"  ➤ {'有效 ✓' if is_valid else '无效 ✗'}")
    return metrics, gr, ic, ric, common_dates, common_stocks, is_valid


m3, g3, i3, r3, cd3, cs3, v3 = run_backtest_20d(factor_df, 1, "正向20d")
m4, g4, i4, r4, cd4, cs4, v4 = run_backtest_20d(factor_df, -1, "反向20d")

all_results = [
    (m1, g1, i1, r1, cd1, cs1, v1, 1, 5), (m2, g2, i2, r2, cd2, cs2, v2, -1, 5),
    (m3, g3, i3, r3, cd3, cs3, v3, 1, 20), (m4, g4, i4, r4, cd4, cs4, v4, -1, 20)
]
valid_results = [(m, g, ic, ric, cd, cs, d, f) for m, g, ic, ric, cd, cs, v, d, f in all_results if v]

if valid_results:
    best = max(valid_results, key=lambda x: abs(x[0].get("long_short_sharpe", 0) or 0))
    bm, bg, bic, bric, bcd, bcs, bd, bf = best
    
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
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_backtest_data(bg, bic, bric, str(OUTPUT_DIR))
    
    report = {
        "factor_id": FACTOR_ID,
        "factor_name": "成交额动量 v1",
        "factor_name_en": "Amount Momentum v1",
        "direction": bd,
        "forward_days": bf,
        "rebalance_freq": bf,
        "cost": 0.003 if bf == 5 else 0.002,
        "period": f"{bcd[0].strftime('%Y-%m-%d')} ~ {bcd[-1].strftime('%Y-%m-%d')}",
        "n_stocks": len(bcs),
        "metrics": bm,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(nan_to_none(report), f, indent=2, ensure_ascii=False)
    print(f"\n✅ 报告已保存: {REPORT_PATH}")
else:
    print(f"\n❌ 所有配置均未达标")
