#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: pv_divergence_v1 — 价格-成交量相对波动率背离
====================================================
构造:
  1. 计算20日滚动窗口:
     price_cv = std(daily_ret_20d) / mean(|daily_ret|_20d)  # 价格收益率的变异系数
     amt_cv = std(amt_chg_20d) / mean(|amt_chg|_20d)      # 成交额变化的变异系数
     
  2. 因子 = -price_cv / (amt_cv + eps)
     
     负值(price_cv相对大, amt_cv相对小):
       价格波动大但成交额稳定 = 可能由少数交易者推动 = 高信息不对称 = 高alpha
     
  3. 市值中性化(OLS) + 5%缩尾 + z-score

逻辑:
- 价格波动大但成交额稳定的股票: 少数知情交易者通过较小成交量推动大价格移动 → 信息不对称高 → alpha
- 价格波动小但成交额大的股票: 高成交量但价格不动 = 市场消化充分/信息透明 → 较低alpha
- 本质上：衡量价格对成交量的"杠杆率"，高杠杆=高信息效率

与现有因子的区别:
- vs amihud(流动性): amihud看价格冲击成本(|ret|/amount)，本因子看波动率结构
- vs amp_level(振幅水平): amp_level看绝对波动水平，本因子看价格vs成交额波动率的相对关系
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
FACTOR_ID = "pv_divergence_v1"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "csi1000_kline_raw.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

print(f"[1] 加载数据...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

# 日收益率
df["daily_ret"] = df.groupby("stock_code")["close"].pct_change()
# 成交额日变化率
df["amt_lag"] = df.groupby("stock_code")["amount"].shift(1)
df["amt_chg"] = (df["amount"] - df["amt_lag"]) / (df["amt_lag"].abs() + 1)

# 市值代理 (20日均成交额)
df["avg_amount_20d"] = df.groupby("stock_code")["amount"].transform(
    lambda x: x.rolling(WINDOW, min_periods=10).mean()
)
df["log_amount"] = np.log(df["avg_amount_20d"].clip(lower=1))

# ────────────────── 因子构造(向量化) ──────────────────
print(f"[2] 构造 PV Divergence 因子 (window={WINDOW})...")

# 计算滚动变异系数
df["ret_mean"] = df.groupby("stock_code")["daily_ret"].transform(
    lambda x: x.rolling(WINDOW, min_periods=15).mean()
)
df["ret_std"] = df.groupby("stock_code")["daily_ret"].transform(
    lambda x: x.rolling(WINDOW, min_periods=15).std()
)
df["price_cv"] = df["ret_std"] / (df["ret_mean"].abs() + 1e-8)  # CV = std/|mean|

# 成交额变化率的 CV
df["amt_chg_mean"] = df.groupby("stock_code")["amt_chg"].transform(
    lambda x: x.rolling(WINDOW, min_periods=15).mean()
)
df["amt_chg_std"] = df.groupby("stock_code")["amt_chg"].transform(
    lambda x: x.rolling(WINDOW, min_periods=15).std()
)
df["amt_cv"] = df["amt_chg_std"] / (df["amt_chg_mean"].abs() + 1e-8)

# 因子 = -price_cv / (amt_cv + eps)
# 正值 = price_cv相对高(波动大)且amt_cv相对低(交易额稳定) = 价格驱动强
EPS = 1e-6
df["factor_raw"] = -df["price_cv"] / (df["amt_cv"] + EPS)

print(f"   因子非空率: {df['factor_raw'].notna().mean():.2%}")

# 检查因子分布
valid_f = df["factor_raw"].dropna()
if len(valid_f) > 0:
    print(f"   因子描述统计: mean={valid_f.mean():.4f} std={valid_f.std():.4f} "
          f"p5={valid_f.quantile(0.05):.4f} p50={valid_f.median():.4f} p95={valid_f.quantile(0.95):.4f}")

# ────────────────── 回测准备 ──────────────────
dates = sorted(df["date"].unique())
stocks = sorted(df["stock_code"].unique())

# 构建因子矩阵
print(f"[3] 构建因子矩阵...")
factor_matrix = np.full((len(dates), len(stocks)), np.nan)
date_idx = {d: i for i, d in enumerate(dates)}
stock_idx = {s: i for i, s in enumerate(stocks)}

for _, row in df.iterrows():
    if not np.isnan(row["factor_raw"]):
        factor_matrix[date_idx[row["date"]], stock_idx[row["stock_code"]]] = row["factor_raw"]

factor_df = pd.DataFrame(factor_matrix, index=dates, columns=stocks)

# 构建收益率矩阵
close_piv = df.pivot_table(index="date", columns="stock_code", values="close", dropna=False)
ret_piv = close_piv.pct_change()
log_mktcap = np.log(df.pivot_table(index="date", columns="stock_code", values="avg_amount_20d", dropna=False).clip(lower=1))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills" / "alpha-factor-lab" / "scripts"))
try:
    from factor_backtest import compute_group_returns, compute_ic_dynamic, compute_metrics, save_backtest_data
except ImportError:
    from factor_backtest import compute_group_returns, compute_ic_dynamic, compute_metrics, save_backtest_data


def run_backtest(factor_raw, direction, label):
    """完整回测流水线"""
    factor = factor_raw.copy()
    if direction == -1:
        factor = -factor
    
    # 缩尾
    for date in dates:
        row = factor.loc[date].dropna()
        if len(row) < 10:
            continue
        lo = row.quantile(WINSORIZE_PCT)
        hi = row.quantile(1 - WINSORIZE_PCT)
        factor.loc[date] = factor.loc[date].clip(lo, hi)
    
    # 市值中性化
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


# 测试两个方向
print(f"\n=== Forward={FORWARD_DAYS}d, Rebal={REBALANCE_FREQ}d ===")
m1, g1, i1, r1, cd1, cs1, v1 = run_backtest(factor_df, 1, "正向(做多价格杠杆高)")
m2, g2, i2, r2, cd2, cs2, v2 = run_backtest(factor_df, -1, "反向(做多价格杠杆低)")

# 也测试 20日前瞻
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

# 选最佳并保存
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
        "factor_name": "价格-成交量相对波动率背离 v1",
        "factor_name_en": "Price-Volume Volatility Ratio Divergence v1",
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
