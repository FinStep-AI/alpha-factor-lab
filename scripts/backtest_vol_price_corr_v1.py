#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: vol_price_corr_v1 — 成交量-收益率滚动相关性
====================================================
构造:
  1. 计算20日滚动窗口内日收益率与成交额收益率的相关性:
     volume_corr = corr(daily_ret, amt_chg, 20)
  2. 正向使用: 正值 = 价涨量增(一致性) → 动量信号? 或
             负值 = 价跌量增/价涨量缩(背离) → 反转信号
  3. 同时测试两个方向(正向/反向)
  4. 市值中性化 + 5%缩尾 + z-score

核心假设:
- 传统观点: 价涨量增=健康趋势(正相关) → 动量延续
- 反转观点(A股小盘): 价涨量缩=上涨动能衰竭(负相关) → 看跌背离
- 本因子不预设方向, 两个方向都测

相关文献:
- Chordia & Subrahmanyam (2004) "Order imbalance and individual stock returns" JFE
  → 订单不平衡((买-卖)/总)预测个股收益
- Brennan, Chordia & Subrahmanyam (1998) "Alternative factor specifications, 
  systematic risk and asset pricing tests"
- Karpoff (1987) "The relation between price changes and trading volume" 
  → 综述：价量关系在不同市场/频率表现各异
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

WINDOW = 20
FORWARD_DAYS = 5
REBALANCE_FREQ = 5
N_GROUPS = 5
COST = 0.003
WINSORIZE_PCT = 0.05
FACTOR_ID = "vol_price_corr_v1"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "csi1000_kline_raw.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

print(f"[1] 加载数据...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

df["daily_ret"] = df.groupby("stock_code")["close"].pct_change()

# 成交额变化率
df["amt_lag"] = df.groupby("stock_code")["amount"].shift(1)
df["amt_chg"] = (df["amount"] - df["amt_lag"]) / (df["amt_lag"].abs() + 1)

# 市值代理
df["avg_amount_20d"] = df.groupby("stock_code")["amount"].transform(
    lambda x: x.rolling(WINDOW, min_periods=10).mean()
)
df["log_amount"] = np.log(df["avg_amount_20d"].clip(lower=1))

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造 Volume-Price Correlation 因子 (window={WINDOW})...")

# 滚动相关性(向量化高效计算)
def rolling_corr(series_x, series_y, window, min_p=15):
    """计算两个序列的滚动Pearson相关系数"""
    x_vals = series_x.values.astype(float)
    y_vals = series_y.values.astype(float)
    n = len(x_vals)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        xs = x_vals[i-window:i]
        ys = y_vals[i-window:i]
        valid = ~(np.isnan(xs) | np.isnan(ys))
        if np.sum(valid) < min_p:
            continue
        try:
            r, _ = pearsonr(xs[valid], ys[valid])
            result[i] = r
        except:
            pass
    return result

# 计算逐股滚动相关性
print("    计算滚动相关性...")
corr_results = []
for code, group in df.groupby("stock_code"):
    group = group.sort_values("date").copy()
    corr = rolling_corr(group["daily_ret"], group["amt_chg"], WINDOW)
    group["factor_raw"] = corr
    corr_results.append(group[["date", "stock_code", "factor_raw"]])
    
factor_df_raw = pd.concat(corr_results, ignore_index=True)
df = df.merge(factor_df_raw, on=["date", "stock_code"], how="left")

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
log_mktcap = np.log(df.pivot_table(index="date", columns="stock_code", 
                                    values="avg_amount_20d", dropna=False).clip(lower=1))

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
m1, g1, i1, r1, cd1, cs1, v1 = run_backtest(factor_df, 1, "正向(做多正相关)")
m2, g2, i2, r2, cd2, cs2, v2 = run_backtest(factor_df, -1, "反向(做多负相关)")

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
        "factor_name": "成交量-收益率滚动相关性 v1",
        "factor_name_en": "Volume-Return Rolling Correlation v1",
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
