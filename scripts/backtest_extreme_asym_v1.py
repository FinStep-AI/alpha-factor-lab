#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: extreme_asym_v1 — 极端收益不对称性
构造:
  1. 过去20日:
     top2 = 最好2天收益率的均值
     bot2 = 最差2天收益率的均值
     factor = top2 + bot2  (注: bot2是负数, 所以这是 best - |worst|)
  2. 正向使用: 
     正值 = 上行极端收益 > |下行极端收益| = 正偏尾部 → 做多
     负值 = |下行极端| > 上行极端 = 负偏尾部 → 做空
  3. 与CVaR(只看下行)不同: 本因子看上下行的对比,捕捉尾部不对称性
  4. 市值中性化(OLS) + 5%缩尾

逻辑:
- 正偏尾部: 好消息冲击力 > 坏消息冲击力, 暗示基本面/市场情绪偏向正面
- 负偏尾部: 坏消息冲击更大, 暗示风险暴露多/流动性差
- 本质是测量股票对正面vs负面信息的不对称反应

也测试反向: 做多负偏尾部(均值回复逻辑)

与CVaR的区别:
- CVaR只看底部(做多无极端下跌), 是纯尾部风险回避
- 本因子看上下对比, 是信息不对称性
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

WINDOW = 20
K = 2  # top/bottom K days
FORWARD_DAYS = 5  # 尝试5日前瞻(参考CVaR用5日效果最好)
REBALANCE_FREQ = 5
N_GROUPS = 5
COST = 0.003
WINSORIZE_PCT = 0.05
FACTOR_ID = "extreme_asym_v1"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "csi1000_kline_raw.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

print(f"[1] 加载数据...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")
ret_piv = close_piv.pct_change()
log_mktcap = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ────────────────── 因子构造(向量化) ──────────────────
print(f"[2] 构造 extreme_asym 因子 (window={WINDOW}, K={K})...")

ret_vals = ret_piv.values  # (T, S)
factor_matrix = np.full_like(ret_vals, np.nan)

for i in range(WINDOW, len(dates)):
    window = ret_vals[i - WINDOW:i, :]  # (20, S)
    
    # 对每列排序
    sorted_window = np.sort(window, axis=0)  # ascending
    
    # bot K = 最差K天均值 (sorted_window[:K])
    bot_k = np.nanmean(sorted_window[:K, :], axis=0)
    # top K = 最好K天均值 (sorted_window[-K:])
    top_k = np.nanmean(sorted_window[-K:, :], axis=0)
    
    # 因子 = top_k + bot_k  (bot_k是负数)
    factor_vals = top_k + bot_k
    
    # 有效性检查
    valid_count = np.sum(~np.isnan(window), axis=0)
    factor_vals[valid_count < 15] = np.nan
    
    factor_matrix[i, :] = factor_vals

factor_df = pd.DataFrame(factor_matrix, index=dates, columns=stocks)
print(f"   因子非空率: {(~np.isnan(factor_matrix)).mean():.2%}")


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
    
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills" / "alpha-factor-lab" / "scripts"))
    from factor_backtest import (
        compute_group_returns, compute_ic_dynamic,
        compute_metrics, save_backtest_data
    )
    
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
    grp = metrics.get("group_returns_annualized", [])
    for i, r in enumerate(grp, 1):
        print(f"    G{i}: {r:.2%}" if r is not None else f"    G{i}: N/A")
    
    is_valid = abs(ic_mean) > 0.015 and abs(ic_t) > 2 and abs(ls_sh) > 0.5
    print(f"  ➤ {'有效 ✓' if is_valid else '无效 ✗'}")
    
    return metrics, gr, ic, ric, common_dates, common_stocks, is_valid


# 测试两个方向 + 两个前瞻窗口
print(f"\n=== Forward={FORWARD_DAYS}d, Rebal={REBALANCE_FREQ}d ===")
m1, g1, i1, r1, cd1, cs1, v1 = run_backtest(factor_df, 1, "正向(做多正偏)")
m2, g2, i2, r2, cd2, cs2, v2 = run_backtest(factor_df, -1, "反向(做多负偏)")

# 也测试20日前瞻
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
    from factor_backtest import (
        compute_group_returns, compute_ic_dynamic,
        compute_metrics, save_backtest_data
    )
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
    grp = metrics.get("group_returns_annualized", [])
    for i, r in enumerate(grp, 1):
        print(f"    G{i}: {r:.2%}" if r is not None else f"    G{i}: N/A")
    is_valid = abs(ic_mean) > 0.015 and abs(ic_t) > 2 and abs(ls_sh) > 0.5
    print(f"  ➤ {'有效 ✓' if is_valid else '无效 ✗'}")
    return metrics, gr, ic, ric, common_dates, common_stocks, is_valid

m3, g3, i3, r3, cd3, cs3, v3 = run_backtest_20d(factor_df, 1, "正向20d")
m4, g4, i4, r4, cd4, cs4, v4 = run_backtest_20d(factor_df, -1, "反向20d")

# 选最佳并保存
all_results = [(m1,g1,i1,r1,cd1,cs1,v1,1,5),(m2,g2,i2,r2,cd2,cs2,v2,-1,5),
               (m3,g3,i3,r3,cd3,cs3,v3,1,20),(m4,g4,i4,r4,cd4,cs4,v4,-1,20)]
valid_results = [(m,g,ic,ric,cd,cs,d,f) for m,g,ic,ric,cd,cs,v,d,f in all_results if v]

if valid_results:
    # 选Sharpe最高的
    best = max(valid_results, key=lambda x: abs(x[0].get("long_short_sharpe",0) or 0))
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
    from factor_backtest import save_backtest_data
    save_backtest_data(bg, bic, bric, str(OUTPUT_DIR))
    
    report = {
        "factor_id": FACTOR_ID,
        "factor_name": f"极端收益不对称性 v1 (dir={'pos' if bd==1 else 'neg'}, fwd={bf}d)",
        "direction": bd,
        "forward_days": bf,
        "rebalance_freq": bf,
        "cost": 0.003 if bf==5 else 0.002,
        "period": f"{bcd[0].strftime('%Y-%m-%d')} ~ {bcd[-1].strftime('%Y-%m-%d')}",
        "n_stocks": len(bcs),
        "metrics": bm,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(nan_to_none(report), f, indent=2, ensure_ascii=False)
    print(f"\n✅ 报告已保存: {REPORT_PATH}")
else:
    print(f"\n❌ 所有配置均未达标")
