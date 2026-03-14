#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: tr_ratio_v1 — 真实波幅比
构造:
  1. True Range = max(high-low, |high-prev_close|, |low-prev_close|)
  2. Simple Range = high - low
  3. tr_ratio = True Range / Simple Range (>= 1.0)
  4. 20日均值 + 市值中性化
  5. 反向使用: 做多低TR比(日内驱动,无大缺口)

逻辑:
- 高TR比: 波动性主要来自跳空缺口(信息冲击/风险事件)
- 低TR比: 波动性主要来自日内交易(供需博弈)
- 低TR比的股票价格发现在日内完成,信息效率更高,后续走势更可预测

也测试: 5日周期(短期信号可能更强)
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WINDOW = 20
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
log_mktcap = np.log(amount_piv.rolling(20).mean().clip(lower=1))

prev_close = close_piv.shift(1)
simple_range = high_piv - low_piv
tr1 = simple_range
tr2 = (high_piv - prev_close).abs()
tr3 = (low_piv - prev_close).abs()
true_range = pd.DataFrame(np.maximum(np.maximum(tr1.values, tr2.values), tr3.values),
                          index=close_piv.index, columns=close_piv.columns)

# 避免除0
tr_ratio_daily = true_range / simple_range.clip(lower=1e-6)
# TR比 >= 1, cap at reasonable level
tr_ratio_daily = tr_ratio_daily.clip(upper=10)

# 20日均值
tr_ratio_20d = tr_ratio_daily.rolling(WINDOW).mean()

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")


def full_pipeline(factor_raw, direction, fwd, rebal, cost, factor_id, label):
    """缩尾+中性化+回测"""
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
    
    # 中性化
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
            factor.loc[date, common] = f_c - X @ beta
        except:
            pass
    
    cd = sorted(factor.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
    cs = sorted(factor.columns.intersection(ret_piv.columns))
    fa = factor.loc[cd, cs]
    ra = ret_piv.loc[cd, cs]
    
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills" / "alpha-factor-lab" / "scripts"))
    from factor_backtest import (
        compute_group_returns, compute_ic_dynamic,
        compute_metrics, save_backtest_data
    )
    
    ic = compute_ic_dynamic(fa, ra, fwd, "pearson")
    ric = compute_ic_dynamic(fa, ra, fwd, "spearman")
    gr, to, hi = compute_group_returns(fa, ra, 5, rebal, cost)
    metrics = compute_metrics(gr, ic, ric, to, 5, holdings_info=hi)
    
    ic_m = metrics.get("ic_mean", 0) or 0
    ic_t = metrics.get("ic_t_stat", 0) or 0
    ls_sh = metrics.get("long_short_sharpe", 0) or 0
    mono = metrics.get("monotonicity", 0) or 0
    sig = "✓" if metrics.get("ic_significant_5pct") else "✗"
    
    print(f"  [{label}] IC={ic_m:.4f}(t={ic_t:.2f},{sig}) Sh={ls_sh:.2f} Mono={mono:.2f}")
    grp = metrics.get("group_returns_annualized", [])
    for i, r in enumerate(grp, 1):
        print(f"    G{i}: {r:.2%}" if r is not None else f"    G{i}: N/A")
    
    is_valid = abs(ic_m) > 0.015 and abs(ic_t) > 2 and abs(ls_sh) > 0.5
    print(f"  ➤ {'有效 ✓' if is_valid else '无效 ✗'}")
    
    if is_valid:
        out = BASE_OUTPUT / factor_id
        out.mkdir(parents=True, exist_ok=True)
        save_backtest_data(gr, ic, ric, str(out))
    
    return metrics, gr, ic, ric, cd, cs, is_valid


# 测试多种配置
configs = [
    (tr_ratio_20d, -1, 5, 5, 0.003, "tr_ratio_neg_5d", "反向 fwd=5d"),
    (tr_ratio_20d, 1, 5, 5, 0.003, "tr_ratio_pos_5d", "正向 fwd=5d"),
    (tr_ratio_20d, -1, 20, 20, 0.002, "tr_ratio_neg_20d", "反向 fwd=20d"),
    (tr_ratio_20d, 1, 20, 20, 0.002, "tr_ratio_pos_20d", "正向 fwd=20d"),
]

print(f"\n[2] 测试TR比因子各配置...")
results = []
for fraw, d, fwd, reb, cost, fid, lab in configs:
    r = full_pipeline(fraw, d, fwd, reb, cost, fid, lab)
    results.append((r, d, fwd, fid))

# 任何有效的?
valid = [(r, d, fwd, fid) for r, d, fwd, fid in results if r[-1]]
if valid:
    best = max(valid, key=lambda x: abs(x[0][0].get("long_short_sharpe",0) or 0))
    print(f"\n✅ 最佳配置: {best[3]}")
else:
    print(f"\n❌ TR比因子所有配置均未达标")
    
# 也测试: 5日短窗口TR比
print(f"\n[3] 测试5日短窗口TR比...")
tr_ratio_5d = tr_ratio_daily.rolling(5).mean()
for d, lab in [(-1, "反向5日窗口"), (1, "正向5日窗口")]:
    full_pipeline(tr_ratio_5d, d, 5, 5, 0.003, f"tr_ratio_5w_{['neg','pos'][d==1]}", lab)
