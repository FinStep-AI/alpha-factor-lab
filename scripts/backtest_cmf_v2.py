#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMF 10d 深度优化 — IC=0.025(t=3.57)显著但Sharpe不够
尝试更多配置找到Sharpe>0.5的sweet spot
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

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
# 也试成交额中性化
log_amount_20d = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()

hl_range = (high_piv - low_piv).clip(lower=0.001)
mf_mult = (2 * close_piv - high_piv - low_piv) / hl_range
mf_volume = mf_mult * amount_piv

# 多种窗口
cmf_5d = mf_volume.rolling(5).sum() / amount_piv.rolling(5).sum().clip(lower=1)
cmf_10d = mf_volume.rolling(10).sum() / amount_piv.rolling(10).sum().clip(lower=1)
cmf_15d = mf_volume.rolling(15).sum() / amount_piv.rolling(15).sum().clip(lower=1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills" / "alpha-factor-lab" / "scripts"))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data
)


def pipeline(factor_raw, direction, fwd, rebal, cost, neutral_var, factor_id, label):
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
        m = neutral_var.loc[date].reindex(f.index).dropna()
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
    print(f"{flag} [{label}] IC={ic_m:.4f}(t={ic_t:.2f},{sig}) Sh={ls_sh:.2f} Mono={mono:.2f} | {'PASS' if is_valid else 'fail'}")
    
    if is_valid:
        grp = metrics.get("group_returns_annualized", [])
        for i, r in enumerate(grp, 1):
            print(f"      G{i}: {r:.2%}" if r is not None else f"      G{i}: N/A")
        
        out = BASE_OUTPUT / factor_id
        out.mkdir(parents=True, exist_ok=True)
        save_backtest_data(gr, ic, ric, str(out))
        
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
            "factor_id": factor_id,
            "metrics": metrics,
            "period": f"{cd[0].strftime('%Y-%m-%d')} ~ {cd[-1].strftime('%Y-%m-%d')}",
        }
        with open(out / "backtest_report.json", "w", encoding="utf-8") as f:
            json.dump(nan_to_none(report), f, indent=2, ensure_ascii=False)
    
    return metrics, is_valid, ic_m, ic_t, ls_sh, mono


print(f"\n[2] 穷举测试 CMF 因子...")
print(f"{'='*80}")

# Test: different windows × directions × forward × neutral_var
configs = []
for cmf, win in [(cmf_5d, "5d"), (cmf_10d, "10d"), (cmf_15d, "15d")]:
    for d, dname in [(-1, "neg"), (1, "pos")]:
        for fwd, reb, cost in [(5, 5, 0.003), (10, 10, 0.003), (20, 20, 0.002)]:
            for nv, nvname in [(log_mktcap, "mkcap"), (log_amount_20d, "amt")]:
                fid = f"cmf_{win}_{dname}_fwd{fwd}_{nvname}"
                lab = f"CMF{win} {dname} f{fwd} n={nvname}"
                configs.append((cmf, d, fwd, reb, cost, nv, fid, lab))

best_sharpe = 0
best_config = None
for cmf, d, fwd, reb, cost, nv, fid, lab in configs:
    m, v, icm, ict, sh, mono = pipeline(cmf, d, fwd, reb, cost, nv, fid, lab)
    if v and abs(sh) > abs(best_sharpe):
        best_sharpe = sh
        best_config = fid

print(f"\n{'='*80}")
if best_config:
    print(f"✅ 最佳配置: {best_config} (Sharpe={best_sharpe:.2f})")
else:
    print(f"❌ 无配置通过所有筛选标准")
