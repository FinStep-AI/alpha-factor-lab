#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: cmf_v1 — Chaikin Money Flow (资金流量)
构造:
  1. MF_Multiplier = (2*close - high - low) / (high - low)  ∈ [-1, 1]
  2. MF_Volume = MF_Multiplier * amount
  3. CMF = sum(MF_Volume, 20) / sum(amount, 20)
  4. 正向使用: CMF>0 = 收盘偏上方+放量 = 资金流入
  5. 市值中性化(OLS) + 5%缩尾

学术/实务背景:
- Chaikin Money Flow (Marc Chaikin, 1980s)
- 与close_location_v1的区别: CMF加权了成交额,大量日的信号权重更大
- 与close_vwap_dev_v1的区别: CMF用(2C-H-L)/(H-L)更标准化,且是累积比值而非偏离度

也测试反向(做多资金流出=均值回复)和不同周期
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

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# MF Multiplier
hl_range = (high_piv - low_piv).clip(lower=0.001)
mf_mult = (2 * close_piv - high_piv - low_piv) / hl_range  # ∈ [-1, 1]
mf_volume = mf_mult * amount_piv

# CMF with different windows
print(f"[2] 构造 CMF 因子...")
cmf_20d = mf_volume.rolling(20).sum() / amount_piv.rolling(20).sum().clip(lower=1)
cmf_10d = mf_volume.rolling(10).sum() / amount_piv.rolling(10).sum().clip(lower=1)
print(f"   CMF_20d 非空率: {cmf_20d.notna().mean().mean():.2%}")
print(f"   CMF_10d 非空率: {cmf_10d.notna().mean().mean():.2%}")


def full_pipeline(factor_raw, direction, fwd, rebal, cost, factor_id, label):
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
            "factor_name": f"CMF资金流量 {label}",
            "direction": direction,
            "forward_days": fwd,
            "rebalance_freq": rebal,
            "cost": cost,
            "period": f"{cd[0].strftime('%Y-%m-%d')} ~ {cd[-1].strftime('%Y-%m-%d')}",
            "n_stocks": len(cs),
            "metrics": metrics,
        }
        with open(out / "backtest_report.json", "w", encoding="utf-8") as f:
            json.dump(nan_to_none(report), f, indent=2, ensure_ascii=False)
        print(f"  ✅ 报告已保存: {out / 'backtest_report.json'}")
    
    return metrics, is_valid


print(f"\n[3] 回测各配置...")
configs = [
    # CMF 20d
    (cmf_20d, 1, 5, 5, 0.003, "cmf_20d_pos_5d", "CMF20d正向 fwd=5d"),
    (cmf_20d, -1, 5, 5, 0.003, "cmf_20d_neg_5d", "CMF20d反向 fwd=5d"),
    (cmf_20d, 1, 20, 20, 0.002, "cmf_20d_pos_20d", "CMF20d正向 fwd=20d"),
    (cmf_20d, -1, 20, 20, 0.002, "cmf_20d_neg_20d", "CMF20d反向 fwd=20d"),
    # CMF 10d
    (cmf_10d, 1, 5, 5, 0.003, "cmf_10d_pos_5d", "CMF10d正向 fwd=5d"),
    (cmf_10d, -1, 5, 5, 0.003, "cmf_10d_neg_5d", "CMF10d反向 fwd=5d"),
]

any_valid = False
for fraw, d, fwd, reb, cost, fid, lab in configs:
    m, v = full_pipeline(fraw, d, fwd, reb, cost, fid, lab)
    if v:
        any_valid = True

if not any_valid:
    print(f"\n❌ CMF 因子所有配置均未达标")
