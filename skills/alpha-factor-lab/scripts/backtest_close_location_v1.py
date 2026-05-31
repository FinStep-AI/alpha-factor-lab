#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: close_location_v1 — 尾盘异动因子 (Close Location Value)
==============================================================
方向: 尾盘异动（Close Location Value = CLV）

逻辑:
  CLV = (2×Close - High - Low) / (High - Low)，取值 [-1, +1]
  +1: 收盘在最高价（尾盘强势，机构/知情交易者买入）
  -1: 收盘在最低价（尾盘弱势，散户砸盘）

  复合因子（反向评分）：
    0.5 × CLV_20日均值  +  0.25 × CLV_趋势  +  0.25 × 量加权CLV均值
  反向使用：高 CLV → 后续收益差
    - 尾盘持续强势 → 已被过度追捧，追涨者接盘
    - 尾盘持续弱势 → 卖压耗尽，后续均值回复
  A股特色: 14:57集合竞价、ETF尾盘调仓、游资尾盘打板/炸板

回测参数:
  WINDOW=20, FORWARD_DAYS=20, REBALANCE=20, N_GROUPS=5, COST=0.002

参考:
  - Arms (1989) "Volume Cycles in the Stock Market"
  - 海通证券《收盘价位置因子研究》
  - Berkman et al. (2012)
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
WINDOW         = 20
FORWARD_DAYS   = 20
REBALANCE_FREQ = 20
N_GROUPS       = 5
COST           = 0.002
WINSORIZE_PCT  = 0.05
DATA_CUTOFF    = "2026-05-01"
FACTOR_ID      = "close_location_v1"

BASE_DIR     = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH    = BASE_DIR / "data" / "csi1000_kline_raw.csv"
SCRIPTS_DIR  = BASE_DIR / "skills" / "alpha-factor-lab" / "scripts"
OUTPUT_DIR   = BASE_DIR / "output" / FACTOR_ID
REPORT_PATH  = OUTPUT_DIR / "backtest_report.json"

# ────────────────── sys.path ──────────────────
sys.path.insert(0, str(SCRIPTS_DIR))

# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据 (cutoff={DATA_CUTOFF})...")
df = pd.read_csv(DATA_PATH, encoding="utf-8")
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] <= DATA_CUTOFF].copy()
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_p   = df.pivot_table(index="date", columns="stock_code", values="close")
open_p    = df.pivot_table(index="date", columns="stock_code", values="open")
high_p    = df.pivot_table(index="date", columns="stock_code", values="high")
low_p     = df.pivot_table(index="date", columns="stock_code", values="low")
amount    = df.pivot_table(index="date", columns="stock_code", values="amount")
turnover  = df.pivot_table(index="date", columns="stock_code", values="turnover")

dates  = close_p.index.tolist()
stocks = close_p.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造尾盘异动因子 (CLV, window={WINDOW}d)...")

# CLV daily
rng    = (high_p - low_p).clip(lower=1e-6)
clv    = (2 * close_p - high_p - low_p) / rng
clv    = clv.clip(-1, 1)

# 组件1: 20日CLV均值
clv_mean    = clv.rolling(WINDOW, min_periods=int(WINDOW*0.75)).mean()

# 组件2: CLV趋势  = 近5日均值 - 20日均值
clv_short5  = clv.rolling(5,  min_periods=3).mean()
clv_trend   = clv_short5 - clv_mean

# 组件3: 量加权CLV均值  （放量日CLV权重更大）
log_amt     = np.log(amount.clip(lower=1))
log_amt_ma  = log_amt.rolling(WINDOW, min_periods=int(WINDOW*0.75)).mean()
vol_rel     = (log_amt - log_amt_ma).clip(-2, 2)
clv_vol_w   = clv * (1 + vol_rel.clip(0, 2))   # 只放大量侧
clv_vol_mean = clv_vol_w.rolling(WINDOW, min_periods=int(WINDOW*0.75)).mean()

# 截面 z-score（每列 = 当天截面）
for mat in [clv_mean, clv_trend, clv_vol_mean]:
    mu  = mat.mean(axis=1).values[:, None]
    sd  = mat.std(axis=1).values[:, None]
    mat.values[:] = np.where(sd > 1e-8, (mat.values - mu) / sd, 0)

# 反向复合因子
factor_raw = -(0.50 * clv_mean.values +
               0.25 * clv_trend.values +
               0.25 * clv_vol_mean.values)
factor_raw = pd.DataFrame(factor_raw, index=dates, columns=stocks)

cov_pct = factor_raw.notna().mean().mean()
print(f"   非空率: {cov_pct:.2%}")

# ────────────────── 5% 缩尾 ──────────────────
print(f"[3] 缩尾 ({WINSORIZE_PCT*100:.0f}%)...")
for d in dates:
    row = factor_raw.loc[d].dropna()
    if len(row) < 10:
        continue
    lo = row.quantile(WINSORIZE_PCT); hi = row.quantile(1 - WINSORIZE_PCT)
    factor_raw.loc[d] = factor_raw.loc[d].clip(lo, hi)

# ────────────────── 成交额中性化 OLS ──────────────────
print(f"[4] 成交额OLS中性化 (20d均值)...")
log_amt_20 = np.log(amount.rolling(WINDOW, min_periods=int(WINDOW*0.75)).mean().clip(lower=1))

factor_neutral = factor_raw.copy()
for d in dates:
    f = factor_raw.loc[d].dropna()
    m = log_amt_20.loc[d].reindex(f.index).dropna()
    c = f.index.intersection(m.index)
    if len(c) < 30:
        continue
    y = f[c].values; x = m[c].values
    x_dm = x - x.mean(); y_dm = y - y.mean()
    beta = float(np.nansum(x_dm * y_dm) / (np.nansum(x_dm ** 2) + 1e-10))
    alpha = float(y.mean() - beta * x.mean())
    factor_neutral.loc[d, c] = y_dm - beta * x_dm

# ────────────────── 最终 Z-score ──────────────────
print(f"[5] 截面标准化...")
mu_all  = factor_neutral.mean(axis=1).values[:, None]
sd_all  = factor_neutral.std(axis=1).values[:, None]
factor_neutral.values[:] = np.where(sd_all > 1e-8,
                                    (factor_neutral.values - mu_all) / sd_all, 0)
factor_neutral = factor_neutral.clip(-3, 3)

# ────────────────── 收益矩阵 ──────────────────
print(f"[6] 构造收益矩阵...")
ret = close_p.pct_change()
log_ret = np.log1p(ret.clip(lower=-0.999))
fwd_cum_log = log_ret.cumsum().shift(-FORWARD_DAYS) - log_ret.cumsum()
fwd_ret = np.expm1(fwd_cum_log)

common_dates = sorted(factor_neutral.dropna(how="all").index
                               .intersection(ret.dropna(how="all").index)
                               .intersection(fwd_ret.dropna(how="all").index))
common_stocks = sorted(factor_neutral.columns.intersection(ret.columns))
fa = factor_neutral.loc[common_dates, common_stocks]
ra = ret.loc[common_dates, common_stocks]

# ────────────────── 方向探索 ──────────────────
print(f"[7] 方向探索 (forward={FORWARD_DAYS}d, rebal={REBALANCE_FREQ}d, cost={COST})...")
from factor_backtest import (compute_group_returns, compute_ic_dynamic,
                              compute_metrics, save_backtest_data,
                              newey_west_t_stat)

ic_pos   = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "pearson")
gr_pos,_,hi_pos = compute_group_returns(fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_pos    = compute_metrics(gr_pos, ic_pos, ic_pos, None, N_GROUPS, holdings_info=hi_pos)

ic_neg   = compute_ic_dynamic(-fa, ra, FORWARD_DAYS, "pearson")
gr_neg,_,hi_neg = compute_group_returns(-fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_neg    = compute_metrics(gr_neg, ic_neg, ic_neg, None, N_GROUPS, holdings_info=hi_neg)

pos_sh, neg_sh = m_pos.get("long_short_sharpe",0) or 0, m_neg.get("long_short_sharpe",0) or 0
pos_ic, neg_ic = m_pos.get("ic_mean",0) or 0,        m_neg.get("ic_mean",0) or 0

print(f"   正向 IC={pos_ic:+.4f} Sharpe={pos_sh:+.4f}")
print(f"   反向 IC={neg_ic:+.4f} Sharpe={neg_sh:+.4f}")

if neg_sh > pos_sh:
    direction = -1; fa_use = -fa
    dir_desc  = "反向（低CLV=低尾盘异动=高预期收益）"
else:
    direction =  1; fa_use = fa
    dir_desc  = "正向（高CLV=高预期收益）"
print(f"   → 采用{'反向' if direction<0 else '正向'}")

# ────────────────── 最终回测 ──────────────────
print(f"[8] 最终回测...")

ic_pearson  = compute_ic_dynamic(fa_use, ra, FORWARD_DAYS, "pearson")
ic_spearman = compute_ic_dynamic(fa_use, ra, FORWARD_DAYS, "spearman")

gr, turns, hi = compute_group_returns(fa_use, ra, N_GROUPS, REBALANCE_FREQ, COST)
metrics = compute_metrics(gr, ic_pearson, ic_spearman, turns, N_GROUPS,
                          holdings_info=hi)

# ────────────────── 相关性 ──────────────────
print(f"[9] 与已有因子相关性...")
corrs = {}
other_factors = {
    "amihud_illiq_v2":     np.log((ret.abs() / (amount/1e8).clip(lower=1e-8))
                                  .rolling(WINDOW, min_periods=10).mean().clip(lower=1e-12)),
    "shadow_pressure_v1":  ((high_p-np.maximum(close_p,open_p)) -
                            (np.minimum(close_p,open_p)-low_p)
                           ).rolling(WINDOW, min_periods=10).mean(),
    "turnover_level_v1":   np.log(turnover.rolling(WINDOW,min_periods=10).mean().clip(lower=1e-8)),
    "vol_skew_v1":         None,   # 不在本文件，跳过
}
for name, other in other_factors.items():
    if other is None: continue
    vs = []
    for d in common_dates[::FORWARD_DAYS]:
        f1 = fa_use.loc[d].dropna()
        f2 = other.loc[d].reindex(f1.index).dropna()
        c  = f1.index.intersection(f2.index)
        if len(c) > 50:
            r, _ = sp_stats.spearmanr(f1[c], f2[c])
            if not np.isnan(r): vs.append(r)
    corrs[name] = round(float(np.mean(vs)), 3) if vs else None

# ────────────────── 输出 ──────────────────
print(f"[10] 写出输出...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_backtest_data(gr, ic_pearson, ic_spearman, str(OUTPUT_DIR))

nw = newey_west_t_stat(ic_pearson)

ic_mean  = float(metrics.get("ic_mean", 0) or 0)
ic_ir    = float(metrics.get("ir", 0) or 0)
t_nw     = float(nw.get("t_stat", 0) or 0)
ls_sh    = float(metrics.get("long_short_sharpe", 0) or 0)
mono     = float(metrics.get("monotonicity", 0) or 0)
ls_cum   = float(metrics.get("long_short_cumulative_return", 0) or 0)

is_valid = abs(ic_mean) > 0.015 and abs(t_nw) > 2 and abs(ls_sh) > 0.5

# cumulative_returns 由 save_backtest_data 产生，补充 ic_series.json
ic_pearson.index = ic_pearson.index.strftime("%Y-%m-%d")
ic_series_dict = {d: float(v) for d, v in ic_pearson.items()}
(OUTPUT_DIR / "ic_series.json").write_text(
    json.dumps(ic_series_dict, indent=2), encoding="utf-8")

report = {
    "factor_id"       : FACTOR_ID,
    "direction"       : "short_high" if direction < 0 else "long_high",
    "direction_desc"  : dir_desc,
    "window"          : WINDOW,
    "forward_days"    : FORWARD_DAYS,
    "rebalance_freq"  : REBALANCE_FREQ,
    "n_groups"        : N_GROUPS,
    "cost_bps"        : int(COST*10000),
    "period"          : f"{common_dates[0].date()} ~ {common_dates[-1].date()}",
    "n_stocks"        : len(common_stocks),
    "ic_mean"         : round(ic_mean, 6),
    "ic_std"          : round(float(metrics.get("ic_std",0) or 0), 6),
    "ic_ir"           : round(ic_ir, 4),
    "ic_positive_rate": round(float(metrics.get("ic_positive_rate",0) or 0), 4),
    "t_stat_nw"       : round(t_nw, 4),
    "t_stat_p"        : round(float(nw.get("p_value",1)) , 4),
    "significant_5pct": bool(nw.get("significant_5pct", False)),
    "long_short_sharpe": round(ls_sh, 4),
    "long_short_mdd"  : round(float(metrics.get("long_short_mdd",0) or 0), 4),
    "long_short_ann_ret": round(float(metrics.get("long_short_ann_return",0) or 0), 6),
    "long_short_cum_return": round(ls_cum, 6),
    "monotonicity"    : round(mono, 4),
    "group_returns_ann": [round(float(x),6) if x is not None and not np.isnan(float(x)) else None
                          for x in metrics.get("group_returns_annualized", [])],
    "group_sharpe"    : [round(float(x),4) if x is not None and not np.isnan(float(x)) else None
                          for x in metrics.get("group_sharpe", [])],
    "turnover_mean"   : round(float(metrics.get("turnover_mean",0) or 0), 4),
    "valid"           : is_valid,
    "correlations"    : corrs,
}
REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

# ────────────────── 汇报 ──────────────────
print(f"\n{'='*62}")
print(f"  {FACTOR_ID}: 尾盘异动（CLV）")
print(f"  方向: {dir_desc}")
print(f"{'='*62}")
print(f"  区间:          {report['period']}")
print(f"  股票:          {report['n_stocks']}")
print(f"  IC均值:        {ic_mean:+.4f}  t_NW={t_nw:.2f}  p={report['t_stat_p']:.3f}  {'✓ sig' if report['significant_5pct'] else '✗ ns'}")
print(f"  IR:            {ic_ir:.4f}")
print(f"  多空Sharpe:    {ls_sh:+.4f}")
print(f"  多空MDD:       {report['long_short_mdd']:.2%}")
print(f"  多空年化收益:  {report['long_short_ann_ret']:+.2%}")
print(f"  单调性:        {mono:.4f}")
print(f"  换手率均值:    {report['turnover_mean']:.2%}")
print(f"{'─'*62}")
for i, r in enumerate(metrics.get("group_returns_annualized", []), 1):
    r_str = f"{r:+.2%}" if r is not None and not np.isnan(float(r)) else "N/A"
    print(f"    G{i}: {r_str}")
print(f"{'─'*62}")
for k, v in corrs.items():
    print(f"   vs {k}: r={v:.3f}" if v is not None else f"   vs {k}: skip")
print(f"{'='*62}")
print(f"\n  ➤ 因子{'有效 ✓ 达标写入' if is_valid else '无效 ✗ 未达标'}")
if not is_valid:
    reasons=[]
    if abs(ic_mean)<=0.015: reasons.append(f"|IC|={abs(ic_mean):.4f}≤0.015")
    if abs(t_nw)<=2:       reasons.append(f"|t_NW|={abs(t_nw):.2f}≤2")
    if abs(ls_sh)<=0.5:    reasons.append(f"|Sharpe|={abs(ls_sh):.4f}≤0.5")
    for r in reasons: print(f"    ✗ {r}")
