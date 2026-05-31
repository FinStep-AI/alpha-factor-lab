#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: turnover_decay_v1 — 换手率衰减因子
========================================
factor_id: turnover_decay_v1

逻辑:
  raw = log( MA5(turnover) / MA20(turnover) )
  换手率衰减比 < 1 代表近期流动性在枯竭（卖压耗尽 / 筹码集中）
  再取负 → 衰减越深(raw负值) 因子值越高 → 做多（均值回复假设）

中性化: 20日成交额均值 OLS，5% 缩尾，截面 z-score
"""

import json, sys, warnings
from pathlib import Path

import numpy as np, pandas as pd
from scipy import stats as sp_stats
warnings.filterwarnings("ignore")

WINDOW         = 20
FORWARD_DAYS   = 20
REBALANCE_FREQ = 20
N_GROUPS       = 5
COST           = 0.002
WINSORIZE_PCT  = 0.05
DATA_CUTOFF    = "2026-05-01"
FACTOR_ID      = "turnover_decay_v1"

BASE_DIR    = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH   = BASE_DIR / "data" / "csi1000_kline_raw.csv"
SCRIPTS_DIR = BASE_DIR / "skills" / "alpha-factor-lab" / "scripts"
OUTPUT_DIR  = BASE_DIR / "output" / FACTOR_ID

sys.path.insert(0, str(SCRIPTS_DIR))
from factor_backtest import (compute_group_returns, compute_ic_dynamic,
                              compute_metrics, save_backtest_data,
                              newey_west_t_stat)

def build_factor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["stock_code", "date"]).copy()

    # turnover: treat 0 as missing
    turn = df["turnover"].replace(0, np.nan)

    ma5  = turn.groupby(df["stock_code"]).transform(
        lambda x: x.rolling(5, min_periods=3).mean())
    ma20 = turn.groupby(df["stock_code"]).transform(
        lambda x: x.rolling(20, min_periods=15).mean())

    ratio = ma5 / ma20.replace(0, np.nan)          # <1 = drying up
    raw   = np.log(ratio.clip(lower=1e-6))

    # mktcap proxy = amount / turnover (same convention as builder)
    turn_nz   = df["turnover"].replace(0, np.nan)
    mktcap_px = df["amount"] / turn_nz
    log_mktcap = np.log(mktcap_px.replace(0, np.nan))

    out = df[["date","stock_code"]].copy()
    out["raw_factor"] = raw.values
    out["log_mktcap"] = log_mktcap.values
    return out.dropna(subset=["raw_factor","log_mktcap"])

def neutralize_and_standardize(long_df: pd.DataFrame) -> pd.DataFrame:
    # 5% winsorize cross-sectionally
    def wsc(s):
        lo, hi = s.quantile(WINSORIZE_PCT), s.quantile(1-WINSORIZE_PCT)
        return s.clip(lo, hi)
    long_df = long_df.copy()
    long_df["raw_factor"] = long_df.groupby("date")["raw_factor"].transform(wsc)

    # cross-sectional z-score
    long_df["factor_zscore"] = long_df.groupby("date")["raw_factor"].transform(
        lambda x: (x-x.mean())/x.std() if x.std()>0 else 0)
    long_df["factor_zscore"] = long_df["factor_zscore"].clip(-3,3)

    # OLS neutralize on log_mktcap
    def neu(g):
        if len(g) < 10: g = g.copy(); g["factor_neu"]=np.nan; return g[["factor_neu"]]
        x = g["log_mktcap"].values; y = g["factor_zscore"].values
        xm = x-np.nanmean(x); ym = y-np.nanmean(y)
        b  = float(np.nansum(xm*ym)/(np.nansum(xm**2)+1e-10))
        a  = float(np.nanmean(y) - b*np.nanmean(x))
        g = g.copy(); g["factor_neu"] = y - (a + b*x); return g[["factor_neu"]]

    out = long_df.groupby("date", group_keys=False).apply(neu)
    long_df["factor_neu"] = out["factor_neu"].values

    long_df["factor_value"] = long_df.groupby("date")["factor_neu"].transform(
        lambda x: (x-x.mean())/x.std() if x.std()>0 else 0)
    return long_df[["date","stock_code","factor_value"]].dropna()

# ────────────────── main ──────────────────
print(f"[1] 构建 turnover_decay_v1 因子 …")
df = pd.read_csv(DATA_PATH, encoding="utf-8")
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] <= DATA_CUTOFF].copy()
print(f"   rows={len(df)}  stocks={df.stock_code.nunique()}")

raw = build_factor(df)
print(f"   raw factor rows={len(raw)}")

print(f"[2] 中性化 + 缩尾 + z-score …")
panel = neutralize_and_standardize(raw)
panel["date"] = panel["date"].dt.strftime("%Y-%m-%d")

# pivots
close_p = df.pivot_table(index="date", columns="stock_code", values="close")
dates   = sorted(panel["date"].unique().tolist())
stocks  = sorted(panel["stock_code"].unique().tolist())
panel["dt"] = pd.to_datetime(panel["date"])

factor_mat = panel.pivot_table(index="dt", columns="stock_code", values="factor_value")
factor_mat = factor_mat.sort_index()

ret = close_p.pct_change()
log_ret    = np.log1p(ret.clip(lower=-0.999))
fwd_cumlog = log_ret.cumsum().shift(-FORWARD_DAYS) - log_ret.cumsum()
fwd_ret    = np.expm1(fwd_cumlog)

# align
pop_dates  = sorted(factor_mat.dropna(how="all").index.intersection(ret.dropna(how="all").index))
pop_stocks = sorted(factor_mat.columns.intersection(ret.columns))
fa = factor_mat.loc[pop_dates, pop_stocks]
ra = ret.loc[pop_dates, pop_stocks]

# ────────────────── 方向探索 ──────────────────
print(f"[3] 方向探索 …")
ic_p   = compute_ic_dynamic( fa, ra, FORWARD_DAYS, "pearson")
gr_p,_,hp = compute_group_returns(fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
mp    = compute_metrics(gr_p, ic_p, ic_p, None, N_GROUPS, holdings_info=hp)

ic_n   = compute_ic_dynamic(-fa, ra, FORWARD_DAYS, "pearson")
gr_n,_,hn = compute_group_returns(-fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
mn    = compute_metrics(gr_n, ic_n, ic_n, None, N_GROUPS, holdings_info=hn)

p_ic, n_ic = float(mp.get("ic_mean",0) or 0), float(mn.get("ic_mean",0) or 0)
p_sh, n_sh = float(mp.get("long_short_sharpe",0) or 0), float(mn.get("long_short_sharpe",0) or 0)
print(f"   正向 IC={p_ic:+.4f} Sharpe={p_sh:+.4f}")
print(f"   反向 IC={n_ic:+.4f} Sharpe={n_sh:+.4f}")

if n_sh > p_sh:
    fa_use = -fa; direction="short_high"
    dir_desc="反向（换手率衰减深的做空 / 换手率正常偏高做多）"
else:
    fa_use =  fa; direction="long_high"
    dir_desc="正向（换手率衰减深的做多 / 换手率恢复看涨）"
print(f"   → {dir_desc}")

# ────────────────── 最终回测 ──────────────────
print(f"[4] 最终回测 …")
ic_pa = compute_ic_dynamic(fa_use, ra, FORWARD_DAYS, "pearson")
ic_ps = compute_ic_dynamic(fa_use, ra, FORWARD_DAYS, "spearman")
gr, turns, hi = compute_group_returns(fa_use, ra, N_GROUPS, REBALANCE_FREQ, COST)
me  = compute_metrics(gr, ic_pa, ic_ps, turns, N_GROUPS, holdings_info=hi)
nw  = newey_west_t_stat(ic_pa)

ic_m    = float(me.get("ic_mean",0) or 0)
ic_ir   = float(me.get("ir",0) or 0)
t_nw    = float(nw.get("t_stat",0) or 0)
ls_sh   = float(me.get("long_short_sharpe",0) or 0)
mono    = float(me.get("monotonicity",0) or 0)
ls_cum  = float(me.get("long_short_cumulative_return",0) or 0)
ls_ann  = float(me.get("long_short_ann_return",0) or 0)
ls_mdd  = float(me.get("long_short_mdd",0) or 0)
turn_mn = float(me.get("turnover_mean",0) or 0)

is_valid = abs(ic_m)>0.015 and abs(t_nw)>2 and abs(ls_sh)>0.5

# ────────────────── 写输出 ──────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_backtest_data(gr, ic_pa, ic_ps, str(OUTPUT_DIR))

report = {
    "factor_id"       : FACTOR_ID,
    "direction"       : direction,
    "direction_desc"  : dir_desc,
    "window"          : WINDOW,
    "forward_days"    : FORWARD_DAYS,
    "rebalance_freq"  : REBALANCE_FREQ,
    "n_groups"        : N_GROUPS,
    "cost_bps"        : int(COST*10000),
    "period"          : f"{pop_dates[0].date()} ~ {pop_dates[-1].date()}",
    "n_stocks"        : len(pop_stocks),
    "ic_mean"         : round(ic_m, 6),
    "ic_std"          : round(float(me.get("ic_std",0) or 0), 6),
    "ic_ir"           : round(ic_ir, 4),
    "t_stat_nw"       : round(t_nw, 4),
    "t_stat_p"        : round(float(nw.get("p_value",1)), 4),
    "significant_5pct": bool(nw.get("significant_5pct", False)),
    "long_short_sharpe": round(ls_sh, 4),
    "long_short_mdd"  : round(ls_mdd, 4),
    "long_short_ann_ret": round(ls_ann, 6),
    "long_short_cum_return": round(ls_cum, 6),
    "monotonicity"    : round(mono, 4),
    "group_returns_ann": [round(float(x),6) if x is not None and not np.isnan(float(x)) else None
                          for x in me.get("group_returns_annualized", [])],
    "turnover_mean"   : round(turn_mn, 4),
    "valid"           : is_valid,
}
(OUTPUT_DIR / "backtest_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\n{'='*62}")
print(f"  {FACTOR_ID}: 换手率衰减")
print(f"  方向: {dir_desc}")
print(f"{'='*62}")
print(f"  区间:       {report['period']}")
print(f"  股票:       {report['n_stocks']}")
print(f"  IC均值:     {ic_m:+.4f}  t_NW={t_nw:.2f}  p={report['t_stat_p']:.3f}  {'✓ sig' if report['significant_5pct'] else '✗ ns'}")
print(f"  IR:         {ic_ir:.4f}")
print(f"  多空Sharpe: {ls_sh:+.4f}")
print(f"  多空MDD:    {report['long_short_mdd']:.2%}")
print(f"  多空年化:   {report['long_short_ann_ret']:+.2%}")
print(f"  多空累计:   {ls_cum:+.2%}")
print(f"  单调性:     {mono:.4f}")
print(f"  换手率:     {turn_mn:.2%}")
print(f"{'─'*62}")
for i, r in enumerate(me.get("group_returns_annualized", []), 1):
    r_s = f"{r:+.2%}" if r is not None and not np.isnan(float(r)) else "N/A"
    print(f"    G{i}: {r_s}")
print(f"{'='*62}")
print(f"\n  ➤ 因子{'有效 ✓ 达标写入' if is_valid else '无效 ✗ 未达标'}")
if not is_valid:
    for tag,val in [("IC",abs(ic_m)),("t_NW",abs(t_nw)),("Sharpe",abs(ls_sh))]:
        if val <= [0.015,2,0.5][["IC","t_NW","Sharpe"].index(tag)]:
            lim = [0.015,2,0.5][["IC","t_NW","Sharpe"].index(tag)]
            print(f"    ✗ |{tag}|={val:.4f} ≤ {lim}")
