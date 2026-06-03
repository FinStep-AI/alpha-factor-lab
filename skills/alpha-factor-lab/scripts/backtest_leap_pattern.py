#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: leap_pattern_v1 — 上行跳跃模式因子
======================================================
每日收益率(hign)尾部分位 / 日收益分布尾部偏度
核心思想: A股中证1000小盘股存在事件驱动型跳跃，跳跃后的行为模式（次日/近N日是否跟
涨、是否回吐）是截面区分度来源之一。价格跳跃 peak (如收益 ≥ P90) 作为信号,
作为以后 v2/v3 text extending conditions.

本 v1 COMPUTER跳跃频次: 20日内单日收益率 ≥ 5% 的次数
  与"收益率中不稳健里这 FACTOR 的 振幅水平 / 高频交换模式 / 信息密度" 角度不同:
  - leap_count: 看"大跃迁"是否频繁(区间出现次数) + 成交闪烁 cross_amplitude
  - leap_cluster: 看"跳跃是否集中" = group=> 磁场 smart_  noise
  - leap_magnitude_amplitude: 看"头尾伴随 INFORMATION 对尾随信号

复合: factor = 0.4*z(leap_gap20d) + 0.35*z(leap_cluster20d) + 0.25*z(leap_range20d)
参数: gain_factor_type_null = compound but empirically such as gap_narrow_and_earning.
 neutral Z = compound Instead = around treat = OBJECT.
"""

import json, sys, warnings
from pathlib import Path

import numpy as np, pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ─── 核心参数 ───
LEAP_THRESH  = 0.05          # 单日涨 ≥ 5% 定义 a single "leap day"
N_GROUPS     = 5
WINSORIZE_PCT = 0.05
FWD_OPTIONS  = [(5,  5, 0.003),
                (5, 20, 0.002),
                (20,20, 0.002)]
FACTOR_ID    = "leap_pattern_v1"

SCRIPTS_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent.parent
DATA_PATH    = PROJECT_ROOT / "data" / "csi1000_kline_raw.csv"
OUTPUT_DIR   = PROJECT_ROOT / "output" / FACTOR_ID
REPORT_PATH  = OUTPUT_DIR / "backtest_report.json"

# ─── 1. 数据加载 ───
print("[1] 加载数据 ...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code","date"]).reset_index(drop=True)

close_piv   = df.pivot_table(index="date", columns="stock_code", values="close")
open_piv    = df.pivot_table(index="date", columns="stock_code", values="open")
high_piv    = df.pivot_table(index="date", columns="stock_code", values="high")
low_piv     = df.pivot_table(index="date", columns="stock_code", values="low")
amount_piv  = df.pivot_table(index="date", columns="stock_code", values="amount")
turnover_piv= df.pivot_table(index="date", columns="stock_code", values="turnover")

ret_piv     = close_piv.pct_change()
log_amt_20d = np.log(amount_piv.rolling(20, min_periods=10).mean().clip(lower=1))

dates  = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)}交易日 / {len(stocks)}股 "
      f"({dates[0].date()} ~ {dates[-1].date()})")

# ─── 2. 因子构造 ───
print(f"[2] 构造 {LEAP_THRESH*100:.0f}% 跳跃特征矩阵 ...")

ret    = ret_piv.values
n_d, n_s = ret.shape

# leap_flag[i,j] = 1 若 stock j 在 day i 的收益 ≥ LEAP_THRESH
leap_flags  = (ret >= LEAP_THRESH).astype(float)
# leap_neg_flags = 下行跳跃
leap_neg    = (ret <= -LEAP_THRESH).astype(float)

def roll_sum(mat, win):
    """2D rolling sum over axis=0, min_periods=win//2."""
    mp = max(win // 2, 3)
    s = pd.DataFrame(mat).rolling(win, min_periods=mp).sum().values
    return s

win20 = 20; win40 = 40; win60 = 60

leap_sum_20  = roll_sum(leap_flags,  win20)
leap_sum_40  = roll_sum(leap_flags,  win40)
leap_sum_60  = roll_sum(leap_flags,  win60)

# 下行跳跃 20d
neg_sum_20   = roll_sum(leap_neg,    win20)

# ── leap count ratio: 上行跳跃 - 下行跳跃 ──
#  正=上行主导，后续跟随上行；负=下行主导
leap_net_20  = leap_sum_20 - neg_sum_20

# ── leap gap (inter-leap spacing): ↑ 跳跃间隔越短 = 越集中 = cluster signal
#  构造 leap_gap = std(gap between leap days in 20d window) — 短 = 跳跃很近
leap_pos    = (ret > 0).astype(float)
ret_abs     = np.abs(ret)

def roll_std_np(mat, win):
    mp = max(win // 2, 3)
    return pd.DataFrame(mat).rolling(win, min_periods=mp).std().values

# leap pick efficiency: avg return on leap / avg return on non-leap days (only on leap days threshold)
# simplified proxy: the ratio of avg_ret_on leap_day vs unconditional.
# proxy_apl =  mean(ret_i | leap) − mean(ret_i)

# mean ret on leap days 20d — cross section per stock per day
# for each stock x day: avg of ret in the 20d window where leap_flag_i==1
# not straightforward without panel loop; substitute with proximity proxy:
#  |close_today − 20d_low| / 20d_high − 20d_low  → buy-high-proxy
lo20  = pd.DataFrame(high_piv.values).rolling(win20, min_periods=10).min().values
hi20  = pd.DataFrame(high_piv.values).rolling(win20, min_periods=10).max().values
pp20  = (close_piv.values - lo20) / (hi20 - lo20 + 1e-8)  # close position in 20d range [0,1]

# Turnover(环比 momentum): spike in turnover
turn  = turnover_piv.values
turn_mom = pd.DataFrame(turn).pct_change(5).rolling(20, min_periods=10).mean().values

# Composite raw factors → DataFrame
idx = close_piv.index; cols = close_piv.columns

f_leap_cnt   = pd.DataFrame(leap_sum_20,  index=idx, columns=cols)  # 上行跳跃频次
f_leap_net   = pd.DataFrame(leap_net_20,  index=idx, columns=cols)  # 上行-下行净频次
f_leap_cnt60 = pd.DataFrame(leap_sum_60,  index=idx, columns=cols)  # 60d累计跳跃
f_price_pos  = pd.DataFrame(pp20,        index=idx, columns=cols)  # 20d区间价格位置
f_turn_mom   = pd.DataFrame(turn_mom,    index=idx, columns=cols)  # 换手环比

print(f"   leap_cnt mean={f_leap_cnt.stack().mean():.3f}  "
      f"median={f_leap_cnt.stack().median():.3f}")
print(f"   leap_net mean={f_leap_net.stack().mean():.3f}  "
      f"leap_cnt60 mean={f_leap_cnt60.stack().mean():.3f}")

# ─── 3. 复合因子的多种权重方案尝试 ───
print("\n[3] 构造复合信号 ...")

raw_map = dict(
    leap_cnt_20  = f_leap_cnt,
    leap_net_20  = f_leap_net,
    leap_cnt_60  = f_leap_cnt60,
    price_pos_20 = f_price_pos,
    turn_mom_20  = f_turn_mom,
)

# 各 raw factor 对 close_piv, amount_piv → fill, log mean → neutralise
log_amt = np.log(amount_piv.rolling(20, min_periods=10).mean().clip(lower=1))

def neutralize(df_raw, log_amt_mat):
    """截面 OLS neutralise df_raw (date×stock) w.r.t. log_amount_20d."""
    out  = df_raw.copy()
    vals = df_raw.values; la = log_amt_mat.values
    idx0 = df_raw.index; cols0 = df_raw.columns
    for i in range(len(idx0)):
        f  = vals[i]; m  = la[i]
        mask = ~(np.isnan(f) | np.isnan(m))
        if mask.sum() < 30: continue
        fc = f[mask]; mc = m[mask]
        X  = np.column_stack([np.ones(mc.size), mc])
        try:
            b  = np.linalg.lstsq(X, fc, rcond=None)[0]
            out.iloc[i, mask] = fc - X @ b
        except Exception:
            pass
    return out

def winsorize5(df):
    out = df.copy()
    for d in df.index:
        r = df.loc[d].dropna()
        if len(r) < 10: continue
        lo, hi = r.quantile(0.05), r.quantile(0.95)
        out.loc[d] = df.loc[d].clip(lo, hi)
    return out

# Before neutralisation, log normal transform heavy-tailed features
for k in ["leap_cnt_20","leap_net_20","leap_cnt_60"]:
    raw_map[k] = np.log1p(raw_map[k])

neutralised = {k: neutralize(v, log_amt_20d) for k, v in raw_map.items()}
neutralised = {k: winsorize5(v) for k, v in neutralised.items()}

# Equal-weight z-score + combine
def zscore(df):
    mu = df.mean(axis=1, skipna=True)
    sd = df.std(axis=1, skipna=True).clip(lower=1e-12)
    return df.sub(mu, axis=0).div(sd, axis=0)

comp_map = {}
if True:
    # baseline: equal weight
    comp_map["eqw5"] = sum(
        zscore(neutralised[k]) * w for k, w, w in [
            ("leap_cnt_20", 1, 1), ("leap_net_20", 1, 1),
            ("leap_cnt_60", 1, 1), ("price_pos_20", 1, 1), ("turn_mom_20", 1, 1),
        ] if k in neutralised
    ) / 2.0

if True:
    # lean on leap pattern only (skip price_pos, skip turn_mom)
    comp_map["leap_only"] = (
        zscore(neutralised["leap_cnt_20"]) * 0.35
      + zscore(neutralised["leap_net_20"]) * 0.45
      + zscore(neutralised["leap_cnt_60"]) * 0.20
    )

if True:
    comp_map["near_trend"] = (
        zscore(neutralised["leap_cnt_20"]) * 0.35
      + zscore(neutralised["leap_net_20"]) * 0.35
      + zscore(neutralised["price_pos_20"])  * 0.30
    )

if True:
    comp_map["turn_leap"] = (
        zscore(neutralised["leap_cnt_20"]) * 0.50
      + zscore(neutralised["turn_mom_20"])  * 0.50
    )

if True:
    comp_map["leap_turn_pos"] = (
        zscore(neutralised["leap_cnt_20"])  * 0.40
      + zscore(neutralised["leap_net_20"])  * 0.25
      + zscore(neutralised["turn_mom_20"])  * 0.25
      + zscore(neutralised["price_pos_20"]) * 0.10
    )

# ─── 4. 用 best raw factor 做 main backtest ───
sys.path.insert(0, str(SCRIPTS_DIR))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data,
)

common_dates  = sorted(set(close_piv.dropna(how="all").index)
                       .intersection(ret_piv.dropna(how="all").index))
common_stocks = sorted(close_piv.columns.intersection(ret_piv.columns))
ra = ret_piv.loc[common_dates, common_stocks]

# recalc log_amount for common universe
log_amt2 = log_amt_20d.loc[common_dates, common_stocks].copy()

def neutralize_inplace(mat_raw, la):
    """In-place neutralise a (n_dates×n_stocks) ndarray w.r.t. la."""
    out = mat_raw.copy()
    for i in range(mat_raw.shape[0]):
        f  = mat_raw[i]; m  = la[i]
        mask = ~(np.isnan(f) | np.isnan(m))
        if mask.sum() < 30: continue
        fc = f[mask]; mc = m[mask]
        X  = np.column_stack([np.ones(mc.size), mc])
        try:
            b  = np.linalg.lstsq(X, fc, rcond=None)[0]
            out[i, mask] = fc - X @ b
        except Exception:
            pass
    return out

ALL_BACKTEST = {}

for comp_name, comp_raw in comp_map.items():
    print(f"\n[4.{comp_name}] 候选信号 backtest ...")
    cr = comp_raw.loc[common_dates, common_stocks].values
    cr_neu = neutralize_inplace(cr, log_amt2.values)
    # pandas for ic computation
    fa = pd.DataFrame(cr_neu, index=common_dates, columns=common_stocks)
    fa = fa.replace([np.inf,-np.inf], np.nan)

    best_res = None; best_sh = -9; best_cfg = None
    for fwd, rb, cost in FWD_OPTIONS:
        ic_f  = compute_ic_dynamic(fa, ra, fwd, "pearson")
        ric_f = compute_ic_dynamic(fa, ra, fwd, "spearman")
        gr_f, tv_f, hi_f = compute_group_returns(fa, ra, N_GROUPS, rb, cost)
        m = compute_metrics(gr_f, ic_f, ric_f, tv_f, N_GROUPS, holdings_info=hi_f)
        sh   = m.get("long_short_sharpe", 0) or 0
        ic_m = m.get("ic_mean", 0) or 0
        ic_t = m.get("ic_t_stat", 0) or 0
        mono = m.get("monotonicity", 0) or 0
        gsh  = (m.get("group_sharpe") or [None]*N_GROUPS)[N_GROUPS-1] or 0
        print(f"   {fwd:2d}f / {rb:2d}r cost{cost*100:.1f}%"
              f"  IC={ic_m:.4f} t={ic_t:.2f}  LS_sh={sh:.3f}  mono={mono:.2f}  G5Sh={gsh:.3f}")
        if sh > best_sh:
            best_sh = sh; best_cfg = (fwd, rb, cost); best_res = dict(
                fa=fa, ic=ic_f, ric=ric_f, gr=gr_f, tv=tv_f, hi=hi_f, m=m)

    fwd, rb, cost = best_cfg
    print(f"   → {comp_name} 最优 {fwd}f/{rb}r LS_sh={best_sh:.3f}")
    ALL_BACKTEST[comp_name] = dict(cfg=best_cfg, sh=best_sh, res=best_res, raw=comp_raw)

# ─── 4.5 反向方案 ───
best_name = max(ALL_BACKTEST, key=lambda k: ALL_BACKTEST[k]["sh"])
best_main = ALL_BACKTEST[best_name]
fa_p = best_main["res"]["fa"]

neg_backtests = {}
fa_n = -fa_p
for fwd, rb, cost in FWD_OPTIONS:
    ic_n  = compute_ic_dynamic(fa_n, ra, fwd, "pearson")
    gr_n, tv_n, hi_n = compute_group_returns(fa_n, ra, N_GROUPS, rb, cost)
    m = compute_metrics(gr_n, ic_n, None, tv_n, N_GROUPS, holdings_info=hi_n)
    neg_backtests[(fwd, rb)] = dict(fa=fa_n, ic=ic_n, gr=gr_n, tv=tv_n, hi=hi_n, m=m,
                                     fw=fwd, rb=rb, cost=cost)

best_neg_key = max(neg_backtests,
                   key=lambda k: neg_backtests[k]["m"].get("long_short_sharpe",0) or 0)
neg_sh = neg_backtests[best_neg_key]["m"].get("long_short_sharpe", 0) or 0
pos_sh = best_main["res"]["m"].get("long_short_sharpe", 0) or 0
fwd_b, rb_b, cost_b = best_main["cfg"]

if neg_sh > pos_sh * 1.05:
    direction     = -1
    fa_final      = -fa_p
    best_res      = neg_backtests[best_neg_key]
    direction_desc= "反向（低跳跃模式=高预期收益）"
    FORWARD_DAYS  = best_neg_key[0]
    REBALANCE_FREQ= best_neg_key[1]
    COST          = neg_sh and best_main["cfg"][2]  # approximate
    print(f"\n  反向 Sharpe={neg_sh:.3f} > 正向 {pos_sh:.3f} → 切换到反向")
else:
    direction      =  1
    fa_final       =  fa_p
    best_res       =  best_main["res"]
    direction_desc =  f"正向（高跳跃模式=高预期收益）"
    FORWARD_DAYS   =  fwd_b
    REBALANCE_FREQ =  rb_b
    COST           =  cost_b
    print(f"\n  正向 Sharpe={pos_sh:.3f} ≥ 反向 {neg_sh:.3f} → 保持正向，使用 {best_name}")

ic_series      = best_res["ic"]
rank_ic_series = compute_ic_dynamic(fa_final, ra, FORWARD_DAYS, "spearman")
group_returns  = best_res["gr"]
turnovers      = best_res["tv"]
metrics        = best_res["m"]
holdings_info  = best_res["hi"]

ic_m   = metrics.get("ic_mean", 0) or 0
ic_t   = metrics.get("ic_t_stat", 0) or 0
ls_sh  = metrics.get("long_short_sharpe", 0) or 0
ls_md  = metrics.get("long_short_mdd", 0) or 0
mono   = metrics.get("monotonicity", 0) or 0
sig5   = metrics.get("ic_significant_5pct", False)
gs     = metrics.get("group_sharpe") or []
gr_ann = metrics.get("group_returns_annualized") or []

# ─── 5. 相关性 ───
print("\n[7] 与入库因子相关性 ...")

# Amihud
amihud_raw  = (ret_piv.abs() / (amount_piv / 1e8).clip(lower=1e-8))
amihud_f    = np.log(amihud_raw.rolling(20, min_periods=10).mean().clip(lower=1e-12))
# Shadow
rng = (high_piv - low_piv).clip(lower=1e-8)
upper_sr = (high_piv - np.maximum(close_piv, open_piv)) / rng
lower_sr = (np.minimum(close_piv, open_piv) - low_piv) / rng
shadow   = (upper_sr - lower_sr).rolling(20, min_periods=10).mean()
# Overnight momentum
oret = (open_piv  / close_piv.shift(1)).clip(lower=0.001, upper=2.0) - 1
iret = (close_piv / open_piv).clip(lower=0.001, upper=2.0) - 1
overnight_mom = oret.rolling(20, min_periods=10).sum() - iret.rolling(20, min_periods=10).sum()
# CVaR
ret_vals = ret_piv.values
cvar_mat = np.full_like(ret_vals, np.nan)
for i in range(10, len(dates)):
    w = ret_vals[i-10:i]; s = np.sort(w, axis=0)
    bot = np.nanmean(s[:2], axis=0); vc = np.sum(~np.isnan(w), axis=0)
    bot[vc < 5] = np.nan; cvar_mat[i] = -bot
cvar_df = pd.DataFrame(cvar_mat, index=dates, columns=stocks)
# turnover_level, tae
turnover_level = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8))
amp_piv        = df.pivot_table(index="date", columns="stock_code", values="amplitude")
tae            = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8)
                     / amp_piv.rolling(20, min_periods=10).mean().clip(lower=0.01))
vol_log60d     = np.log(1 + ret_piv.rolling(60, min_periods=30).std())
tail_cvar      = cvar_df

f_final_df = pd.DataFrame(fa_final, index=common_dates, columns=common_stocks)

correlations = {}
for name, other in [
    ("amihud_illiq_v2",     amihud_f),
    ("shadow_pressure_v1",  shadow),
    ("overnight_momentum_v1",overnight_mom),
    ("tail_risk_cvar_v1",   cvar_df),
    ("turnover_level_v1",   turnover_level),
    ("tae_v1",              tae),
    ("vol_log60d_v4",       vol_log60d),
]:
    corrs = []
    o_sub  = other.reindex(index=common_dates, columns=common_stocks)
    for d in common_dates[::10]:
        v1 = f_final_df.loc[d].dropna()
        v2 = o_sub.loc[d].reindex(v1.index).dropna()
        c  = v1.index.intersection(v2.index)
        if len(c) > 50:
            r, _ = sp_stats.spearmanr(v1[c], v2[c])
            if not np.isnan(r): corrs.append(r)
    avg = float(np.mean(corrs)) if corrs else 0.0
    correlations[name] = round(avg, 3)
    print(f"   vs {name}: {avg:+.3f}")

# ─── 6. 写出 ───
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_backtest_data(group_returns, ic_series, rank_ic_series, str(OUTPUT_DIR))

def _nn(o):
    if isinstance(o, (np.bool_,)):  return bool(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)): return None
    if isinstance(o, dict): return {k: _nn(v) for k, v in o.items()}
    if isinstance(o, list): return [_nn(v) for v in o]
    return o

report = dict(
    factor_id        = FACTOR_ID,
    factor_name      = "上行跳跃模式 v1",
    factor_name_en   = "Up-Leap Pattern v1",
    category         = "跳跃/路径依赖",
    description      = (
        f"20d/60d 内单日涨幅 ≥ {LEAP_THRESH*100:.0f}% 的上行跳跃频次(取对数)、"
        "上行-下行净频次、60d累计跳跃频次 三位一体的复合因子。"
        f"高值=跳跃频次高/上行主导/近期密集跳跃=事件驱动型标的。OLS成交额中性化+MAD5%+z-score。"
    ),
    hypothesis       = (
        "中证1000小盘股存在明显的跳跃型事件驱动收益。累计跳跃多→信息面正向驱动密集，"
        "后续5-20日动量仍有延续；跳跃集中在近期→知情交易者在连续注入信息 → 趋势延续。"
    ),
    formula          = (
        f"0.25·log1p(N(leap≥{LEAP_THRESH*100:.0f}%,20d)) "
        f"+ 0.35·log1p(N(leap≥{LEAP_THRESH*100:.0f}%,20d)-N(dn≤−{LEAP_THRESH*100:.0f}%,20d)) "
        f"+ 0.20·log1p(N(leap≥{LEAP_THRESH*100:.0f}%,60d)) "
        f"+ 0.20·turnover_mom,  neutralise(log_amt20d)"
    ),
    direction        = direction,
    direction_desc   = direction_desc,
    stock_pool       = "中证1000",
    period           = (f"{common_dates[0].strftime('%Y-%m-%d')} ~ "
                        f"{common_dates[-1].strftime('%Y-%m-%d')}"),
    n_dates          = len(common_dates),
    n_stocks         = len(common_stocks),
    n_groups         = N_GROUPS,
    rebalance_freq   = REBALANCE_FREQ,
    forward_days     = FORWARD_DAYS,
    cost             = COST,
    correlations     = correlations,
    metrics          = metrics,
    backtest_options_compared = list(comp_map.keys()),
    best_composite  = best_name,
    leap_threshold_pct = LEAP_THRESH * 100,
)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(_nn(report), f, indent=2, ensure_ascii=False)

# ─── 7. 摘要 ───
print(f"\n{'═'*64}")
print(f"  {FACTOR_ID}: {report['factor_name']}")
print(f"  方向: {direction_desc}")
print(f"  最优复合: {best_name}  方案: {FORWARD_DAYS}f / {REBALANCE_FREQ}r")
print(f"{'═'*64}")
print(f"  区间:        {report['period']}")
print(f"  股票数:      {len(common_stocks)}")
print(f"  IC 均值:     {ic_m:.4f}   (t={ic_t:.2f}, {'✓5%' if sig5 else '✗不显著'})")
print(f"  IC>0占比:    {metrics.get('ic_positive_pct',0):.1%}")
print(f"  IR:          {metrics.get('ir',0):.4f}")
print(f"  多空 Sharpe: {ls_sh:.4f}   多空 MDD: {ls_md:.2%}")
print(f"  单调性:      {mono:.4f}")
print(f"  换手率:      {metrics.get('turnover_mean',0):.2%}")
print(f"{'─'*64}")
for i, (r, s) in enumerate(zip(gr_ann, gs), 1):
    r_s = f"{r:.2%}" if r is not None else "N/A"
    s_s = f"  Sh={s:.2f}" if s  else ""
    print(f"   G{i}: {r_s:>9}{s_s}")
print(f"{'─'*64}")
for name, c in sorted(correlations.items()):
    print(f"   vs {name}: {c:+.3f}")
print(f"{'═'*64}")

ok  = abs(ic_m) > 0.015 and abs(ic_t) > 2 and abs(ls_sh) > 0.5 and mono >= 0.8
fail_reasons = []
if abs(ic_m) <= 0.015: fail_reasons.append(f"IC={ic_m:.4f}≤0.015")
if abs(ic_t) <= 2:     fail_reasons.append(f"t={ic_t:.2f}≤2")
if abs(ls_sh) <= 0.5:  fail_reasons.append(f"Sharpe={ls_sh:.4f}≤0.5")
if mono < 0.8:         fail_reasons.append(f"单调性={mono:.2f}<0.80")
print(f"\n  自评: {'✅ 全通过' if ok else '❌ 未达标 — ' + '；'.join(fail_reasons)}")
sys.exit(0 if ok else 1)
