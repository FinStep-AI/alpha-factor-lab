#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""
因子: intraday_momentum_v1 — 日内动量方向 v1
================================================
构造: 预计算 factor_intraday_mom_v1.csv 直接加载 pivot
正向使用: 第T日 factor 用作 T 截面排序信号，前瞻 N 日

逻辑 (回溯):
  - factor_intraday_mom_v1 = 0 前后翻转后的方向
  - 原始构造: intraday_ret 的 40d 累计相对于 slow MA(120d) 的穿越信号
  - 值越高 → 日内动量越强 → 正向使用做多强日内动量股

标准: |IC|>0.015 | t>2 | |Sharpe|>0.5  → 入库
"""
import json, sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ─────────── 参数 ───────────
FORWARD_DAYS   = 20
REBALANCE_FREQ = 20
N_GROUPS       = 5
COST           = 0.002
WINSORIZE_PCT  = 0.05
FACTOR_ID      = "intraday_momentum_v1"
FACTOR_NAME    = "日内动量方向 v1"
FACTOR_NAME_EN = "Intraday Momentun Direction v1"
CATEGORY       = "量价/日内效应"
DIRECTION      = 1       # 正向: 高值→高收益

BASE      = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE / "data"
SCRIPT_DIR= Path(__file__).resolve().parent
FACTOR_CSV  = DATA_DIR / "factor_intraday_mom_v1.csv"
KLINE_CSV   = DATA_DIR / "csi1000_kline_raw.csv"
OUTPUT_DIR  = BASE / "output" / FACTOR_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

sys.path.insert(0, str(BASE / "skills" / "alpha-factor-lab" / "scripts"))

# ─────────── 1. 加载预计算因子 ───────────
print(f"[1] 加载预计算因子 {FACTOR_CSV.name}...", flush=True)
t0 = time.time()
fdf = pd.read_csv(FACTOR_CSV, usecols=["date","stock_code","factor_value"])
fdf["date"] = pd.to_datetime(fdf["date"])
fdf["stock_code"] = fdf["stock_code"].astype(str).str.zfill(6)
fdf = fdf.drop_duplicates(["date","stock_code"])

# pivot: index=date, columns=stock_code
factor_mat = fdf.pivot_table(index="date", columns="stock_code", values="factor_value")
factor_mat = factor_mat.sort_index()
factor_date_min = factor_mat.index.min()
factor_date_max = factor_mat.index.max()
print(f"   因子范围 {factor_date_min.date()} ~ {factor_date_max.date()}, {factor_mat.shape}",
      flush=True)

# ─────────── 2. 加载收益率 ───────────
print(f"[2] 加载K线 {KLINE_CSV.name}...", flush=True)
t0 = time.time()
kdf = pd.read_csv(KLINE_CSV, usecols=["date","stock_code","close","amount"])
kdf["date"] = pd.to_datetime(kdf["date"])
kdf["stock_code"] = kdf["stock_code"].astype(str).str.zfill(6)
kdf = kdf.sort_values(["stock_code","date"]).reset_index(drop=True)
close_piv = kdf.pivot_table(index="date", columns="stock_code", values="close")
ret_piv   = close_piv.pct_change()
amount_piv = kdf.pivot_table(index="date", columns="stock_code", values="amount")
log_mktcap = np.log(amount_piv.rolling(20).mean().clip(lower=1))
print(f"   加载完成 {time.time()-t0:.1f}s, {close_piv.shape[0]} 日 × {close_piv.shape[1]} 股",
      flush=True)

dates_all = sorted(close_piv.index)
stocks_all = sorted(close_piv.columns.intersection(factor_mat.columns))

# ─────────── 3. 对齐因子: 用 T-1 因子值做截面排序 ───────────
# 取因子文件有值 且 kline 同时有数据的日期
valid_dates = sorted(factor_mat.index.intersection(close_piv.index))
n_days = len(valid_dates)
n_stocks = len(stocks_all)
print(f"[3] 对齐: {n_days} 日 × {n_stocks} 股", flush=True)

factor_panel = pd.DataFrame(np.nan, index=valid_dates, columns=stocks_all, dtype=float)
amt_panel    = pd.DataFrame(np.nan, index=valid_dates, columns=stocks_all, dtype=float)

for dt in valid_dates:
    row = factor_mat.loc[dt, stocks_all].dropna()
    factor_panel.loc[dt, row.index] = row.values
    arow = log_mktcap.loc[dt, stocks_all].dropna()
    amt_panel.loc[dt, arow.index] = arow.values

non_null_rate = factor_panel.notna().mean().mean()
print(f"   因子非空率: {non_null_rate:.2%}", flush=True)

# ─────────── 4. 截面缩尾 + 成交额中性化 ───────────
print(f"[4] 截面 5% 缩尾 + 成交额 OLS 中性化...", flush=True)
factor_neutral = pd.DataFrame(np.nan, index=valid_dates, columns=stocks_all, dtype=float)
n_done = 0
for dt in valid_dates:
    f  = factor_panel.loc[dt].dropna()
    m  = amt_panel.loc[dt].reindex(f.index).dropna()
    idx = f.index.intersection(m.index)
    if len(idx) < 30:
        continue
    fv = f[idx].values.astype(float)
    mv = m[idx].values.astype(float)

    # 5% 缩尾
    qlo, qhi = np.quantile(fv, WINSORIZE_PCT), np.quantile(fv, 1-WINSORIZE_PCT)
    fv = np.clip(fv, qlo, qhi)

    # OLS 中性化
    X = np.column_stack([np.ones(len(mv)), mv])
    try:
        b = np.linalg.lstsq(X, fv, rcond=None)[0]
        resid = fv - X @ b
        factor_neutral.loc[dt, idx] = resid
        n_done += 1
    except Exception:
        pass

print(f"   完成 {n_done} / {n_days} 天", flush=True)
all_vals = factor_neutral.stack().dropna()
print(f"   残差 mean={all_vals.mean():.4f} std={all_vals.std():.4f}"
      f" p10={all_vals.quantile(.1):.4f} p50={all_vals.quantile(.5):.4f} p90={all_vals.quantile(.9):.4f}",
      flush=True)

# ─────────── 5. 方向回顾检验 ───────────
print(f"[5] 方向回顾...", flush=True)
t_warm = 40 + FORWARD_DAYS
samples = min(60, n_days - t_warm)
fwd_ret_full = ret_piv.shift(-FORWARD_DAYS)   # forward ret aligned to T

ic_pos = []; ic_neg = []
for k in range(samples):
    dt = valid_dates[t_warm + k]
    f  = factor_neutral.loc[dt].dropna()
    fr = fwd_ret_full.loc[dt, f.index].dropna()
    c  = f.index.intersection(fr.index)
    if len(c) < 30:
        continue
    ic_pos.append(  np.corrcoef( f[c].values, fr[c].values)[0,1])
    ic_neg.append(  np.corrcoef(-f[c].values, fr[c].values)[0,1])

avg_pos = np.nanmean(ic_pos); avg_neg = np.nanmean(ic_neg)
use_dir = 1 if abs(avg_pos) >= abs(avg_neg) else -1
print(f"   方向回顾 IC 正向={avg_pos:.4f} 反向={avg_neg:.4f}"
      f" → {'正向(现行=高值做多)' if use_dir==1 else '反向(翻转=高值做空)'}", flush=True)
if use_dir == -1:
    factor_neutral = -factor_neutral

# ─────────── 6. 回测 ───────────
print(f"[6] 分层回测: {N_GROUPS}组 / rebal {REBALANCE_FREQ}d / fwd {FORWARD_DAYS}d / cost {COST:.1%}...",
      flush=True)
t1 = time.time()

cd = sorted(factor_neutral.dropna(how="all").index
            .intersection(ret_piv.dropna(how="all").index))
cs = sorted(factor_neutral.columns.intersection(ret_piv.columns))
fa = factor_neutral.loc[cd, cs]
ra = ret_piv.loc[cd, cs]
print(f"   对齐 {len(cd)} 日 × {len(cs)} 股", flush=True)

from factor_backtest import compute_group_returns, compute_ic_dynamic, compute_metrics, save_backtest_data

ic_series   = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "pearson")
rank_ic_ser = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "spearman")
gr, to_all, hi = compute_group_returns(fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m = compute_metrics(gr, ic_series, rank_ic_ser, to_all, N_GROUPS, holdings_info=hi)
print(f"   回测完成 {time.time()-t1:.1f}s", flush=True)

# ─────────── 7. 输出 ───────────
print(f"[7] 写出输出...", flush=True)
save_backtest_data(gr, ic_series, rank_ic_ser, str(OUTPUT_DIR))

def _n2n(o):
    if isinstance(o,(np.bool_,)): return bool(o)
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,(np.floating,)):
        v=float(o); return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(o,float) and (np.isnan(o) or np.isinf(o)): return None
    if isinstance(o,dict): return {k:_n2n(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [_n2n(v) for v in o]
    return o

grp_ann = m.get("group_returns_annualized", [None]*N_GROUPS)
grp_sh  = m.get("group_sharpe",            [None]*N_GROUPS)

report = dict(
    factor_id       = FACTOR_ID,
    factor_name     = FACTOR_NAME,
    factor_name_en  = FACTOR_NAME_EN,
    category        = CATEGORY,
    description     = "日内动量方向因子: factor_intraday_mom_v1 直接 pivot 后截面中性化。"
                      "原始构造=40d累计日内动量穿越slow MA(120d)信号，已预标准化。",
    hypothesis      = "日内动量方向(放量日内方向性)与次日/中短期收益正相关，"
                      "信息在日内率先释放 → 后续收益延续。",
    expected_direction = "正向（高值 = 强日内动量 = 高预期收益）",
    factor_type     = "量价/日内动量",
    formula         = "neutralize(factor_intraday_mom_v1[T][stock], log_amount_20d), T日截面排序",
    direction       = use_dir,
    stock_pool      = "中证1000",
    period          = f"{cd[0].strftime('%Y-%m-%d')} ~ {cd[-1].strftime('%Y-%m-%d')}",
    n_dates         = len(cd),
    n_stocks        = len(cs),
    factor_files    = ["data/factor_intraday_mom_v1.csv"],
    source_type     = "自研(日内价量)",
    source_title    = "日内动量方向因子 — 预计算序列 pivot",
    source_url      = "",
    forward_days    = FORWARD_DAYS,
    rebalance_freq  = REBALANCE_FREQ,
    cost            = COST,
    barra_style     = "MICRO",
    metrics         = m,
)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(_n2n(report), f, indent=2, ensure_ascii=False)

# ─────────── 8. 打印摘要 ───────────
IC  = m.get("ic_mean",0) or 0
ICT = abs(m.get("ic_t_stat",0) or 0)
IR  = m.get("ir",0) or 0
LSH = m.get("long_short_sharpe",0) or 0
LSMD= m.get("long_short_mdd",0) or 0
MONO= m.get("monotonicity",0) or 0
TO  = m.get("turnover_mean",0) or 0
SIG = "✓" if m.get("ic_significant_5pct") else "✗"
pass_ic  = abs(IC)  > 0.015
pass_t   = ICT      > 2.0
pass_sh  = abs(LSH) > 0.5
all_pass = pass_ic and pass_t and pass_sh

print(f"\n{'='*60}")
print(f"  intraday_momentum_v1  回测结果")
print(f"{'='*60}")
print(f"  区间:       {cd[0].date()} ~ {cd[-1].date()}")
print(f"  方向回顾:   IC_pos={avg_pos:.4f}  IC_neg={avg_neg:.4f}  → {'保持' if use_dir==1 else '翻转'}")
print(f"  IC均值:     {IC:.4f}  (t={ICT:.2f}, 5%{SIG})")
print(f"  Rank IC:    {m.get('rank_ic_mean',0):.4f}")
print(f"  IR:         {IR:.4f}")
print(f"  换手率:     {TO:.2%}")
print(f"  多空Sharpe: {LSH:.4f}")
print(f"  多空MDD:    {LSMD:.2%}")
print(f"  单调性:     {MONO:.4f}")
print(f"{'─'*60}")
print(f"  分层年化收益:")
for i,(r,s) in enumerate(zip(grp_ann,grp_sh),1):
    r_s=f"{r:.2%}" if r is not None else "N/A"
    s_s=f"  (Sharpe {s:.2f})" if s is not None else ""
    print(f"    G{i}: {r_s}{s_s}")
print(f"{'═'*60}")
print(f"  达标准则  |IC|>0.015 {('✓' if pass_ic  else '✗')}"
      f"   t>2     {('✓' if pass_t   else '✗')}"
      f"   |Sharpe|>0.5 {('✓' if pass_sh else '✗')}")
print(f"  总评:     {'✅ 全面通过，因子有效' if all_pass else '⚠️  三项指标未全部通过，记录为失败'}")
print(f"\n  输出文件:")
print(f"    {OUTPUT_DIR/'cumulative_returns.json'}")
print(f"    {OUTPUT_DIR/'ic_series.json'}")
print(f"    {OUTPUT_DIR/'backtest_report.json'}")
print(f"{'='*60}")
sys.exit(0 if all_pass else 1)
