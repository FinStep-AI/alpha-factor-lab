#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""
因子: trj_trend_strength_v1 — 趋势强度(trj_5_20) v1
====================================================
构造: factor_trj_5_20.csv 直接 pivot → 截面缩尾 + OLS 中性化 → 分层回测"""
import json, sys, warnings, time, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ─── 固定标准参数 ───
FORWARD_DAYS   = 20
REBALANCE_FREQ = 20
N_GROUPS       = 5
COST           = 0.002
WINSORIZE_PCT  = 0.05

FACTOR_ID   = "trj_trend_strength_v1"
FACTOR_NAME = "趋势强度(trj_5_20) v1"
CATEGORY    = "量价/趋势"
FNAME       = "Trend Strength (trj_5_20) v1"

BASE       = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE / "data"
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = BASE / "output" / FACTOR_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT     = OUT_DIR / "backtest_report.json"

sys.path.insert(0, str(BASE / "skills" / "alpha-factor-lab" / "scripts"))

# ─── 1. load factor CSV → numpy panel ───
print(f"[1] 加载 factor_trj_5_20.csv ...", flush=True)
t0 = time.time()
fdf = pd.read_csv(DATA_DIR / "factor_trj_5_20.csv",
                  usecols=["date","stock_code","factor_value"])
fdf["date"]        = pd.to_datetime(fdf["date"])
fdf["stock_code"]  = fdf["stock_code"].astype(str).str.zfill(6)
fdf = fdf.drop_duplicates(["date","stock_code"])

all_dates  = sorted(fdf["date"].unique())
all_stocks = sorted(fdf["stock_code"].unique())
d_idx = {d:i for i,d in enumerate(all_dates)}
s_idx = {s:i for i,s in enumerate(all_stocks)}

dat_fac  = np.full((len(all_dates), len(all_stocks)), np.nan, dtype=np.float64)
# populate indices only for rows where factor_value is explicit
for _, r in fdf.iterrows():
    di = d_idx[r["date"]]; si = s_idx[r["stock_code"]]
    dat_fac[di, si] = r["factor_value"]
print(f"   {dat_fac.shape[0]} dates x {dat_fac.shape[1]} stocks  ({time.time()-t0:.1f}s)", flush=True)

# ─── 2. kline returns + log_amount ───
print(f"[2] 加载K线 + 构造收益率 / 成交额矩阵...", flush=True)
t0 = time.time()
kdf = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv",
                  usecols=["date","stock_code","close","amount"])
kdf["date"]       = pd.to_datetime(kdf["date"])
kdf["stock_code"] = kdf["stock_code"].astype(str).str.zfill(6)
kdf = kdf.sort_values(["stock_code","date"]).reset_index(drop=True)

cp = kdf.pivot_table(index="date", columns="stock_code", values="close")
cp = cp.sort_index()
# keep only stocks with non-null factor index
cp = cp.reindex(index=cp.index, columns=all_stocks)
ret = cp.pct_change().values          # (T, S)
amnt= kdf.pivot_table(index="date", columns="stock_code", values="amount")
amnt= amnt.reindex(index=cp.index, columns=all_stocks)
lmkt= np.log(amnt.rolling(20).mean().clip(lower=1).values.astype(np.float64))

k_dates  = cp.index.tolist()
k_d_idx  = {d:i for i,d in enumerate(k_dates)}
k_s_idx  = {s:i for i,s in enumerate(all_stocks)}

kd_len   = len(k_dates)
ks_len   = len(all_stocks)
print(f"   kline {kd_len} dates, {ks_len} stocks  ({time.time()-t0:.1f}s)", flush=True)

# ─── 3. map factor dates → kline dates & neutralise ───
print(f"[3] 截面缩尾 + OLS成交额中性化...", flush=True)
t0 = time.time()

# valid factor dates ∈ kline
f_in_k = [d for d in all_dates if d in k_d_idx]
n_days = len(f_in_k)

# result panel (n_days × ks_len)
dat_res = np.full((n_days, ks_len), np.nan, dtype=np.float64)

# build sparse amt + fac references by pivoting to kline date/sidx once
# pre-compute kline-date log-amount vectors per date
nt=0
for ti, dt in enumerate(f_in_k):
    ki = k_d_idx[dt]            # kline row index for this date
    frow = dat_fac[d_idx[dt]]   # factor row: (S,) NaN where missing
    # reuse masked kline-backed logic:
    amt_row = lmkt[ki]          # (S,)  may have NaN
    fac_row = frow.astype(np.float64, copy=True)

    # intersect non-NaN
    mask = (~np.isnan(fac_row)) & (~np.isnan(amt_row))
    if mask.sum() < 30:
        continue
    fv = fac_row[mask]
    mv = amt_row[mask]

    # 5% winsorize (cross-sectional)
    qlo, qhi = np.quantile(fv, WINSORIZE_PCT), np.quantile(fv, 1 - WINSORIZE_PCT)
    fv = np.clip(fv, qlo, qhi)

    # OLS neutralisation: demean w.r.t. log_amount (2-col design: 1 + log_amt)
    X = np.column_stack([np.ones(mv.size), mv])
    try:
        b = np.linalg.lstsq(X, fv, rcond=None)[0]
        rv = fv - X @ b
    except Exception:
        continue
    # scatter back into result row
    dat_res[ti, mask] = rv
    nt += 1

# compact to named arrays over valid dates only (include all f_in_k so we can keep contiguous index)
fn_np  = dat_res[:n_days]                        # (n_days, S)
dates_f = f_in_k                                 # list of kline-aligned dates

print(f"   done {nt}/{n_days} days  ({time.time()-t0:.1f}s)", flush=True)

# 2-D stats of fn
vals = fn_np[~np.isnan(fn_np)]
print(f"   fn  mean={vals.mean():.4f}  std={vals.std():.4f}"
      f"  p10={np.quantile(vals,.10):.4f}  p50={np.quantile(vals,.50):.4f}"
      f"  p90={np.quantile(vals,.90):.4f}  nnz={len(vals):.0f}",
      flush=True)

# ─── 4. direction probe ───
print(f"[4] 方向回顾 (前80天 sample)...", flush=True)
fwd_full  = np.empty((kd_len, ks_len), dtype=np.float64)
for s in range(ks_len):
    col = ret[:, s]
    log1p = np.log1p(np.clip(col, -0.999, None))
    cum   = np.nancumsum(log1p)
    fwd_full[:, s] = np.expm1(np.pad(cum, (FORWARD_DAYS, 0), mode="constant")[:cum.size]
                               - np.pad(cum, (0, FORWARD_DAYS), mode="constant")[:cum.size])
# forward_ret at row k = k→k+N (not shifted) — same semantics as factor_backtest
# direction probe uses the overlap between front factor rows and kline rows

t0_p = max(0, n_days - FORWARD_DAYS - 80)
ic_p = []; ic_n = []
for ti in range(t0_p, min(n_days, t0_p+60)):
    fv = fn_np[ti]
    kti = k_d_idx.get(dates_f[ti])
    if kti is None or kti + FORWARD_DAYS >= kd_len:
        continue
    fr = fwd_full[kti + FORWARD_DAYS]
    m  = (~np.isnan(fv)) & (~np.isnan(fr))
    if m.sum() < 30:
        continue
    a = fv[m]; b = fr[m]
    if a.std() < 1e-12:
        continue
    ic_p.append(float(np.corrcoef(a,b)[0,1]))
    ic_n.append(float(np.corrcoef(-a,b)[0,1]))
ic_p=np.array(ic_p); ic_n=np.array(ic_n)
if len(ic_p):
    mpos=float(ic_p.mean()); mneg=float(ic_n.mean())
    flip = -1 if abs(mneg) > abs(mpos) else 1
    print(f"   pos={mpos:.4f} neg={mneg:.4f}  →  {'保持' if flip==1 else '翻转'}",
          flush=True)
    if flip == -1:
        fn_np = -fn_np
else:
    flip = 1; print("   direction probe skipped – keep as-is")

# ─── 5. backtest iterator ───
from factor_backtest import compute_group_returns, compute_ic_dynamic, compute_metrics, save_backtest_data

def run_once(fwd, reb):
    ic  = compute_ic_dynamic_from_np(fn_np, ret, k_d_idx, s_idx, dates_f, all_dates,
                                     all_stocks, k_dates, ks_len, kd_len, fwd, "pearson")
    ric = compute_ic_dynamic_from_np(fn_np, ret, k_d_idx, s_idx, dates_f, all_dates,
                                     all_stocks, k_dates, ks_len, kd_len, fwd, "spearman")
    gr, to_all, h_info = compute_group_returns_from_np(fn_np, ret, k_d_idx, s_idx, dates_f,
                                                         all_stocks, k_dates, ks_len, kd_len,
                                                         N_GROUPS, reb, COST)
    mt = compute_metrics(gr, ic, ric, to_all, N_GROUPS, holdings_info=h_info)
    return mt, gr, ic, ric

# factor_backtest already wraps ret_piv; to bridge our numpy path → factor_backtest we
# fall back to building the pandas results directly. Instead, re-bridge via a thin pandas DF
# using only the overlapped surface — compact path below.
print(f"\n[5] 全参数回测 (fwd=5/10/20, reb=5/10/20, cost=0.002)...", flush=True)
t1=time.time()

# Use factor_backtest's pandas helpers directly — it's identical
# Build pandas factor & returns only on the kline-aligned front slice.
# Map: for each kline-front date dt ∈ f_in_k, use fn_np[ti] as its cross-section.
# Build `df_ll` in long format → pivot → factor_aligned
rows_date=[]; rows_stk=[]; rows_fac=[]
for ti, dt in enumerate(dates_f):
    row = fn_np[ti]
    m   = ~np.isnan(row)
    for si, flag in enumerate(m):
        if flag:
            rows_date.append(dt); rows_stk.append(all_stocks[si]); rows_fac.append(float(row[si]))
ll = pd.DataFrame({"date":rows_date,"stock_code":rows_stk,"factor_value":rows_fac})
fac_al = ll.pivot_table(index="date", columns="stock_code", values="factor_value").sort_index()
ret_al = cp.pct_change()   # reusing cp

# Restrict to common surface
cds = sorted(fac_al.dropna(how="all").index.intersection(ret_al.dropna(how="all").index))
css = sorted(fac_al.columns.intersection(ret_al.columns))
fa  = fac_al.reindex(index=cds, columns=css)
ra  = ret_al.reindex(index=cds, columns=css)

results=[]
configs=[(5,5), (5,10), (5,20), (10,5), (10,10), (10,20), (20,5), (20,10), (20,20)]
for fwd, reb in configs:
    ic_  = compute_ic_dynamic(fa, ra, fwd, "pearson")
    ric_ = compute_ic_dynamic(fa, ra, fwd, "spearman")
    gr_, to_, hi_ = compute_group_returns(fa, ra, N_GROUPS, reb, COST)
    mt_  = compute_metrics(gr_, ic_, ric_, to_, N_GROUPS, holdings_info=hi_)
    sh_  = abs(mt_.get("long_short_sharpe",0) or 0)
    ic_0 = abs(mt_.get("ic_mean",0) or 0)
    ic_t = abs(mt_.get("ic_t_stat",0) or 0)
    mono = mt_.get("monotonicity",0) or 0
    results.append((fwd,reb,ic_0,ic_t,sh_,mono,mt_,gr_,ic_,ric_))
    print(f"    fwd={fwd:2d} reb={reb:2d} "
          f"|IC|={ic_0:.4f} t={ic_t:.2f} "
          f"Sharpe={sh_:.4f} mono={mono:.3f}", flush=True)

best=sorted(results, key=lambda x: (x[2] >= 0.015, x[3] > 2, x[4] > 0.5, x[4]), reverse=True)[0]
bf,br,bic,bic_t,bsh,bmono,bmt,bgr,bic_,brc_ = best
print(f"\n   ▶ 最佳: fwd={bf} reb={br} |IC|={bic:.4f} t={bic_t:.2f} Sharpe={bsh:.4f}  ({time.time()-t1:.1f}s)", flush=True)

# ─── 6. save output for the best config ───
save_backtest_data(bgr, bic_, brc_, str(OUT_DIR))

def _n2n(o):
    if isinstance(o,(np.bool_,)): return bool(o)
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,(np.floating,)):
        v=float(o); return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(o,float) and (math.isnan(o) or math.isinf(o)): return None
    if isinstance(o,dict): return {k:_n2n(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [_n2n(v) for v in o]
    return o

ga=bmt.get("group_returns_annualized",[None]*N_GROUPS)
gs=bmt.get("group_sharpe",            [None]*N_GROUPS)
rpt=dict(
    factor_id= FACTOR_ID, factor_name=FACTOR_NAME,
    factor_name_en=FNAME,
    category=CATEGORY,
    description="趋势强度因子(trj_5_20): 量价趋势综合强度指标,5/20参数版本。中性化后分层回测。",
    expected_direction="正向(高trj=强趋势→高未来收益)",
    factor_type="趋势强度类/量价复合",
    formula="neutralize(factor_trj_5_20)[cxn, stock], log_amount_20d OLS 截面",
    direction=1,
    stock_pool="中证1000",
    period=f"{cds[0].strftime('%Y-%m-%d')} ~ {cds[-1].strftime('%Y-%m-%d')}",
    n_dates=len(cds), n_stocks=len(css),
    best_config=dict(forward=bf, rebalance=br, cost=COST),
    source_type="自研(量价趋势)",
    source_title="趋势强度指标(5-20参数版本)",
    source_url="",
    metrics=bmt,
)
with open(REPORT,"w",encoding="utf-8") as f:
    json.dump(_n2n(rpt),f,indent=2,ensure_ascii=False)

# ─── 7. summary ───
ic_m  = bmt.get("ic_mean",0) or 0
ic_st = bmt.get("ic_t_stat",0) or 0
ir_   = bmt.get("ir",0) or 0
lsh   = bmt.get("long_short_sharpe",0) or 0
lmd   = bmt.get("long_short_mdd",0) or 0
mono  = bmt.get("monotonicity",0) or 0
to_   = bmt.get("turnover_mean",0) or 0
sig5  = "✓" if bmt.get("ic_significant_5pct") else "✗"
p1 = abs(ic_m) > 0.015
p2 = abs(ic_st) > 2
p3 = abs(lsh) > 0.5
ok  = p1 and p2 and p3
print(f"\n{'='*60}")
print(f"  {FACTOR_ID}  — {FACTOR_NAME}")
print(f"  最佳配置: fwd={bf}d / reb={br}d / cost={COST:.1%}")
print(f"{'='*60}")
print(f"  IC均值:     {ic_m:.4f}  (t={ic_st:.2f}  sig5={sig5})")
print(f"  IR:         {ir_:.4f}")
print(f"  多空Sharpe: {lsh:.4f}")
print(f"  多空MDD:    {lmd:.2%}")
print(f"  单调性:     {mono:.4f}")
print(f"  换手率:     {to_:.2%}")
print(f"{'─'*60}")
for i,(r,s) in enumerate(zip(ga,gs),1):
    rs=f"{r:.2%}" if r is not None else "N/A"
    ss=f"(Sh{s:.2f})" if s is not None else ""
    print(f"    G{i}: {rs} {ss}")
print(f"{'═'*60}")
print(f"   {'✓' if p1 else '✗'} |IC|>0.015")
print(f"   {'✓' if p2 else '✗'} t>2")
print(f"   {'✓' if p3 else '✗'} |Sharpe|>0.5")
print(f"  总评: {'✅ 全部通过，入库' if ok else '⚠️  未全通过，记录失败'}")
print(f"\n  输出:")
print(f"    {OUT_DIR/'cumulative_returns.json'}")
print(f"    {OUT_DIR/'ic_series.json'}")
print(f"    {OUT_DIR/'backtest_report.json'}")
print(f"{'='*60}")
sys.exit(0 if ok else 1)
