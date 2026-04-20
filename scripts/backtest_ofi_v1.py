#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: order_flow_intensity_v1 — 订单流强度 v1
====================================================

构造:
  1. sign(ret) × log(1+vol/MA60(vol)), 20日均值
  2. 成交额中性化(OLS) + 5.2MAD缩尾 + z-score
  3. 5日前瞻, 5日调仓, 5组分层

逻辑:
  OFI直接衡量订单方向性: sign(ret) × volume代表成交量与价格方向的一致性。
  使用 volume/MA60(volume) 消除绝对成交额大小影响, 聚焦相对量级信号。
  高OFI = 持续放量上涨/放量下跌(方向明确), 20日累积代表持续信息流强度。
  A股小盘股散户vs机构的订单流不平衡可产生alpha。

数据: 截至2026-04-15
"""

import json, sys, time, warnings
from pathlib import Path
from scipy import stats as sp_stats
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ────────────────── 参数 ──────────────────
VOL_WIN = 60; ROLL_WIN = 20; FW = 5; RF = 5; NGRP = 5; COST = 0.003
MAD_K = 5.2; DC = "2026-04-15"; FID = "order_flow_intensity_v1"

SD = Path(__file__).resolve().parent
DATA = SD.parent / "data" / "csi1000_kline_raw.csv"
OUT = SD.parent.parent / "output" / FID
REPORT = OUT / "backtest_report.json"
t0 = time.time()

# ────────────────── 数据加载 ──────────────────
print("[1] 加载...")
df = pd.read_csv(DATA); df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] <= DC].sort_values(["stock_code","date"]).reset_index(drop=True)

cp = df.pivot_table(index="date",columns="stock_code",values="close")
ap = df.pivot_table(index="date",columns="stock_code",values="amount")
rp = cp.pct_change()
la = np.log(ap.rolling(20).mean().clip(lower=1))

D = cp.index.tolist(); S = cp.columns.tolist()
print(f"   {len(D)}d x {len(S)} stocks")

# ────────────────── OFI构造 ──────────────────
print(f"[2] OFI构造 (sign(ret)*log(1+V/MA60V), {ROLL_WIN}d均值)...")
t1 = time.time()
df["ret"] = df.groupby("stock_code")["close"].pct_change()
df = df.dropna(subset=["ret"]).reset_index(drop=True)

df["mv"] = df.groupby("stock_code")["volume"].transform(
    lambda x: x.rolling(VOL_WIN, min_periods=VOL_WIN//2).mean().clip(lower=1))
df["vr"] = np.log1p(df["volume"] / df["mv"])
df["sig"] = np.where(df["ret"].notna() & df["vr"].notna(),
                      np.sign(df["ret"]) * df["vr"], np.nan)
df["fr"] = df.groupby("stock_code")["sig"].transform(
    lambda x: x.rolling(ROLL_WIN, min_periods=int(ROLL_WIN*0.75)).mean())

fr = df.pivot_table(index="date", columns="stock_code", values="fr")
# align
cd = sorted(fr.index.intersection(rp.index))
cs = sorted(fr.columns.intersection(rp.columns))
fr = fr.loc[cd, cs]; rp = rp.loc[cd, cs]; la = la.loc[cd, cs]
print(f"   非空率:{fr.notna().mean().mean():.1%} ({time.time()-t1:.1f}s)")

# ────────────────── MAD缩尾 ──────────────────
print("[3] MAD缩尾...")
t1 = time.time()
a = fr.values.copy()
mask = ~np.isnan(a)
for i in range(a.shape[0]):
    v = a[i,mask[i]]
    if len(v)<10: continue
    med=np.median(v); mad=np.median(np.abs(v-med))*1.4826
    if mad<1e-8: continue
    a[i,mask[i]] = np.clip(v, med-MAD_K*mad, med+MAD_K*mad)
fr = pd.DataFrame(a, index=cd, columns=cs)
print(f"   ({time.time()-t1:.1f}s)")

# ────────────────── 中性化 ──────────────────
print("[4] OLS中性化...")
t1 = time.time()
a = fr.values.copy(); la_arr = la.values; cm = (~np.isnan(a))&(~np.isnan(la_arr))&(np.abs(la_arr)>1e-8)
for i in range(a.shape[0]):
    m = cm[i]
    if m.sum()<30: continue
    f=a[i,m]; x=la_arr[i,m]; X=np.column_stack([np.ones(m.sum()),x])
    try: b=np.linalg.lstsq(X,f,rcond=None)[0]; a[i,m]=f-X@b
    except: pass
fn = pd.DataFrame(a, index=cd, columns=cs)
print(f"   ({time.time()-t1:.1f}s)")

# ────────────────── Z-score ──────────────────
fz = fn.sub(fn.mean(axis=1), axis=0).div(fn.std(axis=1), axis=0).clip(-3,3)
cd2 = sorted(fz.dropna(how="all").index.intersection(rp.dropna(how="all").index))
cs2 = sorted(fz.columns.intersection(rp.columns))
fa = fz.loc[cd2, cs2]; ra = rp.loc[cd2, cs2]
print(f"[5] 回测 ({NGRP}组/{RF}d/{COST*100:.0f}bps)...")

sys.path.insert(0,str(SD.parent / "skills" / "alpha-factor-lab" / "scripts"))
from factor_backtest import compute_group_returns, compute_ic_dynamic, compute_metrics, save_backtest_data

ic = compute_ic_dynamic(fa, ra, FW, "pearson")
ric = compute_ic_dynamic(fa, ra, FW, "spearman")
gr, tov, hi = compute_group_returns(fa, ra, NGRP, RF, COST)
mt = compute_metrics(gr, ic, ric, tov, NGRP, holdings_info=hi)
print(f"   ({time.time()-t1:.1f}s)")

# ────────────────── 相关性 ──────────────────
print("[6] 相关性...")
t1 = time.time()
hp = df.pivot_table(index="date",columns="stock_code",values="high",dropna=False)
lp = df.pivot_table(index="date",columns="stock_code",values="low",dropna=False)
op = df.pivot_table(index="date",columns="stock_code",values="open")
tp = df.pivot_table(index="date",columns="stock_code",values="turnover",dropna=False)
aamp = df.pivot_table(index="date",columns="stock_code",values="amplitude")

amihud_f = np.log((rp.abs()/(ap/1e8).clip(lower=1e-8)).rolling(20,min_periods=10).mean().clip(lower=1e-12))
usr = (hp-np.maximum(cp,op))/(hp-lp).clip(lower=1e-8); lsr=(np.minimum(cp,op)-lp)/(hp-lp).clip(lower=1e-8)
shadow_f = (usr-lsr).rolling(20,min_periods=10).mean()
oret=op/cp.shift(1)-1; iret=cp/op-1
overnight_f = oret.rolling(20,min_periods=10).sum()-iret.rolling(20,min_periods=10).sum()
cmat = np.full((len(cd2),len(cs2)),np.nan); rn = rp.values
for i in range(10,len(cd2)):
    w=rn[i-10:i]; s=np.sort(w,axis=0); b2=np.nanmean(s[:2],axis=0); vc=np.sum(~np.isnan(w),axis=0)
    b2[vc<5]=np.nan; cmat[i,:]=-b2
cvdf = pd.DataFrame(cmat,index=cd2,columns=cs2)
negf = (rp<=-0.03).astype(float).rolling(10,min_periods=5).mean()
turnover_l = np.log(tp.rolling(20,min_periods=10).mean().clip(lower=1e-8))
tae_r = np.log(tp.rolling(20,min_periods=10).mean().clip(lower=1e-8)/aamp.rolling(20,min_periods=10).mean().clip(lower=0.01))
amp_l = np.log(aamp.rolling(60,min_periods=30).mean().clip(lower=0.01))
# ma_disp
ms = np.stack([cp.rolling(w).mean().values/cp.values for w in [5,10,20,40,60,120]],axis=-1)
ma_disp_f = pd.DataFrame(np.nanstd(ms,axis=-1),index=cp.index,columns=cp.columns)
# pv_corr proxy
lad = np.log(ap.clip(lower=1)).diff().abs().rolling(20,min_periods=10).mean()
pvc = rp.rolling(20,min_periods=10).corr(lad)

fcorr = fa
sdates = cd2[::max(1,len(cd2)//20)]
corrs = {}
for name, other in [
    ('amihud_illiq_v2',amihud_f),('shadow_pressure_v1',shadow_f),('overnight_momentum_v1',overnight_f),
    ('tail_risk_cvar_v1',cvdf),('neg_day_freq_v1',negf),('turnover_level_v1',turnover_l),
    ('tae_v1',tae_r),('amp_level_v2',amp_l),('ma_disp_v1',ma_disp_f),('pv_corr_v1',pvc),
]:
    cr=[]
    for d in sdates:
        f1=fcorr.loc[d].dropna()
        f2=(other.loc[d] if isinstance(other,pd.DataFrame) else pd.Series(dtype=float)).reindex(f1.index).dropna() if isinstance(other,pd.DataFrame) else other
        c=f1.index.intersection(f2.index)
        if len(c)>50:
            r,_=sp_stats.spearmanr(f1[c],f2[c])
            if not np.isnan(r): cr.append(r)
    corrs[name]=round(float(np.mean(cr)),3) if cr else 0
    print(f"   vs {name}: {corrs[name]:.3f}")
corrs[FID]=1.0; print(f"   ({time.time()-t1:.1f}s)")

# ────────────────── 输出 ──────────────────
print("[7] 写入...")
OUT.mkdir(parents=True,exist_ok=True)
save_backtest_data(gr,ic,ric,str(OUT))
def nn(o):
    if isinstance(o,(np.bool_,)): return bool(o)
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,(np.floating,)): return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o,dict): return {k:nn(v) for k,v in o.items()}
    if isinstance(o,list): return [nn(v) for v in o]
    return o
rpt = {"factor_id":FID,"factor_name":"订单流强度 v1","factor_name_en":"Order Flow Intensity v1",
 "category":"量价/微观结构/情绪",
 "description":f"过去{ROLL_WIN}日 sign(ret)×log(1+vol/MA60(vol))均值,正向=买入力量强→高收益",
 "hypothesis":"A股小盘股订单流不平衡持续存在alpha。高买入OFI=机构主导,动量延续。",
 "formula":"neutralize(MA20(sign(ret)×log(1+vol/MA60(vol))), log_amount_20d)","direction":1,
 "direction_desc":"正向（高OFI=持续买入强=高预期收益）","stock_pool":"中证1000",
 "period":f"{cd2[0].strftime('%Y-%m-%d')} ~ {cd2[-1].strftime('%Y-%m-%d')}",
 "n_dates":len(cd2),"n_stocks":len(cs2),"n_groups":NGRP,"rebalance_freq":RF,
 "forward_days":FW,"cost":COST,"data_cutoff":DC,"correlations":corrs,"metrics":mt}
with open(REPORT,"w",encoding="utf-8") as f: json.dump(nn(rpt),f,indent=2,ensure_ascii=False)

# ────────────────── 摘要 ──────────────────
im=mt.get("ic_mean",0)or 0; it=mt.get("ic_t_stat",0)or 0; lss=mt.get("long_short_sharpe",0)or 0
lsm=mt.get("long_short_mdd",0)or 0; mo=mt.get("monotonicity",0)or 0
sg="✓" if mt.get("ic_significant_5pct") else "✗"
print(f"\n{'='*60}")
print(f"  {FID}: 订单流强度 v1")
print(f"{'='*60}")
print(f"  区间:      {rpt['period']}")
print(f"  总耗时:    {time.time()-t0:.0f}s")
print(f"  IC均值:    {im:.4f}  (t={it:.2f}, {sg})")
print(f"  IR:        {mt.get('ir',0):.4f}")
print(f"  多空Sharpe:{lss:.4f}  MDD:{lsm:.1%}")
print(f"  单调性:    {mo:.4f}  换手:{mt.get('turnover_mean',0):.1%}")
print(f"{'─'*60}")
for i,r in enumerate(mt.get("group_returns_annualized",[]),1):
    print(f"    G{i}: {r:.2%}" if r is not None else f"    G{i}: N/A")
print(f"{'='*60}")
valid=abs(im)>0.015 and abs(it)>2 and abs(lss)>0.5
print(f"  ➤ 因子{'有效 ✓' if valid else '无效 ✗'}")
