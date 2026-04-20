#!/usr/bin/env python3
"""amt_weighted_mom_v1 backtest - clean v2"""
import json,sys,time,warnings
from pathlib import Path
from scipy import stats as sp_stats
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

SD=Path(__file__).resolve().parent
sys.path.insert(0,str(SD.parent/"skills"/"alpha-factor-lab"/"scripts"))
from factor_backtest import compute_group_returns,compute_ic_dynamic,compute_metrics,save_backtest_data

FW,RF,NGRP,COST,WINSOR,DC=5,5,5,0.003,0.05,"2026-04-15"
FID="amt_weighted_mom_v1"
OUT=SD.parent.parent/"output"/FID
t0=time.time()

# Load data
print("[1] 加载...")
df=pd.read_csv(SD.parent/"data"/"csi1000_kline_raw.csv")
df["date"]=pd.to_datetime(df["date"]); df=df[df["date"]<=DC].sort_values(["stock_code","date"]).reset_index(drop=True)
df["sc"] = df["stock_code"].astype(str).str.zfill(6)
cp=df.pivot_table(index="date",columns="sc",values="close")
ap=df.pivot_table(index="date",columns="sc",values="amount")
rp=cp.pct_change()

# Load pre-computed factor
print("[2] 因子...")
fc=pd.read_csv(SD.parent/"data"/"factor_amt_weighted_mom_v1.csv",dtype={"stock_code":str})
fc["date"]=pd.to_datetime(fc["date"]); fc=fc[fc["date"]<=DC]
fr_raw=fc.pivot_table(index="date",columns="stock_code",values="factor",dropna=False)
cd=sorted(fr_raw.index.intersection(rp.index)); cn=sorted(fr_raw.columns.intersection(rp.columns))
fr=fr_raw.loc[cd,cn]; rp2=rp.loc[cd,cn]  # aligned
la=np.log(ap.loc[cd,cn].rolling(20).mean().clip(lower=1))
print(f"   {len(cd)}d × {len(cn)} stocks, non-NaN={fr.notna().mean().mean():.1%}")

# Pre-process: winsor + neutralize + zscore
print("[3] 预处理...")
av=fr.values.copy(); mk=~np.isnan(av)
for i in range(av.shape[0]):
    v=av[i,mk[i]]
    if len(v)>10: lo,hi=np.quantile(v,WINSOR),np.quantile(v,1-WINSOR); av[i,mk[i]]=np.clip(v,lo,hi)
cm=mk & (~np.isnan(la.values)) & (np.abs(la.values)>1e-8)
for i in range(av.shape[0]):
    rm=cm[i]
    if rm.sum()<30: continue
    f=av[i,rm]; x=la.values[i,rm]
    try: b=np.linalg.lstsq(np.column_stack([np.ones(rm.sum()),x]),f,rcond=None)[0]; av[i,rm]=f-np.column_stack([np.ones(rm.sum()),x])@b
    except: pass
fa=pd.DataFrame(av,index=cd,columns=cn)
fa=fa.sub(fa.mean(axis=1),axis=0).div(fa.std(axis=1),axis=0).clip(-3,3)

# Final align
cd2=sorted(fa.dropna(how="all").index.intersection(rp2.dropna(how="all").index))
cn2=sorted(fa.columns.intersection(rp2.columns))
fa=fa.loc[cd2,cn2]; ra=rp2.loc[cd2,cn2]
print(f"   final: {len(cd2)}d × {len(cn2)}")

# Direction test
print("[4] 方向...")
_ic_pos = compute_ic_dynamic(fa, ra, FW, "pearson").mean()
_ic_neg = compute_ic_dynamic(-fa, ra, FW, "pearson").mean()
print(f"   正向: IC={_ic_pos:.4f}  反向: IC={_ic_neg:.4f}")
if _ic_neg > abs(_ic_pos):
    fa = -fa; d = -1; d_desc = "反向(低AWM→高收益:高量日跌→反转)"
    print(f"   → 反向 ✓")
else:
    d = 1; d_desc = "正向(高AWM→高收益:高量日涨→动量)"
    print(f"   → 正向 ✓")

# Re-run IC for final direction
print("[5] 回测...")
ics=compute_ic_dynamic(fa,ra,FW,"pearson")
rics=compute_ic_dynamic(fa,ra,FW,"spearman")
gr,tov,hi=compute_group_returns(fa,ra,NGRP,RF,COST)
mt=compute_metrics(gr,ics,rics,tov,NGRP,holdings_info=hi)

# Correlations
print("[6] 相关性...")
t1=time.time()
hp=df.pivot_table(index="date",columns="sc",values="high",dropna=False)
lp=df.pivot_table(index="date",columns="sc",values="low",dropna=False)
op=df.pivot_table(index="date",columns="sc",values="open")
tp2=df.pivot_table(index="date",columns="sc",values="turnover",dropna=False)
aamp=df.pivot_table(index="date",columns="sc",values="amplitude")
amihud_f=np.log((rp2.abs()/(ap.loc[cd2,cn2]/1e8).clip(lower=1e-8)).rolling(20,min_periods=10).mean().clip(lower=1e-12))
usr=(hp.loc[cd2,cn2]-np.maximum(cp.loc[cd2,cn2],op.loc[cd2,cn2]))/(hp.loc[cd2,cn2]-lp.loc[cd2,cn2]).clip(lower=1e-8)
lsr=(np.minimum(cp.loc[cd2,cn2],op.loc[cd2,cn2])-lp.loc[cd2,cn2])/(hp.loc[cd2,cn2]-lp.loc[cd2,cn2]).clip(lower=1e-8)
shadow_f=(usr-lsr).rolling(20,min_periods=10).mean()
oret=op.loc[cd2,cn2]/cp.loc[cd2,cn2].shift(1)-1; iret=cp.loc[cd2,cn2]/op.loc[cd2,cn2]-1
overnight_f=oret.rolling(20,min_periods=10).sum()-iret.rolling(20,min_periods=10).sum()
cmat=np.full((len(cd2),len(cn2)),np.nan); rn=rp2.values
for i in range(10,len(cd2)):
    w=rn[i-10:i]; s=np.sort(w,axis=0); b2=np.nanmean(s[:2],axis=0); vc=np.sum(~np.isnan(w),axis=0)
    b2[vc<5]=np.nan; cmat[i,:]=-b2
cvdf=pd.DataFrame(cmat,index=cd2,columns=cn2)
negf=(rp2<=-0.03).astype(float).rolling(10,min_periods=5).mean()
tl=np.log(tp2.loc[cd2,cn2].rolling(20,min_periods=10).mean().clip(lower=1e-8))
tae_r=np.log(tp2.loc[cd2,cn2].rolling(20,min_periods=10).mean().clip(lower=1e-8)/aamp.loc[cd2,cn2].rolling(20,min_periods=10).mean().clip(lower=0.01))
al=np.log(aamp.loc[cd2,cn2].rolling(60,min_periods=30).mean().clip(lower=0.01))
ms=np.stack([cp.loc[cd2,cn2].rolling(w).mean().values/cp.loc[cd2,cn2].values for w in [5,10,20,40,60,120]],axis=-1)
mdisp=pd.DataFrame(np.nanstd(ms,axis=-1),index=cd2,columns=cn2)
lad=np.log(ap.loc[cd2,cn2].clip(lower=1)).diff().abs().rolling(20,min_periods=10).mean()
pvc=rp2.rolling(20,min_periods=10).corr(lad)
fcorr=fa; sd=cd2[::max(1,len(cd2)//20)]
corrs={}
for name,other in [('amihud_illiq_v2',amihud_f),('shadow_pressure_v1',shadow_f),('overnight_momentum_v1',overnight_f),
  ('tail_risk_cvar_v1',cvdf),('neg_day_freq_v1',negf),('turnover_level_v1',tl),
  ('tae_v1',tae_r),('amp_level_v2',al),('ma_disp_v1',mdisp),('pv_corr_v1',pvc)]:
    cr=[]
    for dd in sd:
        f1=fcorr.loc[dd].dropna()
        f2=other.loc[dd].reindex(f1.index).dropna() if isinstance(other,pd.DataFrame) else None
        if f2 is None: continue
        c=f1.index.intersection(f2.index)
        if len(c)>50:
            try: r,_=sp_stats.spearmanr(f1[c],f2[c])
            except: continue
            if not np.isnan(r): cr.append(r)
    corrs[name]=round(float(np.mean(cr)),3) if cr else 0
    print(f"   vs {name}: {corrs[name]:.3f}")
corrs[FID]=1.0; print(f"   ({time.time()-t1:.1f}s)")

# Output
print("[7] 写入...")
OUT.mkdir(parents=True,exist_ok=True)
save_backtest_data(gr,ics,rics,str(OUT))
def nn(o):
    if isinstance(o,(np.bool_,)): return bool(o)
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,(np.floating,)): return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o,dict): return {k:nn(v) for k,v in o.items()}
    if isinstance(o,list): return [nn(v) for v in o]
    return o
rpt={"factor_id":FID,"factor_name":"成交额加权动量 v1","factor_name_en":"Amount-Weighted Momentum v1",
 "category":"量价/资金流向","description":"20日(Σret×amt/Σamt − mean(ret)), 高量日回报倾向。",
 "hypothesis":"高量日蕴含信息。AWM>0=买盘推动=动量延续; AWM<0=卖盘主导=后续反弹。",
 "formula":"neutralize(Σ_20d(ret×amt)/Σ_20d(amt) − mean(ret)_20d, log_amount_20d)",
 "direction":d,"direction_desc":d_desc,"stock_pool":"中证1000",
 "period":f"{cd2[0].strftime('%Y-%m-%d')} ~ {cd2[-1].strftime('%Y-%m-%d')}",
 "n_dates":len(cd2),"n_stocks":len(cn2),"n_groups":NGRP,"rebalance_freq":RF,
 "forward_days":FW,"cost":COST,"data_cutoff":DC,"correlations":corrs,"metrics":mt}
with open(str(OUT/"backtest_report.json"),"w",encoding="utf-8") as f: json.dump(nn(rpt),f,indent=2,ensure_ascii=False)

# Summary
im=mt.get("ic_mean",0)or 0; it=mt.get("ic_t_stat",0)or 0; lss=mt.get("long_short_sharpe",0)or 0
lsm=mt.get("long_short_mdd",0)or 0; mo=mt.get("monotonicity",0)or 0
print(f"\n{'='*60}")
print(f"  {FID}: {d_desc}")
print(f"  IC={im:.4f} t={it:.2f} LSS={lss:.3f} Mono={mo:.3f} MDD={lsm:.1%}")
for i,r in enumerate(mt.get("group_returns_annualized",[]),1): print(f"  G{i}: {r:.2%}" if r is not None else f"  G{i}: N/A")
valid=abs(im)>0.015 and abs(it)>2 and abs(lss)>0.5
print(f"  {'✅有效' if valid else '❌无效'}")
print(f"  总耗时: {time.time()-t0:.0f}s")
