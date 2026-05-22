import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
BASE=Path(".")
kline=pd.read_csv(BASE/"data"/"csi1000_kline_raw.csv",parse_dates=["date"])
returns=pd.read_csv(BASE/"data"/"csi1000_returns.csv",parse_dates=["date"])

vol=kline.pivot_table(index="date",columns="stock_code",values="volume").sort_index()
ret=returns.pivot_table(index="date",columns="stock_code",values="return").sort_index()
dates=vol.index.intersection(ret.index); stocks=vol.columns.intersection(ret.columns)
vol=vol.loc[dates,stocks]; ret=ret.loc[dates,stocks]
T,N=vol.shape; vv=vol.values.astype(float)

# volume cross-sectional pct-rank per day
ranks=np.empty_like(vv,dtype=float)
for t in range(T):
    ranks[t]=pd.Series(vv[t]).rank(method="average").values / N

W, HI, LO = 20, 0.70, 0.30

frames=[]
for i in range(W, T):
    pct=ranks[i-W:i]          # (W,N)
    hi=(pct>=HI); lo=(pct<=LO)
    hc=hi.sum(0); lc=lo.sum(0)
    ok=(hc>=3)&(lc>=3)
    r=np.empty(N,dtype=float); r[:]=np.nan
    if ok.any():
        hi_sum=(pct*hi).sum(0); lo_sum=(pct*lo).sum(0)
        r[ok]=(hi_sum[ok]/hc[ok])-(lo_sum[ok]/lc[ok])
    frames.append(pd.Series(r, index=vol.columns, name=vol.index[i]))

fac=pd.DataFrame(frames)   # (T-W, N)

# ── neutralization by 20d log-amount ──
amt=kline.pivot_table(index="date",columns="stock_code",values="amount").sort_index()
amt=amt.loc[vol.index, stocks]
la=np.log(amt.rolling(20).mean()+1)

cd=fac.index.intersection(la.index); cs=fac.columns.intersection(la.columns)
fac=fac.loc[cd,cs]; la2=la.loc[cd,cs]
res=[]
for i in range(len(fac)):
    y=fac.values[i]; x=la2.values[i]; m=np.isfinite(y)&np.isfinite(x)
    if m.sum()<30: res.append(np.full(len(y),np.nan)); continue
    yy,xx=y[m],x[m]; dm=xx-xx.mean()
    b=(dm*yy).sum()/((dm**2).sum()+1e-12); a=yy.mean()-b*xx.mean()
    r=np.full(len(y),np.nan); r[m]=yy-(a+b*xx); res.append(r)
resid=pd.DataFrame(res,index=fac.index,columns=fac.columns)

def madz(s):
    med=s.median(); mad=(s-med).abs().median()*1.4826
    if mad<1e-10: return pd.Series(0.,index=s.index)
    c=s.clip(med-3.5*mad,med+3.5*mad)
    return (c-c.mean())/(c.std()+1e-10)

f=resid.apply(madz,axis=1)
out=f.stack().reset_index(); out.columns=["date","stock_code","factor_value"]
out["stock_code"]=out["stock_code"].astype(str).str.zfill(6)
out["date"]=out["date"].astype(str)
out.dropna(subset=["factor_value"]).to_csv(BASE/"data"/"factor_vol_ext_confirm_v1.csv",index=False)
print(f"rows={len(out)}, mean={out['factor_value'].mean():.4f}, std={out['factor_value'].std():.4f}")
