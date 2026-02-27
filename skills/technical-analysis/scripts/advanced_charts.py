#!/usr/bin/env python3
"""
高级技术分析图表 v2
用法: python3 advanced_charts.py <kline_json> <name> <code> <output_dir> [bench_json]
"""
import json, sys, os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import argrelextrema
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['PingFang SC','Hiragino Sans GB','STHeiti','SimHei','Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

BG='#0d1117'; PNL='#161b22'; GRD='#21262d'
C = dict(up='#ef5350',dn='#26a69a',uf='#ef535060',df='#26a69a60',
    ma5='#ffd700',ma10='#00bfff',ma20='#ff69b4',ma60='#32cd32',
    bup='#ff6b9d',blo='#45b7d1',bmid='#8b949e',
    mp='#ef5350',mn='#26a69a',dif='#ffd700',dea='#00bfff',
    kv='#ffd700',dv='#00bfff',jv='#ff69b4',
    r6='#ffd700',r12='#00bfff',
    vp='#58a6ff',poc='#f0883e',
    rs='#f78166',rsu='#3fb95040',rsd='#f8514940',
    hv='#bc8cff',hv2='#7ee787',
    sup='#3fb950',res='#f85149',pat='#d2a8ff',
    txt='#e6edf3',lbl='#8b949e',obv='#ff7b72')

def parse_kline(src):
    if isinstance(src,str) and os.path.exists(src):
        with open(src) as f: data=json.load(f)
    elif isinstance(src,str): data=json.loads(src)
    else: data=src
    recs = data.get('data',data) if isinstance(data,dict) else data
    df=pd.DataFrame(recs)
    rm={}
    if 'trade_date' in df.columns: rm['trade_date']='date'
    if 'open_price' in df.columns:
        rm.update(open_price='open',high_price='high',low_price='low',close_price='close',trade_amount='volume')
    if rm: df=df.rename(columns=rm)
    df['date']=pd.to_datetime(df['date'].astype(str))
    df=df.sort_values('date').reset_index(drop=True)
    for c in ['open','high','low','close','volume']:
        df[c]=pd.to_numeric(df[c],errors='coerce')
    return df

def calc_ind(df):
    for p in [5,10,20,60]: df[f'ma{p}']=df['close'].rolling(p,min_periods=1).mean()
    m20=df['close'].rolling(20).mean(); s20=df['close'].rolling(20).std()
    df['bup']=m20+2*s20; df['blo']=m20-2*s20; df['bmid']=m20
    e12=df['close'].ewm(span=12,adjust=False).mean()
    e26=df['close'].ewm(span=26,adjust=False).mean()
    df['dif']=e12-e26; df['dea']=df['dif'].ewm(span=9,adjust=False).mean()
    df['mh']=2*(df['dif']-df['dea'])
    lo9=df['low'].rolling(9,min_periods=1).min()
    hi9=df['high'].rolling(9,min_periods=1).max()
    rsv=((df['close']-lo9)/(hi9-lo9).replace(0,np.nan)*100).fillna(50)
    df['kv']=rsv.ewm(com=2,adjust=False).mean()
    df['dv']=df['kv'].ewm(com=2,adjust=False).mean()
    df['jv']=3*df['kv']-2*df['dv']
    d=df['close'].diff()
    g6=d.where(d>0,0).rolling(6).mean(); l6=(-d.where(d<0,0)).rolling(6).mean()
    df['rsi6']=100-100/(1+g6/l6.replace(0,np.nan))
    g12=d.where(d>0,0).rolling(12).mean(); l12=(-d.where(d<0,0)).rolling(12).mean()
    df['rsi12']=100-100/(1+g12/l12.replace(0,np.nan))
    obv=[0]
    for i in range(1,len(df)):
        if df['close'].iloc[i]>df['close'].iloc[i-1]: obv.append(obv[-1]+df['volume'].iloc[i])
        elif df['close'].iloc[i]<df['close'].iloc[i-1]: obv.append(obv[-1]-df['volume'].iloc[i])
        else: obv.append(obv[-1])
    df['obv']=obv
    df['vm5']=df['volume'].rolling(5).mean()
    df['vm10']=df['volume'].rolling(10).mean()
    tr=pd.DataFrame(dict(a=df['high']-df['low'],b=abs(df['high']-df['close'].shift(1)),c=abs(df['low']-df['close'].shift(1)))).max(axis=1)
    df['atr14']=tr.rolling(14).mean()
    df['lr']=np.log(df['close']/df['close'].shift(1))
    df['hv20']=df['lr'].rolling(20).std()*np.sqrt(252)*100
    df['hv60']=df['lr'].rolling(60).std()*np.sqrt(252)*100
    return df

def resample(df, freq='W'):
    d2=df.set_index('date'); f2='ME' if freq=='M' else freq
    r=d2.resample(f2).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    return r.reset_index()

def find_pivots(df, order=5):
    hi=argrelextrema(df['high'].values,np.greater,order=order)[0]
    lo=argrelextrema(df['low'].values,np.less,order=order)[0]
    return hi, lo

def detect_patterns(df, hi, lo):
    pats=[]
    for i in range(len(lo)-1):
        a,b=lo[i],lo[i+1]
        if b-a<5: continue
        pa,pb=df['low'].iloc[a],df['low'].iloc[b]
        if abs(pa-pb)/max(pa,pb)<0.04:
            mh=[h for h in hi if a<h<b]
            if mh:
                nk=max(df['high'].iloc[h] for h in mh)
                pats.append(dict(t='双底(W)',s=a,e=b,nk=nk,tgt=nk+(nk-min(pa,pb))))
    for i in range(len(hi)-1):
        a,b=hi[i],hi[i+1]
        if b-a<5: continue
        pa,pb=df['high'].iloc[a],df['high'].iloc[b]
        if abs(pa-pb)/max(pa,pb)<0.04:
            ml=[l for l in lo if a<l<b]
            if ml:
                nk=min(df['low'].iloc[l] for l in ml)
                pats.append(dict(t='双顶(M)',s=a,e=b,nk=nk,tgt=nk-(max(pa,pb)-nk)))
    for i in range(len(hi)-2):
        a,b,c=hi[i],hi[i+1],hi[i+2]
        pa,pb,pc=df['high'].iloc[a],df['high'].iloc[b],df['high'].iloc[c]
        if pb>pa and pb>pc and abs(pa-pc)/max(pa,pc)<0.06:
            m1=[l for l in lo if a<l<b]; m2=[l for l in lo if b<l<c]
            if m1 and m2:
                nk=(df['low'].iloc[m1[0]]+df['low'].iloc[m2[0]])/2
                pats.append(dict(t='头肩顶',s=a,e=c,nk=nk,tgt=nk-(pb-nk)))
    for i in range(len(lo)-2):
        a,b,c=lo[i],lo[i+1],lo[i+2]
        pa,pb,pc=df['low'].iloc[a],df['low'].iloc[b],df['low'].iloc[c]
        if pb<pa and pb<pc and abs(pa-pc)/max(pa,pc)<0.06:
            m1=[h for h in hi if a<h<b]; m2=[h for h in hi if b<h<c]
            if m1 and m2:
                nk=(df['high'].iloc[m1[0]]+df['high'].iloc[m2[0]])/2
                pats.append(dict(t='头肩底',s=a,e=c,nk=nk,tgt=nk+(nk-pb)))
    if len(hi)>=2 and len(lo)>=2:
        rh=[(h,df['high'].iloc[h]) for h in hi[-4:]]
        rl=[(l,df['low'].iloc[l]) for l in lo[-4:]]
        if len(rh)>=2 and len(rl)>=2 and rh[-1][1]-rh[0][1]<0 and rl[-1][1]-rl[0][1]>0:
            pats.append(dict(t='三角收敛',s=min(rh[0][0],rl[0][0]),e=max(rh[-1][0],rl[-1][0]),hp=rh,lp=rl))
    return pats

def detect_candles(df):
    ps=[]
    for i in range(2,len(df)):
        o,h,l,c=df.iloc[i][['open','high','low','close']]
        po,_,_,pc=df.iloc[i-1][['open','high','low','close']]
        bd=abs(c-o); fl=h-l
        if fl==0: continue
        lo_sh=min(o,c)-l
        if lo_sh>2*bd and (h-max(o,c))<bd*0.3 and bd>0 and i>=5 and df['close'].iloc[i]<df['close'].iloc[i-5]:
            ps.append(dict(t='锤子线',i=i,sig='bull'))
        if pc<po and c>o and c>po and o<pc: ps.append(dict(t='看涨吞没',i=i,sig='bull'))
        if pc>po and c<o and c<po and o>pc: ps.append(dict(t='看跌吞没',i=i,sig='bear'))
        if bd<fl*0.1 and i>=5 and abs(df['close'].iloc[i]-df['close'].iloc[i-5])/df['close'].iloc[i-5]>0.03:
            ps.append(dict(t='十字星',i=i,sig='rev'))
    return ps

def vol_profile(df, bins=40):
    lo_p,hi_p=df['low'].min(),df['high'].max()
    edges=np.linspace(lo_p,hi_p,bins+1); vp=np.zeros(bins)
    for _,r in df.iterrows():
        for j in range(bins):
            ov=max(0,min(r['high'],edges[j+1])-max(r['low'],edges[j]))
            rng=r['high']-r['low']
            if rng>0: vp[j]+=r['volume']*ov/rng
    ctrs=(edges[:-1]+edges[1:])/2; pi=np.argmax(vp); poc=ctrs[pi]
    tot=vp.sum(); tgt=tot*0.7; si=np.argsort(vp)[::-1]; cum=0; vab=[]
    for idx in si:
        cum+=vp[idx]; vab.append(idx)
        if cum>=tgt: break
    return ctrs,vp,poc,edges[max(vab)+1],edges[min(vab)]

def sty(ax,yl='',sx=False):
    ax.set_facecolor(PNL); ax.grid(True,alpha=0.12,color=GRD)
    ax.tick_params(colors=C['lbl'],labelsize=7)
    for s in ax.spines.values(): s.set_color(GRD)
    if yl: ax.set_ylabel(yl,fontsize=8,color=C['lbl'])
    if not sx: plt.setp(ax.get_xticklabels(),visible=False)

def dticks(ax,df,n=10):
    lb=df['date'].dt.strftime('%m/%d')
    tk=list(range(0,len(df),max(1,len(df)//n)))
    ax.set_xticks(tk)
    ax.set_xticklabels([lb.iloc[i] for i in tk],rotation=45,fontsize=7,color=C['lbl'])

def draw_candles(ax,df,w=0.6,ma=True,boll=True):
    for i,r in df.iterrows():
        u=r['close']>=r['open']; cl=C['up'] if u else C['dn']
        ax.plot([i,i],[r['low'],r['high']],color=cl,lw=0.6)
        bb=min(r['open'],r['close'])
        bh=abs(r['close']-r['open']) or r['close']*0.001
        ax.add_patch(Rectangle((i-w/2,bb),w,bh,fc=cl if u else PNL,ec=cl,lw=0.6))
    if ma:
        for col,cl,lb in [('ma5',C['ma5'],'MA5'),('ma10',C['ma10'],'MA10'),('ma20',C['ma20'],'MA20'),('ma60',C['ma60'],'MA60')]:
            if col in df.columns and df[col].notna().any(): ax.plot(range(len(df)),df[col],color=cl,lw=1,label=lb,alpha=0.8)
    if boll and 'bup' in df.columns and df['bup'].notna().any():
        ax.plot(range(len(df)),df['bup'],color=C['bup'],lw=0.7,ls='--',alpha=0.5)
        ax.plot(range(len(df)),df['bmid'],color=C['bmid'],lw=0.7,alpha=0.4)
        ax.plot(range(len(df)),df['blo'],color=C['blo'],lw=0.7,ls='--',alpha=0.5)
        ax.fill_between(range(len(df)),df['bup'],df['blo'],alpha=0.03,color='w')
    ax.set_xlim(-1,len(df))

def draw_vol(ax,df):
    cls=[C['uf'] if df.iloc[i]['close']>=df.iloc[i]['open'] else C['df'] for i in range(len(df))]
    mx=df['volume'].max()
    div,u=(1e8,'亿') if mx>1e8 else (1e4,'万') if mx>1e4 else (1,'')
    ax.bar(range(len(df)),df['volume']/div,color=cls,width=0.6)
    if 'vm5' in df.columns: ax.plot(range(len(df)),df['vm5']/div,color=C['ma5'],lw=0.7,alpha=0.6)
    ax.set_xlim(-1,len(df)); sty(ax,f'量({u})')

# ═══ 图1: 主图 K线+形态+VP ═══
def chart_main(df_full, name, code, out, days=90):
    df=df_full.tail(days).reset_index(drop=True); df=calc_ind(df)
    hi,lo=find_pivots(df,5); pats=detect_patterns(df,hi,lo); cps=detect_candles(df)
    vc,vv,poc,vah,val=vol_profile(df,40)
    fig=plt.figure(figsize=(18,12),facecolor=BG)
    gs=fig.add_gridspec(3,2,height_ratios=[5,1.2,1.5],width_ratios=[5,1],hspace=0.04,wspace=0.02)
    ax=fig.add_subplot(gs[0,0]); draw_candles(ax,df); sty(ax,'价格')
    for h in hi: ax.scatter(h,df['high'].iloc[h],marker='v',color=C['res'],s=25,zorder=5,alpha=0.5)
    for l in lo: ax.scatter(l,df['low'].iloc[l],marker='^',color=C['sup'],s=25,zorder=5,alpha=0.5)
    for p in pats:
        mid=(p['s']+p['e'])//2
        if 'nk' in p:
            ax.axhline(y=p['nk'],xmin=max(0,p['s'])/len(df),xmax=min(p['e'],len(df)-1)/len(df),
                       color=C['pat'],lw=1.2,ls='--',alpha=0.7)
            if 'tgt' in p:
                ax.axhline(y=p['tgt'],xmin=p['e']/len(df),xmax=1,color=C['pat'],lw=0.8,ls=':',alpha=0.4)
            ax.text(mid,p['nk'],f" {p['t']}\n tgt:{p.get('tgt',0):.1f}",fontsize=7,color=C['pat'],
                    fontweight='bold',bbox=dict(boxstyle='round,pad=0.2',fc=PNL,ec=C['pat'],alpha=0.9))
        elif 'hp' in p:
            hp_,lp_=p['hp'],p['lp']
            if len(hp_)>=2: ax.plot([hp_[0][0],hp_[-1][0]],[hp_[0][1],hp_[-1][1]],color=C['res'],lw=1.5,ls='--',alpha=0.7)
            if len(lp_)>=2: ax.plot([lp_[0][0],lp_[-1][0]],[lp_[0][1],lp_[-1][1]],color=C['sup'],lw=1.5,ls='--',alpha=0.7)
    for cp in cps[-5:]:
        idx=cp['i']; rng=df['high'].max()-df['low'].min()
        if cp['sig']=='bull':
            ax.annotate(cp['t'],xy=(idx,df['low'].iloc[idx]),
                xytext=(idx,df['low'].iloc[idx]-rng*0.04),fontsize=6,color=C['sup'],ha='center',
                arrowprops=dict(arrowstyle='->',color=C['sup'],lw=0.5))
        elif cp['sig']=='bear':
            ax.annotate(cp['t'],xy=(idx,df['high'].iloc[idx]),
                xytext=(idx,df['high'].iloc[idx]+rng*0.04),fontsize=6,color=C['res'],ha='center',
                arrowprops=dict(arrowstyle='->',color=C['res'],lw=0.5))
    ax.legend(loc='upper left',fontsize=7,facecolor=PNL,edgecolor=GRD,labelcolor=C['lbl'],framealpha=0.9)
    chg=(df.iloc[-1]['close']/df.iloc[-2]['close']-1)*100 if len(df)>1 else 0
    ax.set_title(f'{name} ({code}) main  {df.iloc[-1]["close"]:.2f}  {chg:+.2f}%',
                 fontsize=14,fontweight='bold',color=C['up'] if chg>=0 else C['dn'],pad=10)
    # VP sidebar
    axv=fig.add_subplot(gs[0,1],sharey=ax); axv.set_facecolor(PNL)
    nvp=vv/vv.max() if vv.max()>0 else vv
    bclr=[C['poc'] if abs(cv-poc)<(vc[1]-vc[0]) else C['vp'] for cv in vc]
    axv.barh(vc,nvp,height=(vc[1]-vc[0])*0.9,color=bclr,alpha=0.7)
    axv.axhline(poc,color=C['poc'],lw=1.5,alpha=0.8)
    axv.text(0.95,poc,f'POC {poc:.1f}',transform=axv.get_yaxis_transform(),fontsize=7,
             color=C['poc'],va='bottom',ha='right',fontweight='bold')
    axv.axhline(vah,color=C['vp'],lw=0.8,ls='--',alpha=0.5)
    axv.axhline(val,color=C['vp'],lw=0.8,ls='--',alpha=0.5)
    axv.text(0.95,vah,f'VAH {vah:.1f}',transform=axv.get_yaxis_transform(),fontsize=6,color=C['vp'],va='bottom',ha='right')
    axv.text(0.95,val,f'VAL {val:.1f}',transform=axv.get_yaxis_transform(),fontsize=6,color=C['vp'],va='top',ha='right')
    axv.set_xlim(0,1.1); axv.set_title('筹码分布',fontsize=9,color=C['lbl'],pad=5)
    axv.tick_params(left=False,labelleft=False,colors=C['lbl'],labelsize=7)
    for sp in axv.spines.values(): sp.set_color(GRD)
    axv.grid(False)
    ax2=fig.add_subplot(gs[1,0],sharex=ax); draw_vol(ax2,df)
    ax3=fig.add_subplot(gs[2,0],sharex=ax)
    mc=[C['mp'] if v>=0 else C['mn'] for v in df['mh']]
    ax3.bar(range(len(df)),df['mh'],color=mc,width=0.6,alpha=0.7)
    ax3.plot(range(len(df)),df['dif'],color=C['dif'],lw=1,label='DIF')
    ax3.plot(range(len(df)),df['dea'],color=C['dea'],lw=1,label='DEA')
    ax3.axhline(0,color=C['lbl'],lw=0.3); ax3.set_xlim(-1,len(df))
    ax3.legend(loc='upper left',fontsize=7,facecolor=PNL,edgecolor=GRD,labelcolor=C['lbl'])
    sty(ax3,'MACD',sx=True); dticks(ax3,df)
    for g in [gs[1,1],gs[2,1]]:
        a_=fig.add_subplot(g); a_.set_facecolor(BG); a_.axis('off')
    plt.savefig(out,dpi=150,bbox_inches='tight',facecolor=BG); plt.close()
    print(f"[1/4] main: {out}")
    return dict(poc=round(poc,2),vah=round(vah,2),val=round(val,2),
                patterns=[p['t'] for p in pats],candle_pats=[p['t'] for p in cps[-5:]],
                pivots_hi=[round(float(df['high'].iloc[h]),2) for h in hi],
                pivots_lo=[round(float(df['low'].iloc[l]),2) for l in lo])

# ═══ 图2: 多周期共振 ═══
def chart_multi(df_full, name, code, out):
    fig,axes=plt.subplots(3,1,figsize=(16,14),facecolor=BG)
    for ax,(freq,label,n) in zip(axes,[('D','日线',90),('W','周线',52),('M','月线',24)]):
        if freq=='D': d=df_full.tail(n).reset_index(drop=True)
        else: d=resample(df_full,freq).tail(n).reset_index(drop=True)
        d=calc_ind(d); draw_candles(ax,d,ma=True,boll=False); sty(ax,label)
        if len(d)>10:
            h2,l2=find_pivots(d,order=max(3,len(d)//15))
            for h in h2: ax.scatter(h,d['high'].iloc[h],marker='v',color=C['res'],s=20,zorder=5,alpha=0.5)
            for l in l2: ax.scatter(l,d['low'].iloc[l],marker='^',color=C['sup'],s=20,zorder=5,alpha=0.5)
        ax.legend(loc='upper left',fontsize=7,facecolor=PNL,edgecolor=GRD,labelcolor=C['lbl'],framealpha=0.9)
        dticks(ax,d); plt.setp(ax.get_xticklabels(),visible=True)
        ax.set_title(f'{name} {label}',fontsize=11,color=C['txt'],pad=5,loc='left')
    fig.suptitle(f'{name} ({code}) 多周期共振',fontsize=14,fontweight='bold',color=C['txt'],y=0.98)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(out,dpi=150,bbox_inches='tight',facecolor=BG); plt.close()
    print(f"[2/4] multi: {out}")

# ═══ 图3: RS+HV+OBV ═══
def chart_rs_vol(df_full, name, code, out, bench_df=None, days=120):
    df=df_full.tail(days).reset_index(drop=True); df=calc_ind(df)
    has_b = bench_df is not None and len(bench_df)>0
    nrows=4 if has_b else 3
    hrs=[2,1.5,1.5,1.5][:nrows]
    fig,axes=plt.subplots(nrows,1,figsize=(16,3*nrows+2),facecolor=BG,gridspec_kw=dict(height_ratios=hrs))
    row=0
    if has_b:
        ax=axes[row]; row+=1
        mg=df[['date','close']].merge(bench_df[['date','close']],on='date',suffixes=('_s','_b'))
        if len(mg)>5:
            mg['sr']=mg['close_s']/mg['close_s'].iloc[0]
            mg['br']=mg['close_b']/mg['close_b'].iloc[0]
            mg['rs_v']=mg['sr']/mg['br']
            mg['ex']=(mg['sr']-mg['br'])*100
            x=range(len(mg))
            ax.plot(x,mg['rs_v'],color=C['rs'],lw=2,label='RS=个股/沪深300')
            ax.axhline(1,color=C['lbl'],lw=0.5,ls='--')
            ax.fill_between(x,1,mg['rs_v'],where=mg['rs_v']>=1,color=C['rsu'],alpha=0.6,label='跑赢')
            ax.fill_between(x,1,mg['rs_v'],where=mg['rs_v']<1,color=C['rsd'],alpha=0.6,label='跑输')
            ex=mg['ex'].iloc[-1]
            ax.text(len(mg)-1,mg['rs_v'].iloc[-1],f' 超额{ex:+.1f}%',fontsize=9,
                    color=C['sup'] if ex>=0 else C['res'],fontweight='bold')
            ax.legend(loc='upper left',fontsize=7,facecolor=PNL,edgecolor=GRD,labelcolor=C['lbl'])
            sty(ax,'RS'); ax.set_xlim(-1,len(mg))
            ax.set_title(f'{name} vs 沪深300 相对强弱',fontsize=11,color=C['txt'],pad=5,loc='left')
    # HV
    ax_hv=axes[row]; row+=1; x=range(len(df))
    ax_hv.plot(x,df['hv20'],color=C['hv'],lw=1.5,label='HV20')
    ax_hv.plot(x,df['hv60'],color=C['hv2'],lw=1.5,label='HV60')
    ax_hv.fill_between(x,df['hv20'],df['hv60'],alpha=0.1,color=C['hv'])
    last_hv=df['hv20'].iloc[-1]
    ax_hv.text(len(df)-1,last_hv,f' {last_hv:.1f}%',fontsize=8,color=C['hv'],fontweight='bold')
    ax_hv.legend(loc='upper left',fontsize=7,facecolor=PNL,edgecolor=GRD,labelcolor=C['lbl'])
    ax_hv.set_title(f'{name} 历史波动率',fontsize=11,color=C['txt'],pad=5,loc='left')
    sty(ax_hv,'HV%'); ax_hv.set_xlim(-1,len(df))
    # ATR
    ax_atr=axes[row]; row+=1
    ax_atr.plot(x,df['atr14'],color=C['lbl'],lw=1.5,label='ATR14')
    ax_atr.fill_between(x,0,df['atr14'],alpha=0.15,color=C['lbl'])
    ax_atr.legend(loc='upper left',fontsize=7,facecolor=PNL,edgecolor=GRD,labelcolor=C['lbl'])
    ax_atr.set_title('ATR(14) 真实波幅',fontsize=11,color=C['txt'],pad=5,loc='left')
    sty(ax_atr,'ATR'); ax_atr.set_xlim(-1,len(df))
    # OBV
    ax_ob=axes[row]; row+=1
    obv_ma=pd.Series(df['obv'].values).rolling(20).mean()
    mx_ob=abs(df['obv'].max())
    div_o,u_o=(1e8,'亿') if mx_ob>1e8 else (1e4,'万') if mx_ob>1e4 else (1,'')
    ax_ob.plot(x,np.array(df['obv'])/div_o,color=C['obv'],lw=1.5,label='OBV')
    ax_ob.plot(x,obv_ma.values/div_o,color=C['ma20'],lw=1,label='OBV MA20',alpha=0.7)
    if len(df)>20:
        p_hi,_=find_pivots(df,order=10)
        for j in range(1,len(p_hi)):
            i1,i2=p_hi[j-1],p_hi[j]
            if df['close'].iloc[i2]>df['close'].iloc[i1] and df['obv'].iloc[i2]<df['obv'].iloc[i1]:
                ax_ob.annotate('量价背离',xy=(i2,df['obv'].iloc[i2]/div_o),fontsize=7,color=C['res'],
                    fontweight='bold',ha='center')
    ax_ob.legend(loc='upper left',fontsize=7,facecolor=PNL,edgecolor=GRD,labelcolor=C['lbl'])
    ax_ob.set_title('OBV 能量潮',fontsize=11,color=C['txt'],pad=5,loc='left')
    sty(ax_ob,f'OBV({u_o})',sx=True); ax_ob.set_xlim(-1,len(df)); dticks(ax_ob,df)
    fig.suptitle(f'{name} ({code}) 强弱&波动率',fontsize=14,fontweight='bold',color=C['txt'],y=0.98)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(out,dpi=150,bbox_inches='tight',facecolor=BG); plt.close()
    print(f'[3/4] rs_vol: {out}')

# ═══ 图4: MACD+KDJ+RSI 指标面板 ═══
def chart_indicators(df_full, name, code, out, days=90):
    df=df_full.tail(days).reset_index(drop=True); df=calc_ind(df)
    fig,axes=plt.subplots(3,1,figsize=(16,10),facecolor=BG,gridspec_kw=dict(height_ratios=[1,1,1]))
    x=range(len(df))
    # MACD
    ax=axes[0]
    mc=[C['mp'] if v>=0 else C['mn'] for v in df['mh']]
    ax.bar(x,df['mh'],color=mc,width=0.6,alpha=0.7)
    ax.plot(x,df['dif'],color=C['dif'],lw=1.2,label='DIF')
    ax.plot(x,df['dea'],color=C['dea'],lw=1.2,label='DEA')
    ax.axhline(0,color=C['lbl'],lw=0.3)
    for i in range(1,len(df)):
        if df['dif'].iloc[i-1]<df['dea'].iloc[i-1] and df['dif'].iloc[i]>=df['dea'].iloc[i]:
            ax.scatter(i,df['dif'].iloc[i],marker='^',color=C['sup'],s=50,zorder=5)
            ax.text(i,df['dif'].iloc[i],' 金叉',fontsize=6,color=C['sup'],va='bottom')
        elif df['dif'].iloc[i-1]>df['dea'].iloc[i-1] and df['dif'].iloc[i]<=df['dea'].iloc[i]:
            ax.scatter(i,df['dif'].iloc[i],marker='v',color=C['res'],s=50,zorder=5)
            ax.text(i,df['dif'].iloc[i],' 死叉',fontsize=6,color=C['res'],va='top')
    ax.legend(loc='upper left',fontsize=7,facecolor=PNL,edgecolor=GRD,labelcolor=C['lbl'])
    ax.set_title(f'{name} MACD',fontsize=11,color=C['txt'],pad=5,loc='left')
    sty(ax,'MACD'); ax.set_xlim(-1,len(df))
    # KDJ
    ax2=axes[1]
    ax2.plot(x,df['kv'],color=C['kv'],lw=1.2,label='K')
    ax2.plot(x,df['dv'],color=C['dv'],lw=1.2,label='D')
    ax2.plot(x,df['jv'],color=C['jv'],lw=1,label='J',alpha=0.7)
    ax2.axhspan(80,100,alpha=0.08,color=C['res'])
    ax2.axhspan(0,20,alpha=0.08,color=C['sup'])
    ax2.axhline(80,color=C['res'],lw=0.5,ls='--',alpha=0.4)
    ax2.axhline(20,color=C['sup'],lw=0.5,ls='--',alpha=0.4)
    ax2.set_ylim(-10,110)
    for i in range(1,len(df)):
        if df['kv'].iloc[i-1]<df['dv'].iloc[i-1] and df['kv'].iloc[i]>=df['dv'].iloc[i] and df['kv'].iloc[i]<30:
            ax2.scatter(i,df['kv'].iloc[i],marker='^',color=C['sup'],s=40,zorder=5)
        elif df['kv'].iloc[i-1]>df['dv'].iloc[i-1] and df['kv'].iloc[i]<=df['dv'].iloc[i] and df['kv'].iloc[i]>70:
            ax2.scatter(i,df['kv'].iloc[i],marker='v',color=C['res'],s=40,zorder=5)
    ax2.legend(loc='upper left',fontsize=7,facecolor=PNL,edgecolor=GRD,labelcolor=C['lbl'])
    ax2.set_title(f'{name} KDJ',fontsize=11,color=C['txt'],pad=5,loc='left')
    sty(ax2,'KDJ'); ax2.set_xlim(-1,len(df))
    # RSI
    ax3=axes[2]
    ax3.plot(x,df['rsi6'],color=C['r6'],lw=1.2,label='RSI6')
    ax3.plot(x,df['rsi12'],color=C['r12'],lw=1.2,label='RSI12')
    ax3.axhspan(70,100,alpha=0.08,color=C['res'])
    ax3.axhspan(0,30,alpha=0.08,color=C['sup'])
    ax3.axhline(70,color=C['res'],lw=0.5,ls='--',alpha=0.4)
    ax3.axhline(30,color=C['sup'],lw=0.5,ls='--',alpha=0.4)
    ax3.axhline(50,color=C['lbl'],lw=0.3,ls=':')
    ax3.set_ylim(10,90)
    if len(df)>20:
        p_hi,p_lo=find_pivots(df,order=8)
        rsi6=df['rsi6']
        for j in range(1,len(p_hi)):
            i1,i2=p_hi[j-1],p_hi[j]
            if i2<len(rsi6) and i1<len(rsi6):
                if df['close'].iloc[i2]>df['close'].iloc[i1] and rsi6.iloc[i2]<rsi6.iloc[i1]:
                    ax3.annotate('顶背离',xy=(i2,rsi6.iloc[i2]),fontsize=6,color=C['res'],fontweight='bold',ha='center')
        for j in range(1,len(p_lo)):
            i1,i2=p_lo[j-1],p_lo[j]
            if i2<len(rsi6) and i1<len(rsi6):
                if df['close'].iloc[i2]<df['close'].iloc[i1] and rsi6.iloc[i2]>rsi6.iloc[i1]:
                    ax3.annotate('底背离',xy=(i2,rsi6.iloc[i2]),fontsize=6,color=C['sup'],fontweight='bold',ha='center')
    ax3.legend(loc='upper left',fontsize=7,facecolor=PNL,edgecolor=GRD,labelcolor=C['lbl'])
    ax3.set_title(f'{name} RSI',fontsize=11,color=C['txt'],pad=5,loc='left')
    sty(ax3,'RSI',sx=True); ax3.set_xlim(-1,len(df)); dticks(ax3,df)
    fig.suptitle(f'{name} ({code}) 指标面板',fontsize=14,fontweight='bold',color=C['txt'],y=0.98)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(out,dpi=150,bbox_inches='tight',facecolor=BG); plt.close()
    print(f'[4/4] indicators: {out}')

# ═══ MAIN ═══
if __name__=='__main__':
    if len(sys.argv)<5:
        print('Usage: python3 advanced_charts.py <kline_json> <name> <code> <output_dir> [bench_json]')
        sys.exit(1)
    kline_path=sys.argv[1]; name=sys.argv[2]; code=sys.argv[3]; outdir=sys.argv[4]
    bench_path=sys.argv[5] if len(sys.argv)>5 else None
    os.makedirs(outdir,exist_ok=True)
    df=parse_kline(kline_path)
    print(f'Loaded {len(df)} klines: {df["date"].iloc[0]} ~ {df["date"].iloc[-1]}')
    bench=parse_kline(bench_path) if bench_path else None
    r1=chart_main(df,name,code,os.path.join(outdir,f'{code}_main.png'))
    chart_multi(df,name,code,os.path.join(outdir,f'{code}_multi_tf.png'))
    chart_rs_vol(df,name,code,os.path.join(outdir,f'{code}_rs_vol.png'),bench)
    chart_indicators(df,name,code,os.path.join(outdir,f'{code}_indicators.png'))
    analysis=dict(code=code,name=name,
        last_close=float(df.iloc[-1]['close']),
        last_date=str(df.iloc[-1]['date'].date()),
        volume_profile=dict(poc=r1['poc'],vah=r1['vah'],val=r1['val']),
        patterns=r1['patterns'],candle_patterns=r1['candle_pats'],
        support=r1['pivots_lo'][-3:] if r1['pivots_lo'] else [],
        resistance=r1['pivots_hi'][-3:] if r1['pivots_hi'] else [])
    with open(os.path.join(outdir,f'{code}_analysis.json'),'w') as f:
        json.dump(analysis,f,ensure_ascii=False,indent=2)
    print(f'Done! {json.dumps(analysis,ensure_ascii=False)}')
