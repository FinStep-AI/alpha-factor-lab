#!/usr/bin/env python3
"""午盘技术分析 - 三标的全指标计算"""
import json, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# === 读取K线 ===
def load_klines(f):
    with open(f) as fh: return json.load(fh)

def calc_ma(closes, period):
    if len(closes) < period: return [None]*len(closes)
    return [sum(closes[i-period+1:i+1])/period for i in range(len(closes))]

def calc_ema(data, period):
    k = 2/(period+1)
    ema = [data[0]]
    for p in data[1:]:
        ema.append(p*k + ema[-1]*(1-k))
    return ema

def calc_macd(closes, fast=12, slow=26, signal=9):
    ef, es = calc_ema(closes, fast), calc_ema(closes, slow)
    dif = [f-s for f,s in zip(ef, es)]
    dea = calc_ema(dif, signal)
    hist = [d-e for d,e in zip(dif, dea)]
    return dif, dea, hist

def calc_kdj(highs, lows, closes, n=9, m1=3, m2=3):
    k_l, d_l, j_l = [], [], []
    rsv_l = []
    for i in range(len(closes)):
        s = max(0, i-n+1)
        hh=max(highs[s:i+1]); ll=min(lows[s:i+1])
        rsv = 50 if hh==ll else (closes[i]-ll)/(hh-ll)*100
        rsv_l.append(rsv)
    k, d = 50.0, 50.0
    for r in rsv_l:
        k = (k*(m1-1)+r)/m1; d = (d*(m2-1)+k)/m2
        k_l.append(k); d_l.append(d); j_l.append(3*k-2*d)
    return k_l, d_l, j_l

def calc_rsi(closes, period=6):
    rsi=[50.0]*len(closes)
    for i in range(period, len(closes)):
        g=[max(0,closes[j]-closes[j-1]) for j in range(i-period+1,i+1)]
        l=[max(0,closes[j-1]-closes[j]) for j in range(i-period+1,i+1)]
        ag,al = sum(g)/period, sum(l)/period
        rsi[i] = 100 if al==0 else 100-100/(1+ag/al)
    return rsi

def calc_rsi_list(closes, periods=[6,12]):
    return {p: calc_rsi(closes,p) for p in periods}

def calc_boll(closes, period=20, stds=2):
    mid=calc_ma(closes,period); up,dn=[],[]
    for i in range(len(closes)):
        if i<period-1: up.append(None); dn.append(None)
        else: s=np.std(closes[i-period+1:i+1]); up.append(mid[i]+stds*s); dn.append(mid[i]-stds*s)
    return up, mid, dn

def calc_obv(closes, vols):
    obv=[0.0]
    for i in range(1,len(closes)):
        if closes[i]>closes[i-1]: obv.append(obv[-1]+vols[i])
        elif closes[i]<closes[i-1]: obv.append(obv[-1]-vols[i])
        else: obv.append(obv[-1])
    return obv

def avg_vol(vols, days=10):
    if len(vols)>=days: return sum(vols[-days:])/days
    return sum(vols)/max(len(vols),1)

def analyze_stock(code, name, klines, entry_info=None):
    closes=[r["close"] for r in klines]
    highs=[r["high"] for r in klines]
    lows=[r["low"] for r in klines]
    vols=[r["volume"] for r in klines]
    dates=[r["date"] for r in klines]
    n=len(closes); i=n-1
    price=closes[i]

    ma5=calc_ma(closes,5)[i]; ma10=calc_ma(closes,10)[i]; ma20=calc_ma(closes,20)[i]
    ma60=calc_ma(closes,60)[i] if n>=60 else None; ma120=calc_ma(closes,120)[i] if n>=120 else None
    dif,dea,hist=calc_macd(closes)
    k,d,j=calc_kdj(highs,lows,closes)
    rsi6=calc_rsi(closes,6)[i]; rsi12=calc_rsi(closes,12)[i]
    b_upper,b_mid,b_lower=calc_boll(closes)
    obv=calc_obv(closes,vols)
    v5=avg_vol(vols,5); v10=avg_vol(vols,10); v20=avg_vol(vols,20)
    obv5=obv[-1]-obv[-6] if n>=6 else obv[-1]-obv[0]
    
    boll_pct=(price-b_lower[i])/(b_upper[i]-b_lower[i])*100 if b_upper[i] and b_lower[i] else None
    
    # === 综合打分 ===
    score=0; breakdown={}
    
    # 趋势 (+30)
    if ma5 and ma10 and ma20:
        if ma5>ma10>ma20:
            score+=30; breakdown["趋势"]=(30,"均线多头排列")
        elif ma5>ma10:
            score+=15; breakdown["趋势"]=(15,"均线短多")
        elif ma5<ma10<ma20:
            score-=20; breakdown["趋势"]=(-20,"均线空头")
        else:
            breakdown["趋势"]=(0,"均线混乱")
    
    # MACD (+30)
    if hist[i]:
        if dif[i]>dea[i] and hist[i]>0:
            score+=30; breakdown["MACD"]=(30,"金叉+红柱扩大")
        elif dif[i]>dea[i] and hist[i]<0:
            score+=10; breakdown["MACD"]=(10,"金叉绿柱缩窄")
        elif dif[i]<dea[i] and hist[i]>0:
            score-=10; breakdown["MACD"]=(-10,"死叉红柱缩窄")
        else:
            score-=25; breakdown["MACD"]=(-25,"死叉+绿柱")
    
    # KDJ (+20)
    if j[i]:
        if j[i]>100: score+=0; breakdown["KDJ"]=(0,f"J={j[i]:.1f}超买")
        elif j[i]>80: score+=0; breakdown["KDJ"]=(0,f"J={j[i]:.1f}偏高")
        elif j[i]<20: score+=20; breakdown["KDJ"]=(20,f"J={j[i]:.1f}超卖")
        elif j[i]<40: score+=10; breakdown["KDJ"]=(10,f"J={j[i]:.1f}偏低")
        else: score+=5; breakdown["KDJ"]=(5,f"J={j[i]:.1f}中性")
    
    # RSI (+15)
    if rsi6:
        if rsi6>85: score-=15; breakdown["RSI"]=(-15,f"RSI6={rsi6:.1f}极端超买")
        elif rsi6>75: score-=10; breakdown["RSI"]=(-10,f"RSI6={rsi6:.1f}超买")
        elif rsi6>65: score-=3; breakdown["RSI"]=(-3,f"RSI6={rsi6:.1f}偏高")
        elif rsi6<20: score+=15; breakdown["RSI"]=(15,f"RSI6={rsi6:.1f}超卖")
        elif rsi6<30: score+=10; breakdown["RSI"]=(10,f"RSI6={rsi6:.1f}偏低")
        elif rsi6<50: score+=5; breakdown["RSI"]=(5,f"RSI6={rsi6:.1f}中性偏弱")
        else: score+=5; breakdown["RSI"]=(5,f"RSI6={rsi6:.1f}中性偏强")
    
    # BOLL (+15)
    if boll_pct is not None:
        if boll_pct>95: score+=0; breakdown["BOLL"]=(0,f"位置={boll_pct:.0f}%极端超买")
        elif boll_pct>80: score-=10; breakdown["BOLL"]=(-10,f"位置={boll_pct:.0f}%偏上超买")
        elif boll_pct>60: score+=0; breakdown["BOLL"]=(0,f"位置={boll_pct:.0f}%偏上正常")
        elif boll_pct<5: score+=15; breakdown["BOLL"]=(15,f"位置={boll_pct:.0f}%极端超卖")
        elif boll_pct<20: score+=10; breakdown["BOLL"]=(10,f"位置={boll_pct:.0f}%偏下低位")
        else: score+=5; breakdown["BOLL"]=(5,f"位置={boll_pct:.0f}%中部")
    
    # 量能 (+10)
    if v20>0:
        vr=vols[i]/v20
        if vr>2.5: score+=8; breakdown["量能"]=(8,f"巨量(vr={vr:.1f}x)")
        elif vr>1.8: score+=5; breakdown["量能"]=(5,f"放量(vr={vr:.1f}x)")
        elif vr<0.4: score-=5; breakdown["量能"]=(-5,f"缩量(vr={vr:.1f}x)")
        else: breakdown["量能"]=(0,f"量能正常(vr={vr:.1f}x)")
    
    # OBV (+10)
    if obv5>0 and obv[-1]>0: score+=5; breakdown["OBV"]=(5,"5日OBV流入")
    elif obv5<0: score-=5; breakdown["OBV"]=(-5,"5日OBV流出")
    else: breakdown["OBV"]=(0,"OBV走平")
    
    # 风控计算
    if entry_info:
        cost=entry_info["avg_cost"]; pnl_pct=(price-cost)/cost*100
        stop_loss=cost*0.93; take_profit=cost*1.20
        position=entry_info.get("volume",0)
    else:
        cost=None; pnl_pct=None; stop_loss=None; take_profit=None; position=0
    
    return {
        "code":code,"name":name,"date":dates[i],"price":price,
        "score":score,"breakdown":breakdown,
        "price_vs": {
            "ma5":round(price-ma5,2) if ma5 else None,
            "ma10":round(price-ma10,2) if ma10 else None,
            "ma20":round(price-ma20,2) if ma20 else None,
            "boll_up":round(b_upper[i],2) if b_upper[i] else None,
            "boll_mid":round(b_mid[i],2) if b_mid[i] else None,
            "boll_dn":round(b_lower[i],2) if b_lower[i] else None,
            "boll_pct":round(boll_pct,1) if boll_pct else None,
        },
        "indicators": {
            "macd":{"dif":round(dif[i],4),"dea":round(dea[i],4),"hist":round(hist[i],4)} if hist[i] else {},
            "kdj":{"k":round(k[i],1),"d":round(d[i],1),"j":round(j[i],1)},
            "rsi":{"rsi6":round(rsi6,1),"rsi12":round(rsi12,1)},
        },
        "volume":{"today":int(vols[i]),"avg5":int(v5),"avg10":int(v10),"avg20":int(v20),"vr_ratio":round(vols[i]/v20,2) if v20>0 else None,"obv5":round(obv5/1e4,1)},
        "risk": {"pnl_pct":round(pnl_pct,2) if pnl_pct else None,"stop_loss":round(stop_loss,2) if stop_loss else None,"take_profit":round(take_profit,2) if take_profit else None,"trigger_stop_loss":False,"trigger_take_profit":False},
        "recent_signal":None,
    }

# === 主程序 ===
BASE="/tmp/klines_fq"
positions={
    "000833.SZ":{"name":"粤桂股份","avg_cost":26.01,"volume":200,"entry_date":"2026-04-24","file":"sz000833.json"},
    "603501.SH":{"name":"豪威集团","avg_cost":104.63,"volume":45000,"entry_date":"2026-04-27","file":"sh603501.json"},
    "000681.SZ":{"name":"视觉中国","avg_cost":22.50,"volume":226000,"entry_date":"2026-04-30","file":"sz000681.json"},
}

results=[]
for code, info in positions.items():
    klines=load_klines(os.path.join(BASE,info["file"]))
    r=analyze_stock(code,info["name"],klines,info)
    
    risk=r["risk"]
    if risk["pnl_pct"] is not None:
        r["risk"]["trigger_stop_loss"] = risk["pnl_pct"] <= -7
        r["risk"]["trigger_take_profit"] = risk["pnl_pct"] >= 20
    
    results.append(r)
    print(f"\n{'='*55}")
    print(f"📊 {r['name']} ({r['code']})  {r['date']}")
    print(f"   现价={r['price']}  | 综合打分={r['score']}")
    if risk["pnl_pct"] is not None:
        print(f"   浮盈={risk['pnl_pct']:+.2f}%  | 止损={risk['stop_loss']}  | 止盈={risk['take_profit']}")
        print(f"   {'🔴 触发止损！' if risk['trigger_stop_loss'] else ('🟢 触发止盈！' if risk['trigger_take_profit'] else '✅ 未触发风控')}")
    print(f"   MA5={r['price_vs']['ma5']:+} MA10={r['price_vs']['ma10']:+} MA20={r['price_vs']['ma20']:+}")
    print(f"   KDJ={r['indicators']['kdj']['k']}/{r['indicators']['kdj']['d']}/{r['indicators']['kdj']['j']}  RSI={r['indicators']['rsi']['rsi6']}  BOLL%={r['price_vs']['boll_pct']}%")
    print(f"   MACD hist={r['indicators']['macd'].get('hist','?')}  VR={r['volume']['vr_ratio']}x  OBV5={r['volume']['obv5']}万")
    print(f"   打分拆解：")
    for k,v in r["breakdown"].items(): print(f"     {k}: {v}")

# 保存结果
with open("/tmp/trader_analysis.json","w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n\n✅ 技术分析完成，结果→/tmp/trader_analysis.json")

PYEOF