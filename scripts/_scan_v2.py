import sys, json, statistics, concurrent.futures, time
sys.path.insert(0,'scripts')
from fintool_client import get_constituent_snapshot, get_kline, get_snapshot
from collections import OrderedDict

# 1) load snapshot of index constituents
all_s = []
for p in range(1,11):
    all_s.extend(get_constituent_snapshot('中证1000', page=p, page_size=100))
print(f'loaded {len(all_s)} snapshots', flush=True)

snap = {}   # code -> {name, price, mv}
cands_raw = []
for s in all_s:
    code  = s.get('security_code','')
    name  = (s.get('security_name','') or '').strip()
    if 'ST' in name: continue
    price = float(s.get('last_price',0) or 0)
    mv    = float(s.get('market_value',0) or 0)
    if price <= 0 or mv < 5e9: continue
    snap[code] = {'name': name, 'price': price, 'mv': mv}
    abs_chg = abs(float(s.get('price_change_rate',0) or 0))
    cands_raw.append((abs_chg, code, name, mv))
cands_raw.sort(reverse=True)
top80 = [(c[1],c[2],c[3]) for c in cands_raw[:80]]
print(f'top-80 by |chg|={cands_raw[0][0]:.3f}~{cands_raw[79][0]:.3f}', flush=True)

END='2026-05-22'; START='2026-04-18'

def worker(item):
    code, name, mv = item
    try:
        hist = get_kline(code, START, END, num=30) or []
    except Exception as e:
        return None
    seen=OrderedDict()
    for r in hist: seen[r.get('trade_date','')]=r
    rows=list(seen.values())[::-1]
    if len(rows)<10: return None
    closes=[float(r.get('close',r.get('last_price',0)) or 0) for r in rows]
    amts  =[float(r.get('amount',0) or 0) for r in rows]
    prices=[float(r.get('close',r.get('last_price',0)) or 0) for r in rows]
    rets  =[(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes)) if closes[i-1]>0]
    if len(rets)<10: return None
    vol20=statistics.pstdev(rets)*(252**0.5)*100
    # turnover proxy: amount / (shares = mv/price) ≈ amount*price/mv; clamp to 0..0.30
    tos=[]
    for a,p in zip(amts,prices):
        if a<=0 or p<=0: continue
        sh = mv/p
        if sh<=0: continue
        tos.append(min(0.30, max(0.0, a*a/sh/mv if 0 else 0)))  # placeholder
        tos[-1] = min(0.30, max(0.0, a/sh/mv))
    if not tos: return None
    to5 = sum(tos[-5:])/min(len(tos[-5:]),5)*100
    return {'code': code, 'name': name,
            'vol20':   round(vol20,2),
            'to5':     round(to5,2),
            'method':  'snapshot_ratio',
            'mv_yi':   round(mv/1e8,2),
            'last_close': closes[-1],
            'last_date': rows[-1].get('trade_date','')}

t0=time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
    res=list(filter(None, ex.map(worker, top80)))
print(f'kline-screened {len(res)} in {time.time()-t0:.1f}s', flush=True)
res.sort(key=lambda x: x['vol20'], reverse=True)
for r in res[:20]:
    print(json.dumps(r, ensure_ascii=False))
