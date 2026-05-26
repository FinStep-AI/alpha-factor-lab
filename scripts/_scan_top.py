import sys, json, concurrent.futures, statistics, time
sys.path.insert(0,'scripts')
from fintool_client import get_constituent_snapshot, get_kline
from collections import OrderedDict

all_s = []
for p in range(1,11):
    all_s.extend(get_constituent_snapshot('中证1000', page=p, page_size=100))

print(f'loaded {len(all_s)}', flush=True)
cands = []
for s in all_s:
    name=(s.get('security_name','') or '').strip()
    if 'ST' in name: continue
    mv=float(s.get('market_value',0) or 0)
    if mv < 5e9: continue
    cands.append((abs(float(s.get('price_change_rate',0) or 0)), s.get('security_code',''), name, mv))
cands.sort(reverse=True)
top = [(c[1],c[2],c[3]) for c in cands[:80]]
print(f'top-80 candidates by |chg|', flush=True)

END='2026-05-22'; START='2026-04-18'

def worker(item):
    code, name, mv = item
    try:
        hist = get_kline(code, START, END, num=30) or []
    except Exception:
        return None
    seen=OrderedDict()
    for r in hist: seen[r.get('trade_date','')]=r
    rows=list(seen.values())[::-1]
    if len(rows)<10: return None
    closes=[float(r.get('close',r.get('last_price',0)) or 0) for r in rows]
    amt=[float(r.get('total_amount',r.get('trade_balance',0)) or 0) for r in rows]
    mv_s=[float(r.get('market_value',0) or 0) for r in rows]
    rets=[(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes)) if closes[i-1]>0]
    if len(rets)<10: return None
    vol20=statistics.pstdev(rets)*(252**0.5)*100
    tos=[]
    for i in range(len(rows)):
        a=amt[i]
        m=mv_s[max(0,i)] if mv_s[max(0,i)]>0 else mv
        tos.append(a/m if m>0 else 0)
    to5=sum(tos[-5:])/max(len(tos[-5:]),1)*100
    return {'code': code, 'name': name, 'vol20': round(vol20,2), 'to5': round(to5,2),
            'mv_yi': round(mv/1e8,2)}

t0=time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
    res=list(filter(None, ex.map(worker, top)))
print(f'done in {time.time()-t0:.1f}s, got {len(res)}', flush=True)
res.sort(key=lambda x: x['vol20'], reverse=True)
for r in res[:20]:
    print(json.dumps(r, ensure_ascii=False))
