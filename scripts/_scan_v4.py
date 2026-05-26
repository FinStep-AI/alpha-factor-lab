import sys, json, statistics, concurrent.futures, time
sys.path.insert(0,'scripts')
from fintool_client import get_constituent_snapshot, get_kline
from collections import OrderedDict

all_s = []
for p in range(1,11):
    all_s.extend(get_constituent_snapshot('中证1000', page=p, page_size=100))
print(f'loaded {len(all_s)} snapshots', flush=True)

snap = {}
cands_raw = []
for s in all_s:
    code  = s.get('security_code','')
    name  = (s.get('security_name','') or '').strip()
    if 'ST' in name: continue
    price = float(s.get('last_price',0) or 0)
    mv    = float(s.get('market_value',0) or 0)
    if price <= 0 or mv < 5e9: continue
    snap[code] = {'name': name, 'price': price, 'mv': mv, 'chg': float(s.get('price_change_rate',0) or 0)}
    cands_raw.append((abs(float(s.get('price_change_rate',0) or 0)), code, name, mv))
cands_raw.sort(reverse=True)
top80=[(c[1],c[2],c[3]) for c in cands_raw[:80]]
print(f'pool={len(cands_raw)} top-80 from', cands_raw[0][0], cands_raw[-1][0], flush=True)

END='2026-05-22'; START='2026-04-01'

def worker(item):
    code, name, mv_snap = item
    sp = snap.get(code, {})
    p0 = float(sp.get('price', 0) or 0)
    # p0 must be >0 and mv_snap must be >0 from snapshot (not mv=0 fintool field)
    if p0 <= 0 or mv_snap <= 0:
        return {'code': code, 'name': name, '_reason': 'bad_price_or_mv', 'p0': p0, 'mv': mv}
    try:
        hist = get_kline(code, START, END, num=60) or []
    except Exception as e:
        return {'code': code, 'name': name, '_reason': 'exc', 'err': str(e)}
    seen=OrderedDict()
    for r in hist: seen[r.get('trade_date','')]=r
    rows=list(seen.values())[::-1]           # chronological
    if len(rows)<10:
        return {'code': code, 'name': name, '_reason': 'lt_10', 'rows': len(rows)}
    closes=[float(r.get('close',0) or 0) for r in rows]
    amts  =[float(r.get('amount', 0) or 0) for r in rows]
    rets  =[(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes)) if closes[i-1]>0]
    if len(rets)<10:
        return {'code': code, 'name': name, '_reason': 'rets_lt10', 'rets': len(rets)}
    vol20=statistics.pstdev(rets)*(252**0.5)*100
    tos=[a/mv_snap for a in amts[-10:]]
    to5=sum(tos[-5:])/min(len(tos[-5:]),5)*100
    return {'code': code, 'name': name,
            'vol20': round(vol20,2), 'to5': round(to5,2),
            'mv_yi': round(mv_snap/1e8,2),
            'last_close': closes[-1], 'last_date': rows[-1].get('trade_date',''),
            'snap_chg': round(float(sp.get('chg',0) or 0),4)}

def score(r):
    return r['vol20']*0.6 + r['to5']*0.4

t0=time.time()
allres=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
    allres=list(filter(lambda x: x is not None, ex.map(worker, top80)))
ok  = [r for r in allres if '_reason' not in r]
bad = [r for r in allres if '_reason' in r]
print(f'done {time.time()-t0:.1f}s ok={len(ok)} bad={len(bad)}', flush=True)
if bad:
  from collections import Counter
  c=Counter(r['_reason'] for r in bad)
  print('fail reasons', dict(c), flush=True)
ok.sort(key=score, reverse=True)
for r in ok[:20]:
    print(json.dumps(r, ensure_ascii=False))
with open('/tmp/_top_vol.json','w') as f:
    json.dump(ok, f, ensure_ascii=False)
print('saved /tmp/_top_vol.json ok_rows=', len(ok))
