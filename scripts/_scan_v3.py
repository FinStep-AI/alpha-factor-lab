import sys, json, statistics, concurrent.futures, time
sys.path.insert(0,'scripts')
from fintool_client import get_constituent_snapshot, get_kline
from collections import OrderedDict

# 1) snapshot 1000
all_s = []
for p in range(1,11):
    all_s.extend(get_constituent_snapshot('中证1000', page=p, page_size=100))
print(f'loaded {len(all_s)} snapshots', flush=True)

# keep: not-ST, mv>=50亿, price>0
snap = {}
cands_raw = []
for s in all_s:
  code = s.get('security_code','')
  name = (s.get('security_name','') or '').strip()
  if 'ST' in name: continue
  price= float(s.get('last_price',0) or 0)
  mv   = float(s.get('market_value',0) or 0)
  if price <= 0 or mv < 5e9: continue
  snap[code] = {'name': name, 'price': price, 'mv': mv, 'chg': float(s.get('price_change_rate',0) or 0)}
  cands_raw.append((abs(float(s.get('price_change_rate',0) or 0)), code, name, mv))
cands_raw.sort(reverse=True)
top80 = [(c[1],c[2],c[3]) for c in cands_raw[:80]]
print(f'top-80 by |chg|={cands_raw[0][0]:.3f}~{cands_raw[79][0]:.3f}', flush=True)

END='2026-05-22'; START='2026-04-01'

def worker(item):
    code, name, mv = item
    sp = snap.get(code,{})
    price_snap = sp.get('price', 0) or 0
    try:
        hist = get_kline(code, START, END, num=60) or []   # num=60 to ensure enough
    except Exception:
        return None
    # dedup by date, keep order
    seen=OrderedDict()
    for r in hist: seen[r.get('trade_date','')]=r
    rows=list(seen.values())                     # fintool returns newest-first in our earlier tests
    if len(rows) < 10:
        return None
    # reverse to chronological (old->new)
    rows = list(reversed(rows))
    closes=[float(r.get('close',r.get('last_price',0)) or 0) for r in rows]
    amts  =[float(r.get('amount',0) or 0)          for r in rows]
    # -> cut to last 10 rows we have market_value for (here, snapshot is current-day)
    # Use latest snapshot price/mv to estimate shares outstanding for the whole window.
    if price_snap <= 0 or mv <= 0:
        return None
    # returns
    rets=[(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes)) if closes[i-1]>0]
    if len(rets) < 10:
        return None
    vol20=statistics.pstdev(rets)*(252**0.5)*100
    # turnover proxy for last 5 trading days:
    # shares_in_circulation ≈ mv(亿) × 1e8 / price(复权); turnover_i ≈ amount_i / (shares * price_i)
    # use snapshot MV as proxy for shares*price:
    tos=[]
    for a in amts[-10:]:
        tos.append(min(0.40, max(0.0, a / (mv))))
    to5 = sum(tos[-5:])/min(len(tos[-5:]),5)*100 if tos else 0
    return {
      'code': code,
      'name': name,
      'vol20': round(vol20,2),
      'to5': round(to5,2),
      'mv_yi': round(mv/1e8,2),
      'last_close': closes[-1],
      'last_date':  rows[-1].get('trade_date',''),
      'chg': round(sp.get('chg',0),4),
    }

t0=time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
    res=list(filter(None, ex.map(worker, top80)))
print(f'done {time.time()-t0:.1f}s, got {len(res)}', flush=True)
res.sort(key=lambda x: (x['vol20']*0.6 + x['to5']*0.4), reverse=True)
for r in res[:20]:
    print(json.dumps(r, ensure_ascii=False))
# persist best
with open('/tmp/_top_vol.json','w') as f:
    json.dump(res[:20], f, ensure_ascii=False)
print('saved /tmp/_top_vol.json')
