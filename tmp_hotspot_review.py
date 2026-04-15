#!/usr/bin/env python3
"""Hotspot close review for 2026-04-15"""
import json

# Intraday closing prices from Tencent
quotes_raw = {
    '002866': {'name':'传艺科技','close':23.62,'pct':-1.25},
    '002881': {'name':'美格智能','close':38.66,'pct':-0.69},
    '688100': {'name':'威胜信息','close':38.20,'pct':2.52},
    '300007': {'name':'汉威科技','close':43.47,'pct':-2.01},
}

with open('paper-trading-data.json') as f:
    data = json.load(f)
hotspot = data['players']['hotspot']
portfolio = hotspot['portfolio']

total_mv = 0
total_pnl = 0
results = []

print('=== 收盘盈亏汇总 (2026-04-15) ===')
for code, pos in portfolio['positions'].items():
    bare = code.split('.')[0]
    q = quotes_raw.get(bare, {})
    close = q.get('close', pos['current_price'])
    pct_today = q.get('pct', 0)
    name = pos['name']
    vol = pos['volume']
    pnl_total = pos['pnl']
    pnl_pct = pos['pnl_pct']
    mv = close * vol
    total_mv += mv
    total_pnl += pnl_total
    sign = '+' if pct_today >= 0 else ''
    print('  {:10s} {:6s} {:4d}股 现价{:<8.2f} 今日{}{:>+.2f}% 累计{:.2f}% 盈亏{:+.0f}'.format(
        code, name, vol, close, sign, pct_today, pnl_pct, pnl_total))
    results.append({
        'code': code, 'name': name, 'close': close,
        'pct_today': pct_today, 'pnl_pct': pnl_pct, 'pnl': pnl_total,
        'mv': mv, 'vol': vol
    })

cash = portfolio['cash']
total_assets = total_mv + cash
pos_pct = total_mv / total_assets * 100

print('\n  现金: {:,.0f}'.format(cash))
print('  持仓市值: {:,.0f}'.format(total_mv))
print('  总资产: {:,.0f}'.format(total_assets))
print('  仓位: {:.1f}%'.format(pos_pct))
print('  累计总盈亏: {:+,.0f}'.format(total_pnl))

# 风控检查
print('\n=== 风控检查 ===')
trades_needed = []
for r in results:
    code = r['code']
    name = r['name']
    pnl_pct = r['pnl_pct']
    entry = portfolio['positions'][code].get('entry_date','?')
    
    if pnl_pct >= 8:
        note = '盈利{:.2f}% 接近止盈(10%)，关注冲高回落'.format(pnl_pct)
        print('  [⚠️] {} {} {}'.format(code, name, note))
    elif pnl_pct <= -5:
        note = '亏损{:.2f}% 触发止损(-5%)，建议收盘前减仓'.format(pnl_pct)
        print('  [🚨] {} {} {}'.format(code, name, note))
        trades_needed.append({
            'code': code, 'name': name,
            'direction': 'sell', 'volume': portfolio['positions'][code]['volume'],
            'reason': '触发止损({:.2f}%)'.format(pnl_pct)
        })
    elif pnl_pct <= -3:
        note = '亏损{:.2f}% 接近止损'.format(pnl_pct)
        print('  [⚠️] {} {} {}'.format(code, name, note))
    else:
        status = '🟢' if pnl_pct > 0 else '🟡'
        print('  {} {} {} 盈亏{:.2f}% (入场{})'.format(status, code, name, pnl_pct, entry))

# Save summary for decision
summary = {
    'date': '2026-04-15',
    'player': 'hotspot',
    'account_value': total_assets,
    'cash': cash,
    'position_pct': round(pos_pct, 1),
    'positions': results,
    'total_pnl': total_pnl,
    'trades_needed': trades_needed,
    'market': {
        '上证': '+0.01%',
        '深证': '-0.97%', 
        '创业板': '-1.22%',
        '科创50': '+0.09%'
    }
}

with open('/tmp/hotspot_close_review.json', 'w') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print('\n✅ 数据已保存到 /tmp/hotspot_close_review.json')
EOF
