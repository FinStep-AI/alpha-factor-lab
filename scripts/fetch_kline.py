#!/usr/bin/env python3
"""通用A股K线获取（支持 fintool/聚源 + 腾讯 + 东方财富 三数据源）
优先级: fintool > tencent > eastmoney
"""
import urllib.request
import json
import time
import sys
import os

def fetch_kline_fintool(code, days=120, end_date=None):
    """fintool K线接口（聚源数据源，优先级最高）
    code: 6位纯数字代码
    days: 条数（最多100/次，自动分段）
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fintool_client import get_kline, get_kline_history
    if end_date is None:
        from datetime import date
        end_date = date.today().strftime('%Y-%m-%d')
    
    if days <= 100:
        return get_kline(code, end_date, days)
    else:
        # 超过100条用分段拉取
        from datetime import date, timedelta
        dt = date.today()
        start = (dt - timedelta(days=int(days * 1.5))).strftime('%Y-%m-%d')
        return get_kline_history(code, start, end_date)


def fetch_kline_tencent(code, days=120):
    """腾讯K线接口（web.ifzq.gtimg.cn）
    code: 6位纯数字代码或带前缀的格式(sh/sz)
    """
    if code.startswith(('sh', 'sz')):
        qq_code = code
    else:
        # 自动判断交易所
        if code.startswith(('6', '9')):
            qq_code = f'sh{code}'
        else:
            qq_code = f'sz{code}'
    
    url = f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={qq_code},day,,,{days},qfq'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=15)
    raw = resp.read().decode('utf-8')
    json_str = raw.split('=', 1)[1] if '=' in raw else raw
    data = json.loads(json_str)
    
    stock_data = data.get('data', {}).get(qq_code, {})
    klines = stock_data.get('qfqday', stock_data.get('day', []))
    
    results = []
    for kl in klines:
        if len(kl) >= 6:
            results.append({
                'date': kl[0],
                'open': float(kl[1]),
                'close': float(kl[2]),
                'high': float(kl[3]),
                'low': float(kl[4]),
                'volume': float(kl[5]),
            })
    return results


def fetch_kline_eastmoney(code, days=120):
    """东方财富K线接口（push2his.eastmoney.com）"""
    if code.startswith(('sh', 'sz')):
        market = '1' if code.startswith('sh') else '0'
        pure_code = code[2:]
    else:
        pure_code = code
        market = '1' if code.startswith(('6', '9')) else '0'
    
    secid = f'{market}.{pure_code}'
    url = f'https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&fields1=f1,f2,f3,f4,f5,f6,f7,f8&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101&fqt=1&end=20261231&lmt={days}'
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
        'Referer': 'https://quote.eastmoney.com/',
    })
    resp = urllib.request.urlopen(req, timeout=15)
    raw = resp.read().decode('utf-8')
    data = json.loads(raw)
    
    if not data.get('data') or not data['data'].get('klines'):
        return []
    
    results = []
    for kl in data['data']['klines']:
        parts = kl.split(',')
        if len(parts) >= 6:
            results.append({
                'date': parts[0],
                'open': float(parts[1]),
                'close': float(parts[2]),
                'high': float(parts[3]),
                'low': float(parts[4]),
                'volume': float(parts[5]),
            })
    return results


def fetch_kline(code, days=120, source='auto'):
    """获取K线，支持 auto/fintool/tencent/eastmoney
    auto: fintool → tencent → eastmoney
    """
    if source == 'fintool':
        return fetch_kline_fintool(code, days)
    elif source == 'tencent':
        return fetch_kline_tencent(code, days)
    elif source == 'eastmoney':
        return fetch_kline_eastmoney(code, days)
    else:  # auto: fintool → tencent → eastmoney
        try:
            result = fetch_kline_fintool(code, days)
            if result:
                return result
        except Exception:
            pass
        try:
            result = fetch_kline_tencent(code, days)
            if result:
                return result
        except Exception:
            pass
        try:
            result = fetch_kline_eastmoney(code, days)
            if result:
                return result
        except Exception:
            pass
        return []


def fetch_realtime(codes):
    """批量获取实时行情（腾讯接口）
    codes: 纯数字代码列表 或 sh/sz前缀列表
    """
    qq_codes = []
    for code in codes:
        if code.startswith(('sh', 'sz')):
            qq_codes.append(code)
        elif code.startswith(('6', '9')):
            qq_codes.append(f'sh{code}')
        else:
            qq_codes.append(f'sz{code}')
    
    url = f'https://qt.gtimg.cn/q={",".join(qq_codes)}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=15)
    raw = resp.read().decode('gbk')
    
    results = {}
    for line in raw.strip().split(';'):
        line = line.strip()
        if not line or '=' not in line:
            continue
        val = line.split('=')[1].strip('"')
        parts = val.split('~')
        if len(parts) < 35:
            continue
        
        code = parts[2]
        results[code] = {
            'name': parts[1],
            'code': code,
            'price': float(parts[3]) if parts[3] else 0,
            'prev_close': float(parts[4]) if parts[4] else 0,
            'open': float(parts[5]) if parts[5] else 0,
            'change_pct': float(parts[32]) if parts[32] else 0,
            'high': float(parts[33]) if parts[33] else 0,
            'low': float(parts[34]) if parts[34] else 0,
            'pe': float(parts[39]) if parts[39] else 0,
            'pb': float(parts[46]) if parts[46] else 0,
        }
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='A股K线获取')
    parser.add_argument('code', help='股票代码(6位或带sh/sz前缀)')
    parser.add_argument('--days', type=int, default=120, help='K线天数')
    parser.add_argument('--source', choices=['auto', 'tencent', 'eastmoney'], default='auto')
    parser.add_argument('--format', choices=['json', 'csv'], default='json')
    args = parser.parse_args()
    
    klines = fetch_kline(args.code, args.days, args.source)
    
    if args.format == 'json':
        print(json.dumps(klines, indent=2))
    else:
        print('date,open,close,high,low,volume')
        for k in klines:
            print(f'{k["date"]},{k["open"]},{k["close"]},{k["high"]},{k["low"]},{k["volume"]}')
