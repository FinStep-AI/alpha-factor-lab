#!/usr/bin/env python3
"""
多市场K线数据获取器
支持 A股（东方财富API）、美股/港股（yfinance）
输出统一格式 JSON，供 stock_chart.py 使用
"""

import json
import sys
import os
import argparse
from datetime import datetime, timedelta
import urllib.request
import urllib.parse

def fetch_a_share(code, days=120):
    """A股数据 - 东方财富API"""
    # 判断市场
    if code.startswith('6') or code.endswith('.SH'):
        secid = f"1.{code.replace('.SH','').replace('.SZ','')}"
    else:
        secid = f"0.{code.replace('.SH','').replace('.SZ','')}"
    
    pure_code = code.replace('.SH','').replace('.SZ','')
    return _fetch_eastmoney(secid, pure_code, "A", days)


def fetch_hk_eastmoney(code, days=120):
    """港股数据 - 东方财富API（secid=116.xxxxx）"""
    pure_code = code.replace('.HK', '').zfill(5)
    secid = f"116.{pure_code}"
    return _fetch_eastmoney(secid, f"{pure_code}.HK", "HK", days)


def _fetch_eastmoney(secid, display_code, market, days=120):
    """东方财富通用K线获取"""
    
    url = (f"https://push2his.eastmoney.com/api/qt/stock/kline/get?"
           f"secid={secid}&fields1=f1,f2,f3,f4,f5,f6&"
           f"fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&"
           f"klt=101&fqt=1&end=20500101&lmt={days}")
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read().decode())
    
    klines = data.get("data", {}).get("klines", [])
    if not klines:
        raise ValueError(f"No data for {display_code}")
    
    records = []
    for line in klines:
        fields = line.split(",")
        records.append({
            "date": fields[0],
            "open": float(fields[1]),
            "close": float(fields[2]),
            "high": float(fields[3]),
            "low": float(fields[4]),
            "volume": float(fields[5]),
            "amount": float(fields[6]),
            "amplitude": float(fields[7]) if len(fields) > 7 else 0,
            "change_pct": float(fields[8]) if len(fields) > 8 else 0,
            "change_amt": float(fields[9]) if len(fields) > 9 else 0,
            "turnover": float(fields[10]) if len(fields) > 10 else 0,
        })
    
    return {
        "market": market,
        "code": display_code,
        "source": "eastmoney",
        "data": records
    }


def fetch_yfinance(ticker, days=120):
    """美股/港股数据 - yfinance (含重试)"""
    try:
        import yfinance as yf
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
        import yfinance as yf
    
    import time
    
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.8))
    
    df = None
    for attempt in range(3):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
            break
        except Exception as e:
            if 'RateLimit' in type(e).__name__ or '429' in str(e):
                wait = 5 * (attempt + 1)
                print(f"Rate limited, retrying in {wait}s... ({attempt+1}/3)")
                time.sleep(wait)
            else:
                raise
    
    if df is None or df.empty:
        print(f"yfinance failed for {ticker}, trying FMP API...")
        try:
            return fetch_fmp(ticker, days)
        except Exception as e2:
            print(f"FMP also failed ({e2}), trying Yahoo v8...")
            try:
                return fetch_yahoo_raw(ticker, days)
            except Exception as e3:
                raise ValueError(f"All data sources failed for {ticker}: yfinance(rate limited), FMP({e2}), Yahoo({e3})")
    
    # 只取最近 days 个交易日
    df = df.tail(days)
    
    records = []
    for idx, row in df.iterrows():
        records.append({
            "date": idx.strftime('%Y-%m-%d'),
            "open": round(float(row['Open']), 4),
            "close": round(float(row['Close']), 4),
            "high": round(float(row['High']), 4),
            "low": round(float(row['Low']), 4),
            "volume": float(row['Volume']),
            "amount": 0,
            "change_pct": 0,
            "change_amt": 0,
            "turnover": 0,
        })
    
    _calc_change_pct(records)
    
    return {
        "market": "US" if not ticker.endswith('.HK') else "HK",
        "code": ticker,
        "source": "yfinance",
        "data": records
    }


def fetch_yahoo_raw(ticker, days=120):
    """备用方案1 - 直接调 Yahoo Finance v8 chart API"""
    import time
    
    end_ts = int(time.time())
    start_ts = end_ts - int(days * 1.8 * 86400)
    
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?period1={start_ts}&period2={end_ts}&interval=1d&includePrePost=false")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json"
    }
    
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read().decode())
    
    chart = data.get("chart", {}).get("result", [{}])[0]
    timestamps = chart.get("timestamp", [])
    quote = chart.get("indicators", {}).get("quote", [{}])[0]
    
    if not timestamps:
        raise ValueError(f"Yahoo v8 API returned no data for {ticker}")
    
    opens = quote.get("open", [])
    highs = quote.get("high", [])
    lows = quote.get("low", [])
    closes = quote.get("close", [])
    volumes = quote.get("volume", [])
    
    records = []
    for i, ts in enumerate(timestamps):
        o = opens[i] if i < len(opens) and opens[i] is not None else 0
        h = highs[i] if i < len(highs) and highs[i] is not None else 0
        l = lows[i] if i < len(lows) and lows[i] is not None else 0
        c = closes[i] if i < len(closes) and closes[i] is not None else 0
        v = volumes[i] if i < len(volumes) and volumes[i] is not None else 0
        if c == 0:
            continue
        dt = datetime.utcfromtimestamp(ts)
        records.append({
            "date": dt.strftime('%Y-%m-%d'),
            "open": round(float(o), 4),
            "close": round(float(c), 4),
            "high": round(float(h), 4),
            "low": round(float(l), 4),
            "volume": float(v),
            "amount": 0,
            "change_pct": 0,
            "change_amt": 0,
            "turnover": 0,
        })
    
    # 去重
    seen = set()
    unique = []
    for r in records:
        if r['date'] not in seen:
            seen.add(r['date'])
            unique.append(r)
    records = unique[-days:]
    
    _calc_change_pct(records)
    
    return {
        "market": "US" if not ticker.endswith('.HK') else "HK",
        "code": ticker,
        "source": "yahoo_v8",
        "data": records
    }


def fetch_finnhub(ticker, days=120):
    """备用方案2 - Finnhub stock candles API（需 API key）"""
    import time
    
    # 读取 API key
    api_key = os.environ.get('FINNHUB_API_KEY', '')
    if not api_key:
        env_path = os.path.expanduser('~/.openclaw/workspace/alpha-factor-lab/.env')
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith('FINNHUB_API_KEY='):
                        api_key = line.strip().split('=', 1)[1]
    
    if not api_key:
        raise ValueError("No FINNHUB_API_KEY found")
    
    end_ts = int(time.time())
    start_ts = end_ts - int(days * 1.8 * 86400)
    
    url = (f"https://finnhub.io/api/v1/stock/candle?"
           f"symbol={ticker}&resolution=D&from={start_ts}&to={end_ts}&token={api_key}")
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read().decode())
    
    if data.get('s') == 'no_data' or not data.get('t'):
        raise ValueError(f"Finnhub returned no data for {ticker}")
    
    records = []
    for i in range(len(data['t'])):
        dt = datetime.utcfromtimestamp(data['t'][i])
        records.append({
            "date": dt.strftime('%Y-%m-%d'),
            "open": round(float(data['o'][i]), 4),
            "close": round(float(data['c'][i]), 4),
            "high": round(float(data['h'][i]), 4),
            "low": round(float(data['l'][i]), 4),
            "volume": float(data['v'][i]),
            "amount": 0,
            "change_pct": 0,
            "change_amt": 0,
            "turnover": 0,
        })
    
    records = records[-days:]
    _calc_change_pct(records)
    
    return {
        "market": "US" if not ticker.endswith('.HK') else "HK",
        "code": ticker,
        "source": "finnhub",
        "data": records
    }


def _calc_change_pct(records):
    """计算涨跌幅"""
    for i in range(1, len(records)):
        prev_close = records[i-1]['close']
        if prev_close > 0:
            records[i]['change_pct'] = round((records[i]['close'] - prev_close) / prev_close * 100, 2)
            records[i]['change_amt'] = round(records[i]['close'] - prev_close, 4)


def fetch_fmp(ticker, days=120):
    """备用方案 - FMP (Financial Modeling Prep) stable API"""
    api_key = os.environ.get('FMP_API_KEY', '')
    if not api_key:
        env_path = os.path.expanduser('~/.openclaw/workspace/alpha-factor-lab/.env')
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith('FMP_API_KEY='):
                        api_key = line.strip().split('=', 1)[1]
    
    if not api_key:
        raise ValueError("No FMP_API_KEY found")
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=int(days * 1.8))).strftime('%Y-%m-%d')
    
    url = (f"https://financialmodelingprep.com/stable/historical-price-eod/full"
           f"?symbol={ticker}&from={start_date}&to={end_date}&apikey={api_key}")
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read().decode())
    
    if isinstance(data, dict) and 'Error' in str(data):
        raise ValueError(f"FMP error: {data}")
    
    items = data if isinstance(data, list) else []
    if not items:
        raise ValueError(f"FMP returned no data for {ticker}")
    
    # FMP 数据是倒序的(最新在前)，需要翻转
    items.sort(key=lambda x: x['date'])
    items = items[-days:]
    
    records = []
    for item in items:
        records.append({
            "date": item['date'],
            "open": round(float(item['open']), 4),
            "close": round(float(item['close']), 4),
            "high": round(float(item['high']), 4),
            "low": round(float(item['low']), 4),
            "volume": float(item['volume']),
            "amount": 0,
            "change_pct": round(float(item.get('changePercent', 0)), 2),
            "change_amt": round(float(item.get('change', 0)), 4),
            "turnover": 0,
        })
    
    return {
        "market": "US" if not ticker.endswith('.HK') else "HK",
        "code": ticker,
        "source": "fmp",
        "data": records
    }


def normalize_ticker(code, market):
    """标准化股票代码"""
    code = code.upper().strip()
    
    if market == "A":
        code = code.replace('.SH', '').replace('.SZ', '')
        return code
    elif market == "HK":
        code = code.replace('.HK', '')
        # yfinance 港股格式: 0700.HK
        return f"{code.zfill(4)}.HK"
    elif market == "US":
        return code
    
    # 自动判断
    if code.replace('.SH','').replace('.SZ','').isdigit() and len(code.replace('.SH','').replace('.SZ','')) == 6:
        return code.replace('.SH','').replace('.SZ','')
    elif code.replace('.HK','').isdigit() and len(code.replace('.HK','')) <= 5:
        c = code.replace('.HK','')
        return f"{c.zfill(4)}.HK"
    else:
        return code


def auto_detect_market(code):
    """自动检测市场"""
    code = code.strip()
    if code.endswith('.SH') or code.endswith('.SZ'):
        return 'A'
    if code.endswith('.HK'):
        return 'HK'
    pure = code.replace('.SH','').replace('.SZ','').replace('.HK','')
    if pure.isdigit():
        if len(pure) == 6:
            return 'A'
        elif len(pure) <= 5:
            return 'HK'
    return 'US'


def main():
    parser = argparse.ArgumentParser(description='获取K线数据')
    parser.add_argument('--market', type=str, default='auto', help='市场: A/US/HK/auto')
    parser.add_argument('--code', type=str, required=True, help='股票代码')
    parser.add_argument('--days', type=int, default=120, help='获取天数')
    parser.add_argument('--output', type=str, default=None, help='输出路径')
    args = parser.parse_args()
    
    market = args.market.upper()
    if market == 'AUTO':
        market = auto_detect_market(args.code)
    
    ticker = normalize_ticker(args.code, market)
    
    print(f"Fetching {ticker} ({market}) - {args.days} days...")
    
    if market == 'A':
        result = fetch_a_share(ticker, args.days)
    elif market == 'HK':
        result = fetch_hk_eastmoney(ticker, args.days)
    else:
        result = fetch_yfinance(ticker, args.days)
    
    output = args.output or f"/tmp/ta_{args.code.replace('.','_')}_kline.json"
    with open(output, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Done. {len(result['data'])} records → {output}")
    return output


if __name__ == "__main__":
    main()
