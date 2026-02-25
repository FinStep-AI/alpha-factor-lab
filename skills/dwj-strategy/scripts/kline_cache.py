#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K线缓存管理器 — 全A股日线数据
支持全量初始化和每日增量更新

用法:
  # 全量初始化（约4-5分钟）
  python3 kline_cache.py init --days 80

  # 每日增量更新（从实时行情追加当天K线，<1秒）
  python3 kline_cache.py update --quotes /tmp/a_share_quotes.json

  # 查询单只
  python3 kline_cache.py get --code sh600519

缓存文件: data/a_share_kline_cache.json
结构: {code: [[date, open, close, high, low, volume], ...]}
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent.parent.parent / "data"
CACHE_FILE = CACHE_DIR / "a_share_kline_cache.json"
CODES_FILE = CACHE_DIR / "a_share_codes.json"


def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    CACHE_DIR.mkdir(exist_ok=True)
    # 先写临时文件再rename，防止中途断电损坏
    tmp = str(CACHE_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, separators=(',', ':'))
    os.replace(tmp, str(CACHE_FILE))


def load_codes():
    if CODES_FILE.exists():
        with open(CODES_FILE) as f:
            return json.load(f)
    print("[ERROR] No codes file. Run realtime_quotes.py --save-codes first", file=sys.stderr)
    sys.exit(1)


def fetch_kline(code, days=80):
    """获取单只股票K线 — 腾讯接口"""
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,{days},qfq"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    data = json.loads(resp.read().decode())
    kdata = data.get("data", {}).get(code, {})
    days_data = kdata.get("qfqday", [])
    
    # 转换为紧凑格式: [date, open, close, high, low, volume]
    result = []
    for d in days_data:
        result.append([
            d[0],           # date
            float(d[1]),    # open
            float(d[2]),    # close
            float(d[3]),    # high
            float(d[4]),    # low
            float(d[5]),    # volume
        ])
    return result


def init_cache(days=80, workers=10):
    """全量初始化K线缓存"""
    codes = load_codes()
    # 过滤明显无效的代码
    valid_codes = [c for c in codes if c.startswith("sh") or c.startswith("sz")]
    print(f"Initializing cache for {len(valid_codes)} codes, {days} days each...", file=sys.stderr)
    
    cache = {}
    failed = []
    t0 = time.time()
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_kline, code, days): code for code in valid_codes}
        done = 0
        for future in as_completed(futures):
            done += 1
            code = futures[future]
            try:
                klines = future.result()
                if klines:
                    cache[code] = klines
            except Exception as e:
                failed.append(code)
            
            if done % 500 == 0:
                elapsed = time.time() - t0
                eta = elapsed / done * (len(valid_codes) - done)
                print(f"  Progress: {done}/{len(valid_codes)} ({len(cache)} valid) "
                      f"elapsed={elapsed:.0f}s eta={eta:.0f}s", file=sys.stderr)
    
    elapsed = time.time() - t0
    print(f"\nDone: {len(cache)} stocks cached, {len(failed)} failed, {elapsed:.1f}s", file=sys.stderr)
    
    save_cache(cache)
    print(f"Cache saved to {CACHE_FILE} ({os.path.getsize(CACHE_FILE) / 1024 / 1024:.1f}MB)", file=sys.stderr)
    
    return {"cached": len(cache), "failed": len(failed), "elapsed": round(elapsed, 1)}


def update_cache(quotes_file):
    """从实时行情文件追加当天K线"""
    cache = load_cache()
    if not cache:
        print("[ERROR] Cache empty. Run init first.", file=sys.stderr)
        sys.exit(1)
    
    with open(quotes_file) as f:
        quotes = json.load(f)
    
    today = datetime.now().strftime("%Y-%m-%d")
    updated = 0
    
    for code, q in quotes.items():
        if code not in cache:
            continue
        
        # 检查是否已有今天数据
        if cache[code] and cache[code][-1][0] == today:
            # 更新最后一行（盘中数据变化）
            cache[code][-1] = [today, q["open"], q["price"], q["high"], q["low"], q["volume"]]
        else:
            # 追加新行
            cache[code].append([today, q["open"], q["price"], q["high"], q["low"], q["volume"]])
            # 保留最近80天
            if len(cache[code]) > 80:
                cache[code] = cache[code][-80:]
        updated += 1
    
    save_cache(cache)
    print(f"Updated {updated} stocks for {today}", file=sys.stderr)
    return {"updated": updated, "date": today}


def get_stock(code):
    """查询单只股票K线"""
    cache = load_cache()
    if code not in cache:
        # 尝试加前缀
        for prefix in ["sh", "sz"]:
            if f"{prefix}{code}" in cache:
                code = f"{prefix}{code}"
                break
    
    if code not in cache:
        print(f"[ERROR] {code} not in cache", file=sys.stderr)
        return None
    
    return cache[code]


def main():
    parser = argparse.ArgumentParser(description="K线缓存管理器")
    sub = parser.add_subparsers(dest="action")
    
    p_init = sub.add_parser("init", help="全量初始化")
    p_init.add_argument("--days", type=int, default=80)
    p_init.add_argument("--workers", type=int, default=10)
    
    p_update = sub.add_parser("update", help="增量更新")
    p_update.add_argument("--quotes", required=True, help="实时行情JSON文件")
    
    p_get = sub.add_parser("get", help="查询单只")
    p_get.add_argument("--code", required=True)
    
    args = parser.parse_args()
    
    if args.action == "init":
        result = init_cache(args.days, args.workers)
        print(json.dumps(result))
    elif args.action == "update":
        result = update_cache(args.quotes)
        print(json.dumps(result))
    elif args.action == "get":
        data = get_stock(args.code)
        if data:
            print(json.dumps(data, ensure_ascii=False))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
