#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全A股实时行情批量获取 — 腾讯行情API
支持批量查询、板块信息获取、量比/换手率过滤

用法:
  python3 realtime_quotes.py --output /tmp/quotes.json
  python3 realtime_quotes.py --codes 600519,000858 --output /tmp/quotes.json
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def get_all_a_share_codes():
    """获取全A股代码列表 — 从东方财富股票列表API"""
    url = ("https://datacenter-web.eastmoney.com/api/data/v1/get?"
           "reportName=RPT_LICO_FN_CPD&columns=SECURITY_CODE,SECURITY_NAME_ABBR,ORG_CODE,TRADE_MARKET_CODE"
           "&pageSize=500&pageNumber={page}&sortColumns=SECURITY_CODE&sortTypes=1"
           "&filter=(TRADE_MARKET_CODE+in+(\"069001001001\",\"069001001002\",\"069001002001\",\"069001002002\"))")
    
    all_codes = []
    for page in range(1, 15):  # 最多14页，500*14=7000
        try:
            req = urllib.request.Request(url.format(page=page), headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read().decode())
            result = data.get("result", {})
            items = result.get("data", [])
            if not items:
                break
            for item in items:
                code = item.get("SECURITY_CODE", "")
                market = item.get("TRADE_MARKET_CODE", "")
                if code:
                    # 沪市: 069001001001/069001001002  深市: 069001002001/069001002002
                    prefix = "sh" if "001001" in market else "sz"
                    all_codes.append(f"{prefix}{code}")
        except Exception as e:
            print(f"[WARN] Page {page} failed: {e}", file=sys.stderr)
            break
    return all_codes


def get_a_share_codes_simple():
    """备用方案：用本地缓存或简单范围生成"""
    cache_file = Path(__file__).parent.parent.parent.parent / "data" / "a_share_codes.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    
    # 最后手段：生成主要范围
    codes = []
    # 沪市主板 600xxx, 601xxx, 603xxx, 605xxx
    for prefix in ['600', '601', '603', '605']:
        for i in range(1000):
            codes.append(f"sh{prefix}{i:03d}")
    # 深市主板 000xxx, 001xxx
    for prefix in ['000', '001']:
        for i in range(1000):
            codes.append(f"sz{prefix}{i:03d}")
    # 中小板/创业板 002xxx, 003xxx, 300xxx, 301xxx
    for prefix in ['002', '003', '300', '301']:
        for i in range(1000):
            codes.append(f"sz{prefix}{i:03d}")
    # 科创板 688xxx
    for i in range(1000):
        codes.append(f"sh688{i:03d}")
    return codes


def batch_fetch_quotes(codes, batch_size=80, max_workers=4):
    """批量获取实时行情
    
    返回: {code: {name, price, yesterday_close, open, high, low, volume, amount, 
                   change_pct, volume_ratio, turnover_rate, ...}}
    """
    results = {}
    batches = [codes[i:i+batch_size] for i in range(0, len(codes), batch_size)]
    
    def fetch_batch(batch):
        batch_results = {}
        codes_str = ",".join(batch)
        url = f"https://qt.gtimg.cn/q={codes_str}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=15)
            raw = resp.read().decode("gbk", errors="ignore")
            
            for line in raw.strip().split("\n"):
                if "v_" not in line or "~" not in line:
                    continue
                # 提取代码
                var_name = line.split("=")[0].strip()
                code = var_name.replace("v_", "")
                
                fields = line.split("~")
                if len(fields) < 50:
                    continue
                
                name = fields[1]
                price = _safe_float(fields[3])
                
                # 跳过无效数据
                if price <= 0 or not name:
                    continue
                
                # 跳过ST
                if "ST" in name or "退" in name:
                    continue
                
                batch_results[code] = {
                    "name": name,
                    "code": code,
                    "price": price,
                    "yesterday_close": _safe_float(fields[4]),
                    "open": _safe_float(fields[5]),
                    "volume": _safe_float(fields[6]),  # 手
                    "high": _safe_float(fields[33]),
                    "low": _safe_float(fields[34]),
                    "change_pct": _safe_float(fields[32]),
                    "turnover_rate": _safe_float(fields[38]),
                    "volume_ratio": _safe_float(fields[49]) if len(fields) > 49 else 0,
                    "pe": _safe_float(fields[39]) if len(fields) > 39 else 0,
                    "market_cap": _safe_float(fields[45]) if len(fields) > 45 else 0,  # 亿
                    "industry": fields[19] if len(fields) > 19 else "",
                }
        except Exception as e:
            pass  # 静默失败，不影响其他批次
        return batch_results
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_batch, b): i for i, b in enumerate(batches)}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                batch_results = future.result()
                results.update(batch_results)
            except:
                pass
            if done % 20 == 0:
                print(f"  Progress: {done}/{len(batches)} batches, {len(results)} stocks", file=sys.stderr)
    
    return results


def _safe_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="全A股实时行情批量获取")
    parser.add_argument("--codes", help="指定代码（逗号分隔），留空=全A股")
    parser.add_argument("--output", default="/tmp/a_share_quotes.json", help="输出文件")
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-codes", action="store_true", help="保存代码列表到缓存")
    args = parser.parse_args()
    
    if args.codes:
        codes = []
        for c in args.codes.split(","):
            c = c.strip()
            if c.startswith("sh") or c.startswith("sz"):
                codes.append(c)
            elif c.startswith("6"):
                codes.append(f"sh{c}")
            else:
                codes.append(f"sz{c}")
    else:
        print("Fetching all A-share codes...", file=sys.stderr)
        codes = get_all_a_share_codes()
        if len(codes) < 1000:
            print(f"[WARN] Only got {len(codes)} from API, using simple range", file=sys.stderr)
            codes = get_a_share_codes_simple()
        print(f"Total codes: {len(codes)}", file=sys.stderr)
        
        if args.save_codes:
            cache_dir = Path(__file__).parent.parent.parent.parent / "data"
            cache_dir.mkdir(exist_ok=True)
            with open(cache_dir / "a_share_codes.json", "w") as f:
                json.dump(codes, f)
            print(f"Codes saved to cache", file=sys.stderr)
    
    print(f"Fetching quotes for {len(codes)} stocks...", file=sys.stderr)
    t0 = time.time()
    quotes = batch_fetch_quotes(codes, args.batch_size, args.workers)
    elapsed = time.time() - t0
    print(f"Got {len(quotes)} valid quotes in {elapsed:.1f}s", file=sys.stderr)
    
    with open(args.output, "w") as f:
        json.dump(quotes, f, ensure_ascii=False)
    print(f"Saved to {args.output}", file=sys.stderr)
    
    # 输出统计到stdout
    print(json.dumps({
        "total_codes": len(codes),
        "valid_quotes": len(quotes),
        "elapsed_seconds": round(elapsed, 1)
    }))


if __name__ == "__main__":
    main()
