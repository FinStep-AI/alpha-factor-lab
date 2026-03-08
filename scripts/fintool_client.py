#!/usr/bin/env python3
"""fintool API统一封装层（聚源+财联社数据源）

优先级最高的A股数据源，无速率限制。
协议: JSON-RPC 2.0 over HTTP+SSE
"""
import requests
import json
import re
import time
import os
import threading
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://fintool-mcp.finstep.cn"
SIGNATURE = "AI-ONE-7727330bb41a3fb4a6446f9c2a120c90"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream"
}

# 全局Session复用TCP连接，大幅降低延迟
# 线程安全：每个线程用独立Session
_thread_local = threading.local()

def _get_session():
    if not hasattr(_thread_local, 'session'):
        _thread_local.session = requests.Session()
        _thread_local.session.headers.update(HEADERS)
    return _thread_local.session

# 主线程也有一个全局session（向后兼容）
_session = requests.Session()
_session.headers.update(HEADERS)


def _call(service: str, tool: str, args: dict, timeout: int = 30) -> dict:
    """底层调用：发送JSON-RPC请求，解析SSE响应（使用Session复用连接）"""
    url = f"{BASE_URL}/{service}?signature={SIGNATURE}"
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 1,
        "params": {"name": tool, "arguments": args}
    }
    session = _get_session()
    r = session.post(url, json=payload, timeout=timeout)
    r.encoding = 'utf-8'

    for line in r.text.split("\n"):
        if line.startswith("data:"):
            data = json.loads(line[5:].strip())
            if "error" in data:
                raise Exception(f"fintool error: {data['error']}")
            content = data.get("result", {}).get("content", [])
            for c in content:
                text = c.get("text", "")
                if text.startswith("Error"):
                    raise Exception(f"fintool tool error: {text}")
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"raw": text}
    raise Exception("No valid SSE data response")


def _parse_number(val) -> float:
    """解析fintool返回的中文格式数字（如 '3.55万手' → 35500）"""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip().replace(',', '')
        # 去掉百分号
        if val.endswith('%'):
            try:
                return float(val[:-1])
            except ValueError:
                return 0.0
        # 处理中文单位
        multipliers = {'万': 10000, '亿': 100000000, '千': 1000, '百': 100}
        for unit, mult in multipliers.items():
            if unit in val:
                # 去掉单位后面的量词（手、元、股等）
                num_str = val.split(unit)[0]
                try:
                    return float(num_str) * mult
                except ValueError:
                    return 0.0
        try:
            return float(val)
        except ValueError:
            return 0.0
    return 0.0


# ===== INDEX =====

def get_constituents(index_name: str = "中证1000") -> List[Dict]:
    """获取指数成分股列表（自动分页，获取全部）"""
    all_stocks = []
    page = 1
    while True:
        resp = _call("index", "get_index_constituent", {
            "keyword": index_name, "page": page, "page_size": 100
        })
        data = resp.get("data", {})
        items = data.get("list", [])
        if not items:
            break
        all_stocks.extend(items)
        # 如果返回数量不足一页，说明到末尾了
        if len(items) < 100:
            break
        page += 1
    return all_stocks


def get_constituent_snapshot(index_name: str = "中证1000", page: int = 1,
                              page_size: int = 100) -> List[Dict]:
    """获取成分股行情快照"""
    resp = _call("index", "get_constituent_stock_snapshot", {
        "keyword": index_name, "page": page, "page_size": page_size
    })
    return resp.get("data", {}).get("list", resp.get("data", []))


# ===== MARKET QUOTE =====

def get_kline(code: str, end_date: str, num: int = 100,
              kline_type: int = 1, reinstatement: int = 2) -> List[Dict]:
    """获取K线数据
    code: 股票代码（纯6位）
    end_date: 截止日期 YYYY-MM-DD
    num: 条数（最多100）
    kline_type: 1=日K 2=周K 3=月K
    reinstatement: 1=不复权 2=前复权 3=后复权
    """
    resp = _call("market_quote", "get_kline", {
        "keyword": code,
        "end_date": end_date,
        "kline_num": min(num, 100),
        "kline_type": kline_type,
        "reinstatement_type": reinstatement,
    })
    items = resp.get("data", [])
    # 标准化字段
    results = []
    for item in items:
        results.append({
            "date": item.get("trade_date", "")[:10],
            "open": _parse_number(item.get("open_price")),
            "close": _parse_number(item.get("close_price")),
            "high": _parse_number(item.get("high_price")),
            "low": _parse_number(item.get("low_price")),
            "volume": _parse_number(item.get("trade_lots")),
            "amount": _parse_number(item.get("trade_balance")),
            "pct_change": _parse_number(item.get("price_change_rate")),
            "amplitude": _parse_number(item.get("amplitude")),
            "turnover": _parse_number(item.get("turnover_rate")) if "turnover_rate" in item else 0,
        })
    return results


def get_kline_history(code: str, start_date: str, end_date: str,
                      kline_type: int = 1, reinstatement: int = 2) -> List[Dict]:
    """获取历史K线（自动分段，突破100条限制）
    通过 end_date 递推，每次拉100条
    """
    all_klines = []
    current_end = end_date

    for _ in range(20):  # 最多20段 = 2000天
        batch = get_kline(code, current_end, 100, kline_type, reinstatement)
        if not batch:
            break

        # 过滤掉早于start_date的
        batch = [k for k in batch if k["date"] >= start_date]
        all_klines = batch + all_klines  # batch较早，拼在前面

        # 如果最早的记录已经到start_date或不足100条，停止
        if len(batch) < 100 or batch[0]["date"] <= start_date:
            break

        # 下一段end_date = 当前最早日期的前一天
        earliest = batch[0]["date"]
        from datetime import datetime, timedelta
        dt = datetime.strptime(earliest, "%Y-%m-%d") - timedelta(days=1)
        current_end = dt.strftime("%Y-%m-%d")

    # 去重（按日期）
    seen = set()
    unique = []
    for k in all_klines:
        if k["date"] not in seen:
            seen.add(k["date"])
            unique.append(k)
    return sorted(unique, key=lambda x: x["date"])


def get_snapshot(code: str) -> Dict:
    """获取实时行情快照"""
    resp = _call("market_quote", "get_snapshot", {"keyword": code})
    data = resp.get("data", {})
    if isinstance(data, list) and data:
        return data[0]
    return data


def get_net_flow(code: str, start_date: str = None, end_date: str = None) -> List[Dict]:
    """获取资金流向"""
    args = {"keyword": code}
    if end_date:
        args["end_date"] = end_date
    if start_date:
        args["start_date"] = start_date
    resp = _call("market_quote", "get_net_flow_list", args)
    return resp.get("data", [])


def get_leader_board(trade_date: str, code: str = None) -> List[Dict]:
    """获取龙虎榜"""
    args = {"trade_date": trade_date}
    if code:
        args["keyword"] = code
    resp = _call("market_quote", "get_leader_board", args)
    return resp.get("data", [])


def get_block_trade(code: str, start_date: str, end_date: str) -> List[Dict]:
    """获取大宗交易"""
    resp = _call("market_quote", "get_block_trade_detail", {
        "keyword": code, "start_date": start_date, "end_date": end_date
    })
    return resp.get("data", [])


# ===== COMPANY INFO =====

def get_industry_sws(code: str) -> Dict:
    """获取申万行业分类"""
    resp = _call("company_info", "get_stock_industry_sws", {"keyword": code})
    data = resp.get("data", [])
    if isinstance(data, list) and data:
        return data[0]
    return data if isinstance(data, dict) else {}


def get_valuation(code: str, end_date: str = None, begin_date: str = None) -> List[Dict]:
    """获取日度估值指标（PE/PB/PS/PCF/股息率）"""
    args = {"keyword": code}
    if end_date:
        args["end_date"] = end_date
    if begin_date:
        args["begin_date"] = begin_date
    resp = _call("company_info", "get_valuation_metrics_daily", args)
    return resp.get("data", [])


def get_financial_indicators(code: str, end_date: str = None,
                             begin_date: str = None) -> List[Dict]:
    """获取报告期财务衍生指标（ROE/ROA/毛利率等）"""
    args = {"keyword": code}
    if end_date:
        args["end_date"] = end_date
    if begin_date:
        args["begin_date"] = begin_date
    resp = _call("company_info", "get_financial_indicators", args)
    return resp.get("data", [])


def get_st_stocks() -> List[Dict]:
    """获取当前ST/*ST股票列表"""
    all_st = []
    page = 1
    while True:
        resp = _call("company_info", "get_security_change", {
            "page": page, "page_size": 100
        })
        # API返回格式: {"code":0,"data":{"items":[...],"total":N,"page":1,"total_pages":2}}
        data = resp.get("data", resp)
        if isinstance(data, dict):
            items = data.get("items", data.get("list", []))
            total_pages = data.get("total_pages", 1)
        elif isinstance(data, list):
            items = data
            total_pages = 1
        else:
            break
        if not items:
            break
        all_st.extend(items)
        if page >= total_pages or len(items) < 100:
            break
        page += 1
    return all_st


def get_shareholder_count(code: str, end_date: str = None) -> List[Dict]:
    """获取股东户数"""
    args = {"keyword": code}
    if end_date:
        args["end_date"] = end_date
    resp = _call("company_info", "get_share_holder_number", args)
    return resp.get("data", [])


def get_transfer_plan(code: str, begin_date: str = None) -> List[Dict]:
    """获取股东增减持计划"""
    args = {"keyword": code}
    if begin_date:
        args["begin_date"] = begin_date
    resp = _call("company_info", "get_transfer_plan", args)
    return resp.get("data", [])


def get_income_statement(code: str, end_date: str = None,
                         start_date: str = None) -> List[Dict]:
    """获取利润表"""
    args = {"keyword": code}
    if end_date:
        args["end_date"] = end_date
    if start_date:
        args["start_date"] = start_date
    resp = _call("company_info", "get_income_statement", args)
    return resp.get("data", [])


def get_balance_sheet(code: str, end_date: str = None,
                      start_date: str = None) -> List[Dict]:
    """获取资产负债表"""
    args = {"keyword": code}
    if end_date:
        args["end_date"] = end_date
    if start_date:
        args["start_date"] = start_date
    resp = _call("company_info", "get_balance_sheet", args)
    return resp.get("data", [])


def get_cash_flow(code: str, end_date: str = None,
                  start_date: str = None) -> List[Dict]:
    """获取现金流量表"""
    args = {"keyword": code}
    if end_date:
        args["end_date"] = end_date
    if start_date:
        args["start_date"] = start_date
    resp = _call("company_info", "get_cash_flow", args)
    return resp.get("data", [])


# ===== PLATES =====

def get_stock_plates(code: str) -> List[Dict]:
    """查询个股所属板块"""
    resp = _call("plates", "get_stock_plate", {"keyword": code})
    return resp.get("data", [])


def get_plate_list(keyword: str = "", sector_type: str = "1") -> List[Dict]:
    """获取板块列表（1=行业 2=地域 3=概念）"""
    resp = _call("plates", "get_plate_list", {
        "keyword": keyword, "sector_type": sector_type
    })
    return resp.get("data", [])


# ===== BATCH OPERATIONS =====

def batch_get_kline(codes: List[str], end_date: str, num: int = 100,
                    max_workers: int = 10, delay: float = 0,
                    progress_fn=None) -> Dict[str, List[Dict]]:
    """批量获取K线（多线程）
    返回: {code: [kline_records]}
    """
    results = {}
    errors = []

    def _fetch_one(code):
        try:
            klines = get_kline(code, end_date, num)
            return code, klines, None
        except Exception as e:
            return code, [], str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, c): c for c in codes}
        done = 0
        for future in as_completed(futures):
            code, klines, err = future.result()
            if err:
                errors.append((code, err))
            else:
                results[code] = klines
            done += 1
            if progress_fn and done % 50 == 0:
                progress_fn(done, len(codes), len(errors))
            if delay > 0:
                time.sleep(delay)

    if errors and progress_fn:
        progress_fn(done, len(codes), len(errors), final=True)

    return results


def batch_get_industry(codes: List[str], max_workers: int = 10) -> Dict[str, Dict]:
    """批量获取申万行业分类
    返回: {code: {first_industry_name, second_industry_name, third_industry_name}}
    """
    results = {}

    def _fetch_one(code):
        try:
            info = get_industry_sws(code)
            return code, info, None
        except Exception as e:
            return code, {}, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, c): c for c in codes}
        for future in as_completed(futures):
            code, info, err = future.result()
            if not err and info:
                results[code] = info

    return results


# ===== CLI =====

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="fintool API CLI")
    parser.add_argument("action", choices=[
        "constituents", "kline", "snapshot", "industry", "valuation",
        "st_list", "net_flow", "leader_board", "test"
    ])
    parser.add_argument("--code", default="600519")
    parser.add_argument("--index", default="中证1000")
    parser.add_argument("--end-date", default="2026-03-07")
    parser.add_argument("--num", type=int, default=5)
    args = parser.parse_args()

    if args.action == "constituents":
        stocks = get_constituents(args.index)
        print(f"成分股: {len(stocks)} 只")
        for s in stocks[:5]:
            print(f"  {s['security_code']} {s['security_name']}")

    elif args.action == "kline":
        klines = get_kline(args.code, args.end_date, args.num)
        print(f"K线({args.code}): {len(klines)} 条")
        for k in klines[-3:]:
            print(f"  {k['date']} O={k['open']} C={k['close']} H={k['high']} L={k['low']} V={k['volume']:.0f}")

    elif args.action == "snapshot":
        snap = get_snapshot(args.code)
        print(json.dumps(snap, ensure_ascii=False, indent=2))

    elif args.action == "industry":
        info = get_industry_sws(args.code)
        print(json.dumps(info, ensure_ascii=False, indent=2))

    elif args.action == "valuation":
        vals = get_valuation(args.code, args.end_date)
        for v in vals[:3]:
            print(json.dumps(v, ensure_ascii=False))

    elif args.action == "st_list":
        st = get_st_stocks()
        print(f"ST股票: {len(st)} 只")
        for s in st[:5]:
            print(f"  {s}")

    elif args.action == "net_flow":
        flows = get_net_flow(args.code, end_date=args.end_date)
        for f in flows[:3]:
            print(json.dumps(f, ensure_ascii=False))

    elif args.action == "leader_board":
        lb = get_leader_board(args.end_date)
        print(f"龙虎榜: {len(lb)} 条")
        for item in lb[:3]:
            print(f"  {item.get('security_code')} {item.get('security_name')}")

    elif args.action == "test":
        print("=== 全面测试 ===")
        # 成分股
        stocks = get_constituents("中证1000")
        print(f"✅ 成分股: {len(stocks)}")
        # K线
        klines = get_kline("600519", "2026-03-07", 3)
        print(f"✅ K线: {len(klines)} 条, 最新={klines[-1]['date'] if klines else 'N/A'}")
        # 行业
        ind = get_industry_sws("600519")
        print(f"✅ 行业: {ind.get('first_industry_name', '?')}/{ind.get('second_industry_name', '?')}")
        # 估值
        val = get_valuation("600519", "2026-03-07")
        print(f"✅ 估值: PE={val[0].get('pe_ttm', '?') if val else 'N/A'}")
        # 资金流
        flow = get_net_flow("600519", end_date="2026-03-07")
        print(f"✅ 资金流: {len(flow)} 条")
        print("\n全部通过!")
