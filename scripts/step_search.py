#!/usr/bin/env python3
"""
step_search.py — StepSearch API wrapper，替代 web_search (Brave)

用法：
  python3 step_search.py "搜索词"
  python3 step_search.py "搜索词" --topk 5 --location CN
  python3 step_search.py "搜索词" --topk 10 --location US --json

API:
  URL: https://staging-stepsearch-enginemixer.stepfun-inc.com/v1/search
  Method: POST (JSON)
  Header: x-stepsearch-auth: <token>
  Body: { "query": str, "topk": int, "location": str }
  
响应字段：
  search_results[]: id, engine, url, position, title, snippet, site, time, 
                    content, favicon, normed_url, authority_score, data_type
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error

API_URL = "https://staging-stepsearch-enginemixer.stepfun-inc.com/v1/search"
API_KEY = os.environ.get("STEP_SEARCH_KEY", "04UpIIMvOswyYQwGRddKefbZGJUPDpQm")


def search(query: str, topk: int = 5, location: str = "CN") -> dict:
    """执行搜索，返回原始API响应dict"""
    payload = json.dumps({
        "query": query,
        "topk": topk,
        "location": location,
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-stepsearch-auth": API_KEY,
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        return {"error": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return {"error": str(e)}


def format_results(data: dict) -> str:
    """格式化搜索结果为可读文本"""
    results = data.get("search_results", [])
    if not results:
        err = data.get("error", "")
        return f"无搜索结果{' — ' + err if err else ''}"

    lines = [f"搜索: {data.get('query', '?')} ({len(results)} 条, {data.get('time_cost_ms', '?')}ms)\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "(无标题)")
        url = r.get("url", "")
        snippet = r.get("snippet", "")
        site = r.get("site", "")
        time_str = r.get("time", "")
        
        lines.append(f"[{i}] {title}")
        if site or time_str:
            meta = " | ".join(filter(None, [site, time_str[:16] if time_str else ""]))
            lines.append(f"    {meta}")
        lines.append(f"    {url}")
        if snippet:
            lines.append(f"    {snippet[:200]}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="StepSearch 搜索工具")
    parser.add_argument("query", help="搜索词")
    parser.add_argument("--topk", type=int, default=5, help="返回条数 (default: 5)")
    parser.add_argument("--location", default="CN", help="地区 CN/US/... (default: CN)")
    parser.add_argument("--json", action="store_true", help="输出原始JSON")
    args = parser.parse_args()

    data = search(args.query, args.topk, args.location)

    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(format_results(data))


if __name__ == "__main__":
    main()
