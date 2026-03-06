#!/usr/bin/env python3
"""
热点猎手 — 新闻热点扫描与评分工具

用法:
  python3 hotspot_scanner.py --output /tmp/hotspot_scan.json
  python3 hotspot_scanner.py --query "人工智能" --topk 5
  python3 hotspot_scanner.py --check-holdings '{"002230.SZ":"科大讯飞"}'

功能:
1. 扫描多个新闻源获取最新财经新闻
2. 用热度评分模型量化每条新闻的交易价值
3. 识别受益板块和个股
4. 检查持仓热点是否仍在发酵
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.parse
import re
from datetime import datetime, timedelta
from pathlib import Path


def fetch_eastmoney_news(limit=20):
    """东方财富快讯"""
    results = []
    try:
        url = f"https://np-listapi.eastmoney.com/comm/web/getNewsByColumns?client=web&columns=74&limit={limit}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read().decode("utf-8"))
        for item in (data.get("data") or {}).get("list", []):
            results.append({
                "title": item.get("title", ""),
                "summary": item.get("digest", item.get("title", "")),
                "time": item.get("showtime", ""),
                "url": item.get("url", ""),
                "source": "eastmoney"
            })
    except Exception as e:
        print(f"[WARN] eastmoney fetch failed: {e}", file=sys.stderr)
    return results


def fetch_sina_news(limit=20):
    """新浪财经滚动新闻"""
    results = []
    try:
        url = f"https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2509&num={limit}&encode=utf-8"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read().decode("utf-8"))
        for item in data.get("result", {}).get("data", []):
            results.append({
                "title": item.get("title", ""),
                "summary": item.get("summary", item.get("title", "")),
                "time": item.get("ctime", ""),
                "url": item.get("url", ""),
                "source": "sina"
            })
    except Exception as e:
        print(f"[WARN] sina fetch failed: {e}", file=sys.stderr)
    return results


def step_search(query, topk=5):
    """使用 StepSearch 搜索"""
    results = []
    try:
        script_dir = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
        search_script = script_dir / "step_search.py"
        if search_script.exists():
            import subprocess
            proc = subprocess.run(
                [sys.executable, str(search_script), query, "--topk", str(topk), "--json"],
                capture_output=True, text=True, timeout=15
            )
            if proc.returncode == 0:
                data = json.loads(proc.stdout)
                for r in data.get("search_results", []):
                    results.append({
                        "title": r.get("title", ""),
                        "summary": r.get("snippet", ""),
                        "time": r.get("time", ""),
                        "url": r.get("url", ""),
                        "source": "stepsearch"
                    })
    except Exception as e:
        print(f"[WARN] step_search failed: {e}", file=sys.stderr)
    return results


# 热点关键词分类
POLICY_KEYWORDS = ["国务院", "国常会", "部委", "规划", "意见", "政策", "补贴", "减税",
                   "央行", "降准", "降息", "MLF", "LPR", "财政", "专项债"]
TECH_KEYWORDS = ["突破", "首发", "量产", "自主可控", "国产替代", "芯片", "AI", "人工智能",
                 "大模型", "机器人", "无人驾驶", "新能源", "光伏", "锂电", "氢能"]
EVENT_KEYWORDS = ["地震", "台风", "制裁", "关税", "冲突", "疫情", "断供", "涨价", "短缺"]


def classify_news(title, summary):
    """分类新闻类型"""
    text = title + summary
    if any(kw in text for kw in POLICY_KEYWORDS):
        return "policy"
    elif any(kw in text for kw in TECH_KEYWORDS):
        return "tech"
    elif any(kw in text for kw in EVENT_KEYWORDS):
        return "event"
    return "general"


def score_hotspot(news_item, existing_hotspots=None):
    """
    热度评分模型（满分100）
    - 新鲜度 30分
    - 影响力 30分  
    - 可交易性 20分
    - 持续性 20分
    """
    title = news_item.get("title", "")
    summary = news_item.get("summary", "")
    text = title + summary
    news_type = classify_news(title, summary)
    
    score = 0
    breakdown = {}
    
    # 1. 新鲜度（30分）
    freshness = 25  # 默认较新
    time_str = news_item.get("time", "")
    if time_str:
        try:
            # 尝试解析时间
            now = datetime.now()
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M"]:
                try:
                    t = datetime.strptime(time_str, fmt)
                    hours_ago = (now - t).total_seconds() / 3600
                    if hours_ago < 6:
                        freshness = 30
                    elif hours_ago < 24:
                        freshness = 25
                    elif hours_ago < 48:
                        freshness = 15
                    else:
                        freshness = 5
                    break
                except ValueError:
                    continue
        except:
            pass
    breakdown["freshness"] = freshness
    score += freshness
    
    # 2. 影响力（30分）
    impact = 10
    if news_type == "policy":
        if any(kw in text for kw in ["国务院", "国常会", "央行"]):
            impact = 30
        else:
            impact = 20
    elif news_type == "tech":
        if any(kw in text for kw in ["突破", "首发", "量产"]):
            impact = 25
        else:
            impact = 15
    elif news_type == "event":
        if any(kw in text for kw in ["制裁", "关税", "断供"]):
            impact = 25
        else:
            impact = 15
    breakdown["impact"] = impact
    score += impact
    
    # 3. 可交易性（20分）
    tradability = 5
    # 有具体行业/板块 → 更可交易
    sector_keywords = ["板块", "概念", "产业链", "龙头", "受益", "利好",
                       "半导体", "医药", "军工", "消费", "汽车", "银行"]
    matches = sum(1 for kw in sector_keywords if kw in text)
    if matches >= 3:
        tradability = 20
    elif matches >= 2:
        tradability = 15
    elif matches >= 1:
        tradability = 10
    breakdown["tradability"] = tradability
    score += tradability
    
    # 4. 持续性（20分）
    persistence = 10
    if news_type == "policy":
        persistence = 18  # 政策通常持续性强
    elif news_type == "tech":
        persistence = 15
    elif news_type == "event":
        persistence = 8  # 事件往往一日游
    breakdown["persistence"] = persistence
    score += persistence
    
    return {
        "score": score,
        "breakdown": breakdown,
        "type": news_type,
        "title": title,
        "summary": summary[:200]
    }


def scan_all_sources():
    """扫描所有新闻源"""
    all_news = []
    
    # 1. 东方财富快讯
    em_news = fetch_eastmoney_news(20)
    all_news.extend(em_news)
    print(f"  东方财富: {len(em_news)}条", file=sys.stderr)
    
    # 2. 新浪财经
    sina_news = fetch_sina_news(20)
    all_news.extend(sina_news)
    print(f"  新浪财经: {len(sina_news)}条", file=sys.stderr)
    
    # 3. StepSearch 热点搜索
    for query in ["A股 热点 今日", "政策利好 板块", "产业 突破 利好"]:
        results = step_search(query, topk=3)
        all_news.extend(results)
    print(f"  StepSearch: {len(all_news) - len(em_news) - len(sina_news)}条", file=sys.stderr)
    
    # 去重（按标题）
    seen = set()
    unique = []
    for item in all_news:
        title = item.get("title", "").strip()
        if title and title not in seen:
            seen.add(title)
            unique.append(item)
    
    print(f"  去重后: {len(unique)}条", file=sys.stderr)
    return unique


def main():
    parser = argparse.ArgumentParser(description="热点猎手 - 新闻扫描与评分")
    parser.add_argument("--output", "-o", help="输出JSON文件路径")
    parser.add_argument("--query", "-q", help="自定义搜索关键词")
    parser.add_argument("--topk", type=int, default=5, help="搜索结果数")
    parser.add_argument("--threshold", type=int, default=50, help="热度评分阈值")
    parser.add_argument("--check-holdings", help="检查持仓热点(JSON dict: code->name)")
    args = parser.parse_args()
    
    print(f"🔥 热点猎手扫描开始 {datetime.now().strftime('%Y-%m-%d %H:%M')}", file=sys.stderr)
    
    if args.query:
        # 自定义搜索
        news = step_search(args.query, args.topk)
    else:
        # 全量扫描
        news = scan_all_sources()
    
    # 评分
    scored = []
    for item in news:
        result = score_hotspot(item)
        result["url"] = item.get("url", "")
        result["time"] = item.get("time", "")
        result["source"] = item.get("source", "")
        scored.append(result)
    
    # 按分数排序
    scored.sort(key=lambda x: x["score"], reverse=True)
    
    # 筛选
    hot = [s for s in scored if s["score"] >= args.threshold]
    
    result = {
        "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_scanned": len(news),
        "above_threshold": len(hot),
        "threshold": args.threshold,
        "hotspots": hot[:20],  # Top 20
        "all_scored": scored[:50]  # Top 50 for reference
    }
    
    # 持仓检查
    if args.check_holdings:
        try:
            holdings = json.loads(args.check_holdings)
            holding_checks = []
            for code, name in holdings.items():
                check_news = step_search(f"{name} 最新消息", topk=3)
                still_hot = any(score_hotspot(n)["score"] >= 40 for n in check_news) if check_news else False
                holding_checks.append({
                    "code": code,
                    "name": name,
                    "recent_news": len(check_news),
                    "still_hot": still_hot,
                    "latest": check_news[0]["title"] if check_news else "无新消息"
                })
            result["holding_checks"] = holding_checks
        except Exception as e:
            print(f"[WARN] holding check failed: {e}", file=sys.stderr)
    
    # 输出
    output_json = json.dumps(result, ensure_ascii=False, indent=2)
    
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"✅ 结果已写入 {args.output}", file=sys.stderr)
    else:
        print(output_json)
    
    # 摘要
    print(f"\n📊 扫描摘要:", file=sys.stderr)
    print(f"  总计: {len(news)}条新闻", file=sys.stderr)
    print(f"  热点(≥{args.threshold}分): {len(hot)}条", file=sys.stderr)
    if hot:
        print(f"  最热: [{hot[0]['score']}分] {hot[0]['title'][:50]}", file=sys.stderr)
    
    return result


if __name__ == "__main__":
    main()
