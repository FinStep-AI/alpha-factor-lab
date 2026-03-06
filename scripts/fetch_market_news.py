#!/usr/bin/env python3
"""
A股市场新闻获取脚本 — 纯API调用，无JS渲染依赖
数据源：
  1. 新浪财经快讯 (feed.mix.sina.com.cn)
  2. 东方财富研报评级 (datacenter-web.eastmoney.com)

用法：
  python3 fetch_market_news.py                     # 获取最新市场新闻
  python3 fetch_market_news.py --stocks 贵州茅台,恒瑞医药  # 过滤特定股票相关新闻
  python3 fetch_market_news.py --output /tmp/news.json    # 输出到文件
"""
import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime


def fetch(url, timeout=10):
    """简单HTTP GET"""
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Referer': 'https://finance.sina.com.cn/'
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"  [WARN] fetch failed: {url[:80]}... → {e}", file=sys.stderr)
        return None


def fetch_sina_news(pages=5, hours=24):
    """新浪财经快讯 — 获取最近N小时的财经新闻"""
    news = []
    cutoff = time.time() - hours * 3600
    for page in range(1, pages + 1):
        url = f'https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2516&num=50&page={page}'
        text = fetch(url)
        if not text:
            break
        try:
            data = json.loads(text)
            items = data.get('result', {}).get('data', [])
            for it in items:
                ts = int(it.get('ctime', 0))
                if ts < cutoff:
                    continue
                title = it.get('title', '').strip()
                if not title:
                    continue
                news.append({
                    'title': title,
                    'source': '新浪快讯',
                    'time': datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M'),
                    'ts': ts,
                    'url': it.get('url', '')
                })
        except Exception:
            break
        time.sleep(0.3)
    return news


def fetch_em_reports(stock_name=None, days=3, limit=20):
    """东方财富研报评级（纯API，非JS渲染）"""
    reports = []
    try:
        url = ('https://datacenter-web.eastmoney.com/api/data/v1/get?'
               'reportName=RPT_RATING_CHANGE&columns=ALL&pageSize=50&pageNumber=1'
               '&sortColumns=NOTICE_DATE&sortTypes=-1&quoteColumns=')
        text = fetch(url, timeout=15)
        if not text:
            return reports
        data = json.loads(text)
        items = data.get('result', {}).get('data', []) or []
        for it in items:
            name = it.get('SECURITY_NAME_ABBR', '')
            if stock_name and stock_name not in name:
                continue
            reports.append({
                'stock': name,
                'code': it.get('SECURITY_CODE', ''),
                'org': it.get('ORG_NAME', ''),
                'rating': it.get('RATING_CHANGE_NAME', ''),
                'prev_rating': it.get('PREVIOUS_RATING_NAME', ''),
                'date': (it.get('NOTICE_DATE', '') or '')[:10],
                'source': '东方财富研报'
            })
            if len(reports) >= limit:
                break
    except Exception as e:
        print(f"  [WARN] EM reports failed: {e}", file=sys.stderr)
    return reports


def filter_by_stocks(news_list, stock_names):
    """过滤包含特定股票名称的新闻"""
    if not stock_names:
        return news_list
    matched = []
    for n in news_list:
        title = n.get('title', '')
        for name in stock_names:
            if name in title:
                n['matched_stock'] = name
                matched.append(n)
                break
    return matched


def classify_importance(news_list):
    """简单分类：重大/一般"""
    important_keywords = [
        '央行', '降息', '降准', '加息', 'GDP', '美联储', '关税', '制裁',
        '暴跌', '暴涨', '涨停', '跌停', '熔断', '退市', '停牌',
        '并购', '重组', '增持', '减持', '回购', '分红',
        '财报', '业绩', '预告', '快报', '超预期', '不及预期',
        '利好', '利空', '突破', '新高', '新低',
        '政策', '监管', '罚款', '调查', '违规',
        '战争', '疫情', '地震', '灾害'
    ]
    for n in news_list:
        title = n.get('title', '')
        hit = [k for k in important_keywords if k in title]
        n['importance'] = 'high' if hit else 'normal'
        n['keywords'] = hit
    return news_list


def main():
    parser = argparse.ArgumentParser(description='A股市场新闻获取')
    parser.add_argument('--stocks', type=str, default='',
                        help='逗号分隔的股票名称，过滤相关新闻')
    parser.add_argument('--hours', type=int, default=24,
                        help='获取最近N小时的新闻（默认24）')
    parser.add_argument('--output', type=str, default='',
                        help='输出JSON文件路径')
    parser.add_argument('--summary', action='store_true',
                        help='只输出摘要（适合cron调用）')
    args = parser.parse_args()

    stock_names = [s.strip() for s in args.stocks.split(',') if s.strip()]

    # 1. 获取新浪快讯
    print(f"📡 获取新浪快讯（最近{args.hours}小时）...", file=sys.stderr)
    all_news = fetch_sina_news(pages=5, hours=args.hours)
    print(f"  → {len(all_news)} 条快讯", file=sys.stderr)

    # 2. 获取东方财富研报（如有持仓股）
    reports = []
    if stock_names:
        print(f"📊 获取东方财富研报评级...", file=sys.stderr)
        for name in stock_names:
            r = fetch_em_reports(stock_name=name)
            reports.extend(r)
        print(f"  → {len(reports)} 条相关研报", file=sys.stderr)

    # 3. 分类重要性
    all_news = classify_importance(all_news)

    # 4. 过滤持仓相关
    stock_news = filter_by_stocks(all_news, stock_names) if stock_names else []

    # 5. 选取重大新闻
    important = [n for n in all_news if n['importance'] == 'high']

    result = {
        'fetch_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'total_news': len(all_news),
        'important_news': important[:30],
        'stock_related': stock_news[:20],
        'reports': reports[:20],
        'all_news': all_news[:50] if not args.summary else []
    }

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ 输出到 {args.output}", file=sys.stderr)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
