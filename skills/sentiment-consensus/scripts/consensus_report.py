#!/usr/bin/env python3
"""
舆情共识分析报告生成器
采集大V/机构对某只股票的观点 → 汇总分析 → 可视化图表

数据源：
1. 东方财富reportapi — 机构研报评级 (最稳定)
2. 东方财富search-api — 该股相关新闻
3. 同花顺诊股 — 多维度评分
4. 新浪快讯 — 全市场快讯(按股票名过滤)

用法:
  python3 consensus_report.py --code 600519 --name 贵州茅台 --output /tmp/consensus/
  python3 consensus_report.py --code 002594 --name 比亚迪 --output /tmp/consensus/ --days 7

输出:
  {code}_consensus.png   — 共识仪表盘可视化图
  {code}_consensus.json  — 结构化数据
"""

import argparse, json, os, re, sys, time, warnings
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from urllib.parse import quote
from urllib.error import URLError, HTTPError
import numpy as np

warnings.filterwarnings('ignore')

UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'

def fetch(url, timeout=10, encoding='utf-8'):
    """安全HTTP GET"""
    try:
        req = Request(url, headers={'User-Agent': UA, 'Accept': '*/*'})
        with urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return raw.decode(encoding, errors='replace')
    except Exception as e:
        print(f"  [WARN] fetch failed: {url[:80]}... -> {e}", file=sys.stderr)
        return None

# ═══════════════════════════════════════
# 数据源1: 东方财富研报评级 (reportapi)
# ═══════════════════════════════════════
def fetch_em_report_ratings(code, months=6):
    """东方财富reportapi机构评级 — 最可靠的数据源"""
    c = code.replace('.SH','').replace('.SZ','').replace('.BJ','')
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=months*30)).strftime('%Y-%m-%d')
    url = (f'https://reportapi.eastmoney.com/report/list?industryCode=*&pageSize=30'
           f'&industry=*&rating=*&ratingChange=*&beginTime={start}&endTime={end}'
           f'&pageNo=1&fields=&qType=0&orgCode=&rcode=&code={c}')
    text = fetch(url)
    if not text: return []
    try:
        data = json.loads(text)
        ratings = []
        for it in data.get('data', []):
            ratings.append(dict(
                org=it.get('orgSName', ''),
                rating=it.get('emRatingName', ''),
                prev_rating=it.get('lastEmRatingName', ''),
                target_eps=it.get('predictNextYearEps', ''),
                target_pe=it.get('predictNextYearPe', ''),
                report_date=it.get('publishDate', '')[:10],
                title=it.get('title', ''),
                author=','.join([a.split('.')[-1] for a in it.get('author', [])]),
                source='东方财富研报'
            ))
        return ratings
    except:
        return []

# ═══════════════════════════════════════
# 数据源2: 东方财富搜索新闻
# ═══════════════════════════════════════
def fetch_em_news(name, limit=15):
    """东方财富search-api搜索该股新闻"""
    param = json.dumps({
        "uid": "", "keyword": name,
        "type": ["cmsArticleWebOld"],
        "client": "web", "clientType": "web", "clientVersion": "curr",
        "param": {"cmsArticleWebOld": {
            "searchScope": "default", "sort": "default",
            "pageIndex": 1, "pageSize": limit,
            "preTag": "", "postTag": ""
        }}
    }, ensure_ascii=False)
    url = f'https://search-api-web.eastmoney.com/search/jsonp?cb=&param={quote(param)}'
    text = fetch(url)
    if not text: return []
    try:
        raw = text.strip()
        if raw.startswith('('): raw = raw[1:]
        if raw.endswith(');'): raw = raw[:-2]
        elif raw.endswith(')'): raw = raw[:-1]
        d = json.loads(raw)
        items = d.get('result', {}).get('cmsArticleWebOld', [])
        news = []
        for it in items:
            title = it.get('title', '').strip()
            title = re.sub(r'<[^>]+>', '', title)  # strip HTML tags
            if title:
                news.append(dict(
                    title=title, source='东方财富',
                    media=it.get('mediaName', ''),
                    time=it.get('date', '')[:16]
                ))
        return news
    except:
        return []

# ═══════════════════════════════════════
# 数据源3: 同花顺诊股
# ═══════════════════════════════════════
def fetch_ths_diagnosis(code):
    """同花顺AI诊股数据"""
    c = code.replace('.SH','').replace('.SZ','').replace('.BJ','')
    url = f'http://doctor.10jqka.com.cn/{c}/'
    html = fetch(url)
    if not html: return None
    result = {}
    m = re.search(r'gnzf["\']>\s*(\d+(?:\.\d+)?)', html)
    if m: result['score'] = float(m.group(1))
    dims = re.findall(r'(技术面|资金面|基本面|消息面)[^<]*?(\d+(?:\.\d+)?)\s*分', html)
    for name, score in dims:
        result[name] = float(score)
    m2 = re.search(r'class="doctor_title"[^>]*>(.*?)</div>', html, re.DOTALL)
    if m2: result['conclusion'] = re.sub(r'<[^>]+>', '', m2.group(1)).strip()[:100]
    return result if result else None

# ═══════════════════════════════════════
# 数据源4: 新浪快讯(按名称过滤)
# ═══════════════════════════════════════
def fetch_sina_news(name, days=7, limit=20):
    """新浪财经快讯全量 → 过滤含股票名的"""
    news = []
    cutoff = time.time() - days * 86400
    for page in range(1, 4):
        url = f'https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2516&num=50&page={page}'
        text = fetch(url)
        if not text: break
        try:
            data = json.loads(text)
            items = data.get('result', {}).get('data', [])
            for it in items:
                ts = int(it.get('ctime', 0))
                if ts < cutoff: continue
                title = it.get('title', '').strip()
                if name in title:
                    news.append(dict(title=title, source='新浪快讯',
                                    time=datetime.fromtimestamp(ts).strftime('%m-%d %H:%M')))
        except:
            break
        if len(news) >= limit: break
        time.sleep(0.3)
    return news[:limit]

BULL_WORDS = ['看好','利好','买入','增持','推荐','目标价','上涨','突破','底部','反弹','金叉',
              '放量','强势','龙头','主力','加仓','回调买','低估','超预期','景气','高增长',
              '翻倍','牛','爆发','起飞','机会','建仓','核心资产','确定性']
BEAR_WORDS = ['看空','利空','卖出','减持','回避','下跌','破位','顶部','高估','风险','死叉',
              '缩量','弱势','套牢','出货','止损','泡沫','不及预期','业绩雷','暴雷',
              '减仓','割肉','见顶','跳水','崩','跌停']
NEUTRAL_WORDS = ['观望','震荡','持有','中性','分化','不确定','等待','横盘']

def analyze_sentiment(text):
    """简易情感打分 → -1~+1"""
    if not text: return 0
    bull = sum(1 for w in BULL_WORDS if w in text)
    bear = sum(1 for w in BEAR_WORDS if w in text)
    neutral = sum(1 for w in NEUTRAL_WORDS if w in text)
    total = bull + bear + neutral
    if total == 0: return 0
    score = (bull - bear) / total
    return max(-1, min(1, score))

def rating_to_score(rating):
    """评级文字 → 数值(-1~+1)"""
    r = str(rating).strip()
    mapping = {
        '买入': 1.0, '强推': 1.0, '强烈推荐': 1.0,
        '增持': 0.7, '推荐': 0.7, '跑赢大市': 0.7, '优于大市': 0.7,
        '审慎增持': 0.5,
        '中性': 0.0, '持有': 0.0, '同步大市': 0.0,
        '减持': -0.7, '卖出': -1.0, '回避': -0.8,
    }
    for k, v in mapping.items():
        if k in r: return v
    return 0

# ═══════════════════════════════════════
# 可视化: 共识仪表盘
# ═══════════════════════════════════════
def draw_consensus(data, name, code, output_path):
    """绘制共识仪表盘"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Arc, Wedge
    import matplotlib.patheffects as pe

    plt.rcParams['font.sans-serif'] = ['PingFang SC','Hiragino Sans GB','STHeiti','SimHei','Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    BG='#0d1117'; PNL='#161b22'; GRD='#21262d'
    TXT='#e6edf3'; LBL='#8b949e'
    BULL='#3fb950'; BEAR='#f85149'; NEU='#d29922'

    fig = plt.figure(figsize=(16, 20), facecolor=BG)

    # === 标题区 ===
    fig.text(0.5, 0.97, f'{name} ({code})', fontsize=24, fontweight='bold',
             color=TXT, ha='center', va='top')
    fig.text(0.5, 0.955, f'舆情共识分析  {datetime.now().strftime("%Y-%m-%d %H:%M")}',
             fontsize=11, color=LBL, ha='center', va='top')

    # 数据提取
    consensus_score = data.get('consensus_score', 0)  # -1 ~ +1
    bull_pct = data.get('bull_pct', 0)
    bear_pct = data.get('bear_pct', 0)
    neu_pct = data.get('neutral_pct', 0)
    ratings = data.get('ratings', [])
    guba_sentiment = data.get('guba_sentiment', data.get('news_sentiment', 0))
    news_sentiment = data.get('news_sentiment', 0)
    ths_data = data.get('ths_diagnosis', {})
    top_opinions = data.get('top_opinions', [])
    rating_dist = data.get('rating_distribution', {})
    orgs = data.get('org_ratings', [])

    # === 1. 共识仪表盘 (大半圆) ===
    ax_gauge = fig.add_axes([0.1, 0.74, 0.8, 0.2])
    ax_gauge.set_xlim(-1.5, 1.5); ax_gauge.set_ylim(-0.3, 1.2)
    ax_gauge.set_facecolor(BG); ax_gauge.axis('off')

    # 半圆弧 (绿-黄-红)
    for angle, color in [(0, BEAR), (60, NEU), (120, BULL)]:
        w = Wedge((0, 0), 1.0, angle, angle+60, width=0.15, fc=color, alpha=0.3)
        ax_gauge.add_patch(w)

    # 指针
    ptr_angle = 90 + consensus_score * 90  # -1→0°, 0→90°, +1→180°
    ptr_rad = np.radians(ptr_angle)
    ax_gauge.annotate('', xy=(0.85*np.cos(ptr_rad), 0.85*np.sin(ptr_rad)),
                      xytext=(0, 0),
                      arrowprops=dict(arrowstyle='->', color=TXT, lw=3))

    # 刻度标签
    for val, label in [(-1, '极度看空'), (-0.5, '看空'), (0, '中性'), (0.5, '看多'), (1, '极度看多')]:
        a = np.radians(90 + val * 90)
        ax_gauge.text(1.15*np.cos(a), 1.15*np.sin(a), label,
                     fontsize=8, color=LBL, ha='center', va='center')

    # 中心分数
    score_color = BULL if consensus_score > 0.2 else BEAR if consensus_score < -0.2 else NEU
    label_text = '看多' if consensus_score > 0.2 else '看空' if consensus_score < -0.2 else '中性'
    ax_gauge.text(0, -0.15, f'{consensus_score:+.2f}', fontsize=28, fontweight='bold',
                 color=score_color, ha='center', va='center',
                 path_effects=[pe.withStroke(linewidth=3, foreground=BG)])
    ax_gauge.text(0, -0.28, label_text, fontsize=14, color=score_color, ha='center')

    # === 2. 多头/空头/中性占比 (三色条) ===
    ax_bar = fig.add_axes([0.1, 0.70, 0.8, 0.03])
    ax_bar.set_facecolor(BG); ax_bar.axis('off')
    ax_bar.set_xlim(0, 1); ax_bar.set_ylim(0, 1)

    x = 0
    for pct, color, label in [(bull_pct, BULL, '看多'), (neu_pct, NEU, '中性'), (bear_pct, BEAR, '看空')]:
        if pct > 0:
            ax_bar.barh(0.5, pct/100, left=x, height=0.8, color=color, alpha=0.8)
            if pct > 8:
                ax_bar.text(x + pct/200, 0.5, f'{label} {pct:.0f}%', fontsize=9,
                           color='white', ha='center', va='center', fontweight='bold')
            x += pct/100

    # === 3. 各维度雷达/评分 ===
    dims_data = []
    if ths_data:
        for k in ['技术面','资金面','基本面','消息面']:
            if k in ths_data:
                dims_data.append((k, ths_data[k]))
    if guba_sentiment != 0:
        dims_data.append(('新闻情绪', (guba_sentiment + 1) * 50))  # -1~+1 → 0~100
    if news_sentiment != 0:
        dims_data.append(('新闻情绪', (news_sentiment + 1) * 50))

    if dims_data:
        ax_radar = fig.add_axes([0.05, 0.52, 0.45, 0.18])
        ax_radar.set_facecolor(PNL)
        for s in ax_radar.spines.values(): s.set_color(GRD)
        ax_radar.tick_params(colors=LBL, labelsize=8)

        names_d = [d[0] for d in dims_data]
        vals = [d[1] for d in dims_data]
        colors_d = [BULL if v > 60 else BEAR if v < 40 else NEU for v in vals]

        bars = ax_radar.barh(range(len(dims_data)), vals, color=colors_d, alpha=0.7, height=0.6)
        ax_radar.set_yticks(range(len(dims_data)))
        ax_radar.set_yticklabels(names_d, fontsize=9, color=TXT)
        ax_radar.set_xlim(0, 100)
        ax_radar.set_xlabel('分数', fontsize=8, color=LBL)
        ax_radar.axvline(50, color=LBL, lw=0.5, ls='--', alpha=0.4)
        for i, v in enumerate(vals):
            ax_radar.text(v + 1, i, f'{v:.0f}', fontsize=9, color=TXT, va='center')
        ax_radar.set_title('多维度评分', fontsize=11, color=TXT, pad=8, loc='left')
        ax_radar.grid(axis='x', alpha=0.1, color=GRD)

    # === 4. 机构评级分布 (饼图) ===
    if rating_dist:
        ax_pie = fig.add_axes([0.55, 0.52, 0.4, 0.18])
        ax_pie.set_facecolor(PNL)

        labels_p = list(rating_dist.keys())
        sizes = list(rating_dist.values())
        colors_p = []
        for l in labels_p:
            s = rating_to_score(l)
            colors_p.append(BULL if s > 0.3 else BEAR if s < -0.3 else NEU)

        if sum(sizes) > 0:
            wedges, texts, autotexts = ax_pie.pie(sizes, labels=labels_p, autopct='%1.0f%%',
                colors=colors_p, textprops=dict(color=TXT, fontsize=8),
                pctdistance=0.75, startangle=90)
            for at in autotexts: at.set_fontsize(8); at.set_color('white')
            ax_pie.set_title(f'机构评级分布 ({sum(sizes)}家)', fontsize=11, color=TXT, pad=8, loc='left')

    # === 5. 机构评级时间线 ===
    if orgs:
        ax_tl = fig.add_axes([0.05, 0.30, 0.9, 0.20])
        ax_tl.set_facecolor(PNL)
        for s in ax_tl.spines.values(): s.set_color(GRD)
        ax_tl.tick_params(colors=LBL, labelsize=7)

        orgs_show = orgs[:15]
        y_labels = [f"{o['org'][:8]}" for o in orgs_show]
        x_scores = [rating_to_score(o.get('rating', '')) for o in orgs_show]
        colors_o = [BULL if s > 0.3 else BEAR if s < -0.3 else NEU for s in x_scores]
        dates = [o.get('report_date', '')[-5:] for o in orgs_show]

        ax_tl.barh(range(len(orgs_show)), x_scores, color=colors_o, alpha=0.7, height=0.6)
        ax_tl.set_yticks(range(len(orgs_show)))
        ax_tl.set_yticklabels(y_labels, fontsize=8, color=TXT)
        ax_tl.set_xlim(-1.2, 1.2)
        ax_tl.axvline(0, color=LBL, lw=0.5)
        for i, (sc, dt, o) in enumerate(zip(x_scores, dates, orgs_show)):
            r = o.get('rating', '')
            ax_tl.text(sc + (0.05 if sc >= 0 else -0.05), i,
                      f'{r} ({dt})', fontsize=7, color=TXT,
                      va='center', ha='left' if sc >= 0 else 'right')
        ax_tl.set_title('机构评级详情 (近期)', fontsize=11, color=TXT, pad=8, loc='left')
        ax_tl.grid(axis='x', alpha=0.1, color=GRD)

    # === 6. 舆情关键词/观点摘要 ===
    if top_opinions:
        ax_op = fig.add_axes([0.05, 0.03, 0.9, 0.25])
        ax_op.set_facecolor(PNL)
        ax_op.axis('off')
        ax_op.set_title('近期重要观点', fontsize=11, color=TXT, pad=8, loc='left')

        y = 0.92
        for i, op in enumerate(top_opinions[:12]):
            src = op.get('source', '')
            title = op.get('title', '')[:60]
            sentiment = op.get('sentiment', 0)
            marker = '🟢' if sentiment > 0.2 else '🔴' if sentiment < -0.2 else '🟡'
            score_t = f'{sentiment:+.1f}' if sentiment != 0 else '  0'

            ax_op.text(0.0, y, marker, fontsize=9, transform=ax_op.transAxes, va='top')
            ax_op.text(0.03, y, f'[{src}]', fontsize=8, color=LBL, transform=ax_op.transAxes, va='top')
            ax_op.text(0.15, y, title, fontsize=8, color=TXT, transform=ax_op.transAxes, va='top')
            ax_op.text(0.95, y, score_t, fontsize=8,
                      color=BULL if sentiment > 0.2 else BEAR if sentiment < -0.2 else NEU,
                      transform=ax_op.transAxes, va='top', ha='right')
            y -= 0.075

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Chart saved: {output_path}")

# ═══════════════════════════════════════
# 主流程
# ═══════════════════════════════════════
def run(code, name, output_dir, days=7):
    os.makedirs(output_dir, exist_ok=True)
    c = code.replace('.SH','').replace('.SZ','').replace('.BJ','')

    print(f"\n{'='*50}")
    print(f"  舆情共识分析: {name} ({code})")
    print(f"{'='*50}\n")

    # 1. 采集
    print("[1/4] 东方财富机构研报评级...")
    em_ratings = fetch_em_report_ratings(code, months=6)
    print(f"  → {len(em_ratings)} 条评级")

    print("[2/4] 东方财富相关新闻...")
    em_news = fetch_em_news(name, limit=15)
    print(f"  → {len(em_news)} 条新闻")

    print("[3/4] 同花顺诊股...")
    ths = fetch_ths_diagnosis(code)
    print(f"  → {'有数据' if ths else '无数据'}")

    print("[4/4] 新浪快讯...")
    sina_news = fetch_sina_news(name, days=days)
    print(f"  → {len(sina_news)} 条快讯")

    # 合并新闻
    all_news = em_news + sina_news

    # 2. 情感分析
    print("\n[分析] 计算情感得分...")

    # 新闻情感
    news_scores = [analyze_sentiment(n['title']) for n in all_news]
    news_avg = np.mean(news_scores) if news_scores else 0

    # 机构评级
    rating_scores = [rating_to_score(r.get('rating', '')) for r in em_ratings]
    rating_avg = np.mean(rating_scores) if rating_scores else 0

    # 同花顺评分
    ths_score = 0
    if ths and 'score' in ths:
        ths_score = (ths['score'] - 50) / 50  # 0~100 → -1~+1

    # 加权共识分数 (机构权重最高)
    weights = []
    scores = []
    if rating_scores:
        weights.append(4.0); scores.append(rating_avg)
    if ths_score != 0:
        weights.append(2.0); scores.append(ths_score)
    if news_scores:
        weights.append(1.5); scores.append(news_avg)

    consensus = np.average(scores, weights=weights) if scores else 0
    consensus = max(-1, min(1, consensus))

    # 多空比例
    all_sentiments = news_scores + rating_scores
    if all_sentiments:
        bull_pct = sum(1 for s in all_sentiments if s > 0.15) / len(all_sentiments) * 100
        bear_pct = sum(1 for s in all_sentiments if s < -0.15) / len(all_sentiments) * 100
        neu_pct = 100 - bull_pct - bear_pct
    else:
        bull_pct = bear_pct = 0; neu_pct = 100

    # 评级分布
    rating_dist = {}
    for r in em_ratings:
        rt = r.get('rating', '').strip()
        if rt: rating_dist[rt] = rating_dist.get(rt, 0) + 1

    # 合并观点列表
    top_opinions = []
    for r in em_ratings[:15]:
        s = rating_to_score(r.get('rating', ''))
        top_opinions.append(dict(
            title=f"{r.get('org','')}: {r.get('rating','')} — {r.get('title','')[:40]}",
            source='研报', sentiment=s))
    for n in all_news:
        s = analyze_sentiment(n['title'])
        top_opinions.append(dict(title=n['title'], source=n.get('source','新闻'), sentiment=s))

    # 按|sentiment|排序，取最有态度的
    top_opinions.sort(key=lambda x: abs(x.get('sentiment', 0)), reverse=True)

    print(f"  共识分数: {consensus:+.2f}")
    print(f"  多头: {bull_pct:.0f}%  空头: {bear_pct:.0f}%  中性: {neu_pct:.0f}%")
    print(f"  新闻情绪: {news_avg:+.2f}  机构评级: {rating_avg:+.2f}")

    # 3. 构建数据
    result = dict(
        code=code, name=name,
        date=datetime.now().strftime('%Y-%m-%d %H:%M'),
        consensus_score=round(consensus, 3),
        consensus_label='看多' if consensus > 0.2 else '看空' if consensus < -0.2 else '中性',
        bull_pct=round(bull_pct, 1),
        bear_pct=round(bear_pct, 1),
        neutral_pct=round(neu_pct, 1),
        news_sentiment=round(news_avg, 3),
        rating_avg=round(rating_avg, 3),
        ths_diagnosis=ths or {},
        rating_distribution=rating_dist,
        org_ratings=[dict(org=r.get('org',''), rating=r.get('rating',''),
                         report_date=r.get('report_date',''), title=r.get('title','')[:50])
                    for r in em_ratings[:15]],
        top_opinions=top_opinions[:15],
        sources=dict(em_ratings=len(em_ratings), em_news=len(em_news),
                    sina_news=len(sina_news), ths=bool(ths))
    )

    # 4. 保存JSON
    json_path = os.path.join(output_dir, f'{c}_consensus.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nJSON: {json_path}")

    # 5. 绘图
    print("[绘图] 生成共识仪表盘...")
    png_path = os.path.join(output_dir, f'{c}_consensus.png')
    draw_consensus(result, name, code, png_path)

    return result


def update_reports_json(result, repo_dir):
    """将分析结果写入 sentiment-reports.json 并复制图表到 charts/"""
    reports_path = os.path.join(repo_dir, 'sentiment-reports.json')
    charts_dir = os.path.join(repo_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)

    # 读取现有报告
    reports = []
    if os.path.exists(reports_path):
        try:
            with open(reports_path) as f:
                reports = json.load(f)
        except:
            reports = []

    # 准备前端记录
    c = result['code'].replace('.SH','').replace('.SZ','').replace('.BJ','')
    chart_rel = f'charts/{c}_consensus.png'
    entry = dict(
        name=result['name'],
        code=result['code'],
        date=result['date'],
        consensus_score=result['consensus_score'],
        consensus_label=result['consensus_label'],
        bull_pct=result['bull_pct'],
        bear_pct=result['bear_pct'],
        neutral_pct=result['neutral_pct'],
        news_sentiment=result.get('news_sentiment', 0),
        rating_avg=result.get('rating_avg', 0),
        ths_diagnosis=result.get('ths_diagnosis', {}),
        rating_distribution=result.get('rating_distribution', {}),
        org_ratings=result.get('org_ratings', []),
        top_opinions=result.get('top_opinions', []),
        sources=result.get('sources', {}),
        chart=chart_rel
    )

    # 更新或追加（同code覆盖）
    updated = False
    for i, r in enumerate(reports):
        if r.get('code') == result['code']:
            reports[i] = entry
            updated = True
            break
    if not updated:
        reports.append(entry)

    # 写入
    with open(reports_path, 'w') as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print(f"Updated: {reports_path} ({len(reports)} reports)")

    # 复制图表
    src_png = os.path.join(os.path.dirname(reports_path), '..', 'tmp', f'{c}_consensus.png')
    # 图表可能在output目录，由调用者复制
    return chart_rel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='舆情共识分析')
    parser.add_argument('--code', required=True, help='股票代码 (如 600519 或 600519.SH)')
    parser.add_argument('--name', required=True, help='股票名称')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--days', type=int, default=7, help='新闻回溯天数')
    parser.add_argument('--repo', default='', help='仓库根目录(自动写入sentiment-reports.json)')
    args = parser.parse_args()

    result = run(args.code, args.name, args.output, args.days)
    print(f"\n完成! 共识: {result['consensus_label']} ({result['consensus_score']:+.3f})")

    # 自动写入前端数据
    if args.repo:
        repo = args.repo
    else:
        # 自动检测仓库目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))

    if os.path.exists(os.path.join(repo, 'sentiment-reports.json')) or os.path.exists(os.path.join(repo, 'index.html')):
        import shutil
        c = result['code'].replace('.SH','').replace('.SZ','').replace('.BJ','')
        charts_dir = os.path.join(repo, 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        src_png = os.path.join(args.output, f'{c}_consensus.png')
        dst_png = os.path.join(charts_dir, f'{c}_consensus.png')
        if os.path.exists(src_png):
            shutil.copy2(src_png, dst_png)
            print(f"Chart copied: {dst_png}")
        update_reports_json(result, repo)
    else:
        print(f"[INFO] 未检测到仓库，跳过前端数据写入")
