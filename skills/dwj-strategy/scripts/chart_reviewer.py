#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势游侠 AI看图复评模块
=======================
scanner.py筛出候选后，批量生成K线图，输出标准化图片供AI视觉分析。

流程：
1. 读取scanner输出的候选列表
2. 从kline_cache加载K线数据
3. 用matplotlib生成标准化K线图（日线120根+成交量+均线+KDJ）
4. 输出图片到指定目录
5. 生成review_manifest.json（候选清单+图片路径+量化指标摘要）

AI复评由上层agent（赛博巴菲特）逐张看图打分，不在本脚本内调用LLM。

用法:
  python3 chart_reviewer.py --signals /tmp/dwj_signals.json --output /tmp/dwj_charts/
  python3 chart_reviewer.py --signals /tmp/dwj_signals.json --output /tmp/dwj_charts/ --top-n 10
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime

import numpy as np

# matplotlib headless
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

# 路径
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent.parent / "data"
CACHE_FILE = DATA_DIR / "a_share_kline_cache.json"

# 中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['font.monospace'] = ['WenQuanYi Zen Hei Mono', 'DejaVu Sans Mono']
plt.rcParams['axes.unicode_minus'] = False

# 颜色 — 暗色主题
BG_COLOR = '#1a1a2e'
PANEL_COLOR = '#16213e'
TEXT_COLOR = '#e0e0e0'
GRID_COLOR = '#2a2a4a'
UP_COLOR = '#ef5350'
DOWN_COLOR = '#26a69a'
UP_BODY = '#ef5350'
DOWN_BODY = '#26a69a'
MA_COLORS = {
    'BBI': '#ffd700',
    'MA60': '#00bcd4',
    'MA5': '#ff9800',
    'MA10': '#2196f3',
    'MA20': '#9c27b0',
}
VOL_UP = '#ef535088'
VOL_DOWN = '#26a69a88'
KDJ_K = '#ff9800'
KDJ_D = '#2196f3'
KDJ_J = '#e040fb'


def load_kline_cache():
    """加载K线缓存"""
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    print(f"[ERROR] K线缓存不存在: {CACHE_FILE}", file=sys.stderr)
    return {}


def calc_ma(closes, n):
    """简单移动平均"""
    result = np.full(len(closes), np.nan)
    if len(closes) >= n:
        cumsum = np.cumsum(np.insert(closes, 0, 0))
        result[n-1:] = (cumsum[n:] - cumsum[:-n]) / n
    return result


def calc_bbi(closes):
    """BBI = (MA3+MA6+MA12+MA24)/4"""
    ma3 = calc_ma(closes, 3)
    ma6 = calc_ma(closes, 6)
    ma12 = calc_ma(closes, 12)
    ma24 = calc_ma(closes, 24)
    return (ma3 + ma6 + ma12 + ma24) / 4


def calc_kdj(highs, lows, closes, n=9):
    """KDJ指标"""
    length = len(closes)
    rsv = np.full(length, 50.0)
    for i in range(n-1, length):
        hh = np.max(highs[i-n+1:i+1])
        ll = np.min(lows[i-n+1:i+1])
        if hh != ll:
            rsv[i] = (closes[i] - ll) / (hh - ll) * 100
    
    k = np.full(length, 50.0)
    d = np.full(length, 50.0)
    for i in range(1, length):
        k[i] = (2 * k[i-1] + rsv[i]) / 3
        d[i] = (2 * d[i-1] + k[i]) / 3
    j = 3 * k - 2 * d
    return k, d, j


def calc_vol_ma(volumes, n):
    """成交量均线"""
    return calc_ma(volumes, n)


def generate_review_chart(klines, code, name, price_now, output_path, 
                          show_bars=120, candidate_info=None):
    """
    生成用于AI复评的标准化K线图
    
    布局（暗色主题，3面板）：
    - 主图：K线 + BBI + MA60 + MA5/MA10/MA20（120根日线）
    - 成交量：红绿柱 + 5日/20日均量线
    - KDJ：K/D/J三线 + 超买超卖区
    
    右上角标注：代码/名称/价格/涨跌幅/scanner评分/信号类型
    """
    # 解析K线数据
    dates = [k[0] for k in klines]
    opens = np.array([k[1] for k in klines], dtype=float)
    closes = np.array([k[2] for k in klines], dtype=float)
    highs = np.array([k[3] for k in klines], dtype=float)
    lows = np.array([k[4] for k in klines], dtype=float)
    volumes = np.array([k[5] if len(k) > 5 else 0 for k in klines], dtype=float)
    
    # 更新最后一根为实时价
    if price_now and price_now > 0:
        closes[-1] = price_now
    
    # 截取最后show_bars根
    n = len(closes)
    start = max(0, n - show_bars)
    dates = dates[start:]
    opens = opens[start:]
    closes = closes[start:]
    highs = highs[start:]
    lows = lows[start:]
    volumes = volumes[start:]
    
    # 计算指标（用完整数据再截取，确保长周期MA准确）
    full_closes = np.array([k[2] for k in klines], dtype=float)
    full_highs = np.array([k[3] for k in klines], dtype=float)
    full_lows = np.array([k[4] for k in klines], dtype=float)
    full_volumes = np.array([k[5] if len(k) > 5 else 0 for k in klines], dtype=float)
    if price_now and price_now > 0:
        full_closes[-1] = price_now
    
    bbi_full = calc_bbi(full_closes)
    ma60_full = calc_ma(full_closes, 60)
    ma5_full = calc_ma(full_closes, 5)
    ma10_full = calc_ma(full_closes, 10)
    ma20_full = calc_ma(full_closes, 20)
    k_full, d_full, j_full = calc_kdj(full_highs, full_lows, full_closes)
    vol_ma5_full = calc_vol_ma(full_volumes, 5)
    vol_ma20_full = calc_vol_ma(full_volumes, 20)
    
    # 截取显示部分
    bbi = bbi_full[start:]
    ma60 = ma60_full[start:]
    ma5 = ma5_full[start:]
    ma10 = ma10_full[start:]
    ma20 = ma20_full[start:]
    k_val = k_full[start:]
    d_val = d_full[start:]
    j_val = j_full[start:]
    vol_ma5 = vol_ma5_full[start:]
    vol_ma20 = vol_ma20_full[start:]
    
    bars = len(closes)
    x = np.arange(bars)
    
    # ============ 画图 ============
    fig = plt.figure(figsize=(14, 8), facecolor=BG_COLOR)
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.8], hspace=0.02)
    
    # ---------- 主图：K线+均线 ----------
    ax1 = fig.add_subplot(gs[0], facecolor=PANEL_COLOR)
    
    # K线
    width = 0.6
    for i in range(bars):
        is_up = closes[i] >= opens[i]
        color = UP_COLOR if is_up else DOWN_COLOR
        body_color = UP_BODY if is_up else DOWN_BODY
        
        # 影线
        ax1.plot([i, i], [lows[i], highs[i]], color=color, linewidth=0.7)
        # 实体
        body_bottom = min(opens[i], closes[i])
        body_height = abs(closes[i] - opens[i])
        if body_height < (highs[i] - lows[i]) * 0.005:
            body_height = (highs[i] - lows[i]) * 0.005  # 十字星最小高度
        rect = Rectangle((i - width/2, body_bottom), width, body_height,
                         facecolor=body_color, edgecolor=color, linewidth=0.7)
        ax1.add_patch(rect)
    
    # 均线
    for arr, label, color, lw in [
        (bbi, 'BBI', MA_COLORS['BBI'], 1.5),
        (ma60, 'MA60', MA_COLORS['MA60'], 1.5),
        (ma5, 'MA5', MA_COLORS['MA5'], 0.8),
        (ma10, 'MA10', MA_COLORS['MA10'], 0.8),
        (ma20, 'MA20', MA_COLORS['MA20'], 0.8),
    ]:
        valid = ~np.isnan(arr)
        if valid.any():
            ax1.plot(x[valid], arr[valid], color=color, linewidth=lw, label=label, alpha=0.9)
    
    ax1.legend(loc='upper left', fontsize=7, framealpha=0.6, facecolor=PANEL_COLOR, 
               edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax1.set_xlim(-1, bars)
    ax1.set_ylabel('价格', fontsize=9, color=TEXT_COLOR)
    ax1.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax1.grid(True, alpha=0.2, color=GRID_COLOR)
    ax1.set_xticklabels([])
    
    # 右上角信息标注
    change_pct = candidate_info.get('change_pct', 0) if candidate_info else 0
    score = candidate_info.get('score', 0) if candidate_info else 0
    sig_type = candidate_info.get('signal_type', '') if candidate_info else ''
    j_now = candidate_info.get('j', j_val[-1]) if candidate_info else j_val[-1]
    
    info_text = (
        f"{name} ({code})\n"
        f"¥{closes[-1]:.2f}  {change_pct:+.2f}%\n"
        f"J={j_now:.1f}  Score={score}\n"
        f"信号: {sig_type}"
    )
    ax1.text(0.98, 0.97, info_text, transform=ax1.transAxes,
             fontsize=9, color=TEXT_COLOR, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=BG_COLOR, edgecolor=GRID_COLOR, alpha=0.8),
             fontfamily='sans-serif')
    
    # ---------- 成交量 ----------
    ax2 = fig.add_subplot(gs[1], facecolor=PANEL_COLOR, sharex=ax1)
    
    # 自适应单位
    max_vol = np.max(volumes) if np.max(volumes) > 0 else 1
    if max_vol > 1e8:
        vol_div, vol_unit = 1e8, '亿'
    elif max_vol > 1e4:
        vol_div, vol_unit = 1e4, '万'
    else:
        vol_div, vol_unit = 1, ''
    
    vol_colors = [VOL_UP if closes[i] >= opens[i] else VOL_DOWN for i in range(bars)]
    ax2.bar(x, volumes / vol_div, color=vol_colors, width=0.6)
    
    # 均量线
    valid5 = ~np.isnan(vol_ma5)
    valid20 = ~np.isnan(vol_ma20)
    if valid5.any():
        ax2.plot(x[valid5], vol_ma5[valid5] / vol_div, color=MA_COLORS['MA5'], linewidth=0.8, label='Vol MA5')
    if valid20.any():
        ax2.plot(x[valid20], vol_ma20[valid20] / vol_div, color=MA_COLORS['MA20'], linewidth=0.8, label='Vol MA20')
    
    ax2.set_ylabel(f'成交量({vol_unit})', fontsize=8, color=TEXT_COLOR)
    ax2.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax2.grid(True, alpha=0.2, color=GRID_COLOR)
    ax2.legend(loc='upper left', fontsize=6, framealpha=0.6, facecolor=PANEL_COLOR,
               edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax2.set_xticklabels([])
    
    # ---------- KDJ ----------
    ax3 = fig.add_subplot(gs[2], facecolor=PANEL_COLOR, sharex=ax1)
    
    ax3.plot(x, k_val, color=KDJ_K, linewidth=0.9, label='K')
    ax3.plot(x, d_val, color=KDJ_D, linewidth=0.9, label='D')
    ax3.plot(x, j_val, color=KDJ_J, linewidth=0.9, label='J')
    ax3.axhline(y=80, color='#ef535055', linewidth=0.5, linestyle='--')
    ax3.axhline(y=20, color='#26a69a55', linewidth=0.5, linestyle='--')
    ax3.axhline(y=13, color='#ffd70055', linewidth=0.5, linestyle=':')
    ax3.fill_between(x, 80, 100, alpha=0.05, color=UP_COLOR)
    ax3.fill_between(x, 0, 20, alpha=0.05, color=DOWN_COLOR)
    
    ax3.set_ylim(-10, 110)
    ax3.set_ylabel('KDJ', fontsize=8, color=TEXT_COLOR)
    ax3.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax3.grid(True, alpha=0.2, color=GRID_COLOR)
    ax3.legend(loc='upper left', fontsize=6, framealpha=0.6, facecolor=PANEL_COLOR,
               edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    
    # X轴日期
    tick_step = max(1, bars // 10)
    tick_pos = list(range(0, bars, tick_step))
    ax3.set_xticks(tick_pos)
    ax3.set_xticklabels([dates[i][-5:] if len(dates[i]) > 5 else dates[i] for i in tick_pos],
                        rotation=45, fontsize=7, color=TEXT_COLOR)
    
    # 底部时间范围
    fig.text(0.99, 0.01, f"{dates[0]} ~ {dates[-1]}", ha='right', va='bottom',
             fontsize=7, color='#666666')
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    
    return output_path


def generate_review_manifest(signals_file, output_dir, top_n=20):
    """
    主函数：读取scanner输出 → 生成K线图 → 输出manifest
    
    返回 manifest dict:
    {
        "scan_time": "...",
        "chart_dir": "/tmp/dwj_charts/",
        "candidates": [
            {
                "code": "600519",
                "name": "贵州茅台",
                "price": 1580.0,
                "signal_type": "main",
                "score": 85,
                "j": -2.3,
                "chart_path": "/tmp/dwj_charts/600519.png",
                "signals": ["J=-2.3<13 + 上升趋势"],
                ...
            }
        ]
    }
    """
    # 读取scanner输出
    with open(signals_file) as f:
        scan_result = json.load(f)
    
    candidates = scan_result.get("top_candidates", [])[:top_n]
    if not candidates:
        print("[WARN] 无候选股票", file=sys.stderr)
        return {"scan_time": None, "chart_dir": str(output_dir), "candidates": []}
    
    # 加载K线缓存
    kline_cache = load_kline_cache()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_candidates = []
    
    for i, c in enumerate(candidates):
        code = c["code"]
        name = c.get("name", code)
        price = c.get("price", 0)
        
        if code not in kline_cache:
            print(f"[SKIP] {code} {name} — 无K线缓存", file=sys.stderr)
            continue
        
        klines = kline_cache[code]
        if len(klines) < 30:
            print(f"[SKIP] {code} {name} — K线不足30根", file=sys.stderr)
            continue
        
        chart_path = output_dir / f"{code}.png"
        
        try:
            generate_review_chart(
                klines=klines,
                code=code,
                name=name,
                price_now=price,
                output_path=str(chart_path),
                show_bars=120,
                candidate_info=c,
            )
            print(f"[{i+1}/{len(candidates)}] {code} {name} ✓", file=sys.stderr)
        except Exception as e:
            print(f"[{i+1}/{len(candidates)}] {code} {name} ✗ {e}", file=sys.stderr)
            continue
        
        manifest_candidates.append({
            "code": code,
            "name": name,
            "price": price,
            "change_pct": c.get("change_pct", 0),
            "signal_type": c.get("signal_type", ""),
            "score": c.get("score", 0),
            "j": c.get("j", 0),
            "k": c.get("k", 0),
            "d": c.get("d", 0),
            "bbi": c.get("bbi", 0),
            "ma60": c.get("ma60", 0),
            "volume_ratio": c.get("volume_ratio", 0),
            "market_cap": c.get("market_cap", 0),
            "industry": c.get("industry", ""),
            "signals": c.get("signals", []),
            "is_uptrend": c.get("is_uptrend", False),
            "chart_path": str(chart_path),
        })
    
    manifest = {
        "scan_time": scan_result.get("scan_time"),
        "chart_dir": str(output_dir),
        "total_scanned": scan_result.get("total_scanned", 0),
        "candidates_count": len(manifest_candidates),
        "candidates": manifest_candidates,
    }
    
    manifest_path = output_dir / "review_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    print(f"\n生成 {len(manifest_candidates)} 张K线图 → {output_dir}", file=sys.stderr)
    print(f"Manifest: {manifest_path}", file=sys.stderr)
    
    return manifest


# ============ AI复评评分框架（供agent调用） ============

REVIEW_PROMPT = """你是一名专业波段交易员，擅长仅凭股票日线图做主观交易评估。

## 任务
根据图表中的趋势、位置、量价、历史异动、追高风险，判断该股票当前是否具备波段爆发潜力。
这是纯视觉图形分析任务，只依据图中真实可见的信息判断。

## 评分维度（1-5分）

### 1. 趋势结构 (权重20%)
- 5分：均线刚进入多头结构，短期均线上穿长期均线
- 4分：多头排列已形成，均线整体向上
- 3分：均线偏多但不够理想，价格频繁跌破均线
- 2分：均线频繁交叉，趋势不清晰
- 1分：空头排列，均线向下

### 2. 价格位置 (权重20%)
- 5分：中低位刚突破平台，上方空间大
- 4分：脱离整理平台，正在向前高推进
- 3分：接近前期压力位
- 2分：接近历史高位，上方空间有限
- 1分：明显高位或过热区，远离均线

### 3. 量价行为 (权重30%)
- 5分：上涨放量+第一根阴线明显缩量，最大成交量在上涨阶段
- 4分：上涨有放量，回调基本缩量
- 3分：量价中性，上涨与回调量能差异不大
- 2分：上涨没有放量，回调不缩量
- 1分：放量大阴线，出货迹象

### 4. 前期建仓异动 (权重30%)
- 5分：异常放量中大阳突破关键位，涨幅<50%
- 4分：明显放量阳线，突破不够明显，涨幅<50%
- 3分：有一定放量上涨，但放量不够突出
- 2分：普通上涨或轻微放量
- 1分：主升浪已完成(涨幅>100%)或有出货迹象

## 信号类型（三选一）
- trend_start：主升启动
- rebound：跌后反弹
- distribution_risk：出货风险

## 判定
- PASS：total_score ≥ 4.0
- WATCH：3.2 ≤ total_score < 4.0
- FAIL：total_score < 3.2
- 特殊：volume_behavior = 1 → 必须 FAIL

## 输出格式（严格JSON）
{
  "trend_reasoning": "...",
  "position_reasoning": "...",
  "volume_reasoning": "...",
  "abnormal_move_reasoning": "...",
  "scores": {
    "trend_structure": N,
    "price_position": N,
    "volume_behavior": N,
    "previous_abnormal_move": N
  },
  "total_score": N.N,
  "signal_type": "trend_start|rebound|distribution_risk",
  "verdict": "PASS|WATCH|FAIL",
  "comment": "一句中文交易员点评"
}
"""


def calc_total_score(scores):
    """按权重计算总分"""
    weights = {
        'trend_structure': 0.20,
        'price_position': 0.20,
        'volume_behavior': 0.30,
        'previous_abnormal_move': 0.30,
    }
    total = sum(scores.get(k, 3) * w for k, w in weights.items())
    return round(total, 2)


def merge_review_results(manifest_path, reviews):
    """
    将AI复评结果合并到manifest中，生成最终推荐列表
    
    reviews: dict[code] = { scores, total_score, verdict, signal_type, comment, ... }
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    recommendations = []
    excluded = []
    
    for c in manifest["candidates"]:
        code = c["code"]
        review = reviews.get(code)
        if not review:
            excluded.append({"code": code, "reason": "未完成复评"})
            continue
        
        c["review"] = review
        c["review_score"] = review.get("total_score", 0)
        c["review_verdict"] = review.get("verdict", "FAIL")
        c["review_comment"] = review.get("comment", "")
        c["review_signal"] = review.get("signal_type", "")
        
        if review.get("verdict") == "PASS":
            recommendations.append(c)
        elif review.get("verdict") == "WATCH":
            recommendations.append(c)
        else:
            excluded.append({"code": code, "reason": f"FAIL (score={review.get('total_score', 0):.1f})"})
    
    # 按review_score降序
    recommendations.sort(key=lambda x: -x.get("review_score", 0))
    
    # 给推荐排名
    for i, r in enumerate(recommendations):
        r["rank"] = i + 1
    
    result = {
        "scan_time": manifest.get("scan_time"),
        "total_candidates": len(manifest["candidates"]),
        "pass_count": sum(1 for r in recommendations if r.get("review_verdict") == "PASS"),
        "watch_count": sum(1 for r in recommendations if r.get("review_verdict") == "WATCH"),
        "fail_count": len(excluded),
        "recommendations": recommendations,
        "excluded": excluded,
    }
    
    output_path = Path(manifest_path).parent / "review_result.json"
    with open(output_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="趋势游侠 AI看图复评 — K线图生成")
    parser.add_argument("--signals", required=True, help="scanner输出的候选JSON")
    parser.add_argument("--output", default="/tmp/dwj_charts/", help="图表输出目录")
    parser.add_argument("--top-n", type=int, default=20, help="处理前N只候选")
    args = parser.parse_args()
    
    manifest = generate_review_manifest(args.signals, args.output, args.top_n)
    
    # 输出摘要
    print(json.dumps({
        "candidates_count": manifest["candidates_count"],
        "chart_dir": manifest["chart_dir"],
    }))


if __name__ == "__main__":
    main()
