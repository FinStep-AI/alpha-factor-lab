#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势游侠策略 — 全A股扫描器
多级漏斗：基础过滤 → 趋势过滤 → 长阴排除 → 三通道信号 → 排序 → 板块聚类

三条选股通道：
1. 主策略：N型上升趋势 + J<13
2. 单针下30：MA多头排列上升趋势 + 短期急跌长期强势
3. 金叉缩量回调：BBI上穿MA60 + 放量上涨 + 缩量回调

用法:
  python3 scanner.py --quotes /tmp/a_share_quotes.json --output /tmp/dwj_signals.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# 数据路径
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
CACHE_FILE = DATA_DIR / "a_share_kline_cache.json"
INDUSTRY_FILE = DATA_DIR / "a_share_industry_map.json"


def load_kline_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    print("[ERROR] K-line cache not found. Run kline_cache.py init first.", file=sys.stderr)
    return {}


def load_industry_map():
    if INDUSTRY_FILE.exists():
        with open(INDUSTRY_FILE) as f:
            return json.load(f)
    print("[WARN] Industry map not found, sector clustering disabled.", file=sys.stderr)
    return {}


def calc_ma(closes, n):
    """计算移动平均"""
    if len(closes) < n:
        return None
    return np.mean(closes[-n:])


def calc_bbi(closes):
    """BBI = (MA3 + MA6 + MA12 + MA24) / 4"""
    if len(closes) < 24:
        return None
    ma3 = np.mean(closes[-3:])
    ma6 = np.mean(closes[-6:])
    ma12 = np.mean(closes[-12:])
    ma24 = np.mean(closes[-24:])
    return (ma3 + ma6 + ma12 + ma24) / 4


def calc_kdj(highs, lows, closes, n=9, m1=3, m2=3):
    """计算KDJ指标，返回最近的K, D, J值"""
    if len(closes) < n + m1 + m2:
        return None, None, None
    
    rsv_list = []
    for i in range(len(closes)):
        if i < n - 1:
            rsv_list.append(50.0)
            continue
        hh = max(highs[i-n+1:i+1])
        ll = min(lows[i-n+1:i+1])
        if hh == ll:
            rsv_list.append(50.0)
        else:
            rsv_list.append((closes[i] - ll) / (hh - ll) * 100)
    
    # 递推计算K, D
    k_list = [50.0]
    d_list = [50.0]
    for i in range(1, len(rsv_list)):
        k = (2 * k_list[-1] + rsv_list[i]) / 3
        d = (2 * d_list[-1] + k) / 3
        k_list.append(k)
        d_list.append(d)
    
    k = k_list[-1]
    d = d_list[-1]
    j = 3 * k - 2 * d
    return k, d, j


def calc_needle30(highs, lows, closes):
    """
    单针下30指标
    短期 = 100*(C-LLV(L,3))/(HHV(C,3)-LLV(L,3))
    长期 = 100*(C-LLV(L,21))/(HHV(C,21)-LLV(L,21))
    """
    if len(closes) < 22:  # 需要至少22天（21+1天REF）
        return None
    
    def calc_indicator(closes_arr, lows_arr, highs_arr, n):
        c = closes_arr[-1]
        ll = min(lows_arr[-n:])
        hh = max(closes_arr[-n:])  # 注意：HHV用的是C不是H
        if hh == ll:
            return 50.0
        return 100 * (c - ll) / (hh - ll)
    
    # 今日
    short_today = calc_indicator(closes, lows, highs, 3)
    long_today = calc_indicator(closes, lows, highs, 21)
    
    # 昨日（去掉最后一天）
    short_yesterday = calc_indicator(closes[:-1], lows[:-1], highs[:-1], 3)
    long_yesterday = calc_indicator(closes[:-1], lows[:-1], highs[:-1], 21)
    
    return {
        "short_today": short_today,
        "long_today": long_today,
        "short_yesterday": short_yesterday,
        "long_yesterday": long_yesterday,
        "triggered": (long_today >= 85 and short_today <= 30 and 
                     long_yesterday >= 85 and short_yesterday >= 85)
    }


def detect_uptrend(highs, lows, closes, lookback=60):
    """
    检测N型上升波段：低点抬高 + 高点抬高
    使用局部极值点检测
    """
    if len(closes) < min(lookback, 30):
        return False, 0
    
    data = closes[-lookback:] if len(closes) >= lookback else closes
    n = len(data)
    
    # 找局部极值点（窗口=5）
    window = 5
    local_highs = []
    local_lows = []
    
    for i in range(window, n - window):
        if data[i] == max(data[i-window:i+window+1]):
            local_highs.append((i, data[i]))
        if data[i] == min(data[i-window:i+window+1]):
            local_lows.append((i, data[i]))
    
    if len(local_highs) < 2 or len(local_lows) < 2:
        return False, 0
    
    # 取最近3个高低点判断
    recent_highs = local_highs[-3:]
    recent_lows = local_lows[-3:]
    
    # 低点抬高
    lows_rising = all(recent_lows[i][1] < recent_lows[i+1][1] 
                      for i in range(len(recent_lows)-1))
    # 高点抬高
    highs_rising = all(recent_highs[i][1] < recent_highs[i+1][1] 
                       for i in range(len(recent_highs)-1))
    
    # 趋势强度分数 0-1
    score = 0
    if lows_rising:
        score += 0.5
    if highs_rising:
        score += 0.5
    
    return (lows_rising and highs_rising), score


def check_bbi_ma60_golden_cross(closes, lookback=5):
    """检查BBI是否刚上穿MA60（最近lookback天内），返回金叉位置索引或None"""
    if len(closes) < 61:
        return None
    
    for i in range(max(1, len(closes) - lookback), len(closes)):
        subset = closes[:i+1]
        prev_subset = closes[:i]
        
        if len(prev_subset) < 60:
            continue
        
        bbi_now = (np.mean(subset[-3:]) + np.mean(subset[-6:]) + 
                   np.mean(subset[-12:]) + np.mean(subset[-24:])) / 4
        ma60_now = np.mean(subset[-60:])
        
        bbi_prev = (np.mean(prev_subset[-3:]) + np.mean(prev_subset[-6:]) + 
                    np.mean(prev_subset[-12:]) + np.mean(prev_subset[-24:])) / 4
        ma60_prev = np.mean(prev_subset[-60:])
        
        if bbi_prev <= ma60_prev and bbi_now > ma60_now:
            return i  # 返回金叉位置
    
    return None


def detect_big_selloff(klines, lookback=5, vol_ratio_threshold=2.0, drop_threshold=-0.015):
    """
    检测近期是否有长阴放量（大资金撤退信号）
    条件：近lookback天内出现 阴线(C<O) + 跌幅≥1.5% + 成交量≥20日均量的2倍
    返回：(has_selloff, details)
    """
    if len(klines) < 25:
        return False, None
    
    # kline格式: [date, open, close, high, low, volume]
    volumes = [k[5] for k in klines if len(k) > 5 and k[5] > 0]
    if len(volumes) < 20:
        return False, None
    
    # 20日均量（不含最后lookback天，用之前的数据作为基准）
    base_end = max(len(klines) - lookback, 20)
    avg_vol_20 = np.mean([k[5] for k in klines[base_end-20:base_end] if len(k) > 5 and k[5] > 0])
    if avg_vol_20 <= 0:
        return False, None
    
    recent = klines[-lookback:]
    for k in recent:
        if len(k) < 6:
            continue
        date, open_p, close_p, high_p, low_p, vol = k[0], k[1], k[2], k[3], k[4], k[5]
        if open_p <= 0 or vol <= 0:
            continue
        
        # 阴线: 收盘 < 开盘
        is_bearish = close_p < open_p
        # 跌幅
        change = (close_p - open_p) / open_p
        # 放量
        vol_ratio = vol / avg_vol_20
        
        if is_bearish and change <= drop_threshold and vol_ratio >= vol_ratio_threshold:
            return True, {
                "date": date,
                "change_pct": round(change * 100, 2),
                "vol_ratio": round(vol_ratio, 2),
            }
    
    return False, None


def detect_ma_uptrend(closes):
    """
    严格MA多头排列上升趋势（用于单针下30通道）
    条件1：MA5 > MA10 > MA15 > MA30 > MA60
    条件2：五条均线斜率均为正（今日MA > 昨日MA）
    """
    if len(closes) < 62:  # 需要61天算昨日MA60 + 今日MA60
        return False
    
    periods = [5, 10, 15, 30, 60]
    ma_today = {}
    ma_yesterday = {}
    
    for p in periods:
        ma_today[p] = np.mean(closes[-p:])
        ma_yesterday[p] = np.mean(closes[-(p+1):-1])
    
    # 条件1：完美多头排列
    aligned = (ma_today[5] > ma_today[10] > ma_today[15] > 
               ma_today[30] > ma_today[60])
    if not aligned:
        return False
    
    # 条件2：所有均线斜率为正
    all_rising = all(ma_today[p] > ma_yesterday[p] for p in periods)
    
    return aligned and all_rising


def detect_golden_pullback(closes, volumes, lookback=20):
    """
    BBI金叉 + 量价齐升 + 缩量回调 选股信号（第三通道）
    
    三步检测：
    1. 近20日内BBI上穿MA60
    2. 金叉后出现放量上涨段（量增价升）
    3. 之后出现缩量回调段（价跌量缩，量缩至放量段的50%以下越好）
    
    返回：(triggered, details)
    """
    if len(closes) < 65 or len(volumes) < 65:
        return False, None
    
    # Step 1: 找近20日内的BBI金叉点
    gc_idx = check_bbi_ma60_golden_cross(closes, lookback=lookback)
    if gc_idx is None:
        return False, None
    
    # 金叉后到当前的数据
    post_gc_closes = closes[gc_idx:]
    post_gc_volumes = volumes[gc_idx:]
    
    if len(post_gc_closes) < 3:  # 金叉后至少3天才能有"放量上涨+缩量回调"
        return False, None
    
    # Step 2: 找放量上涨段
    # 金叉前20日均量作为基准
    pre_gc_start = max(0, gc_idx - 20)
    pre_gc_vols = [v for v in volumes[pre_gc_start:gc_idx] if v > 0]
    if not pre_gc_vols:
        return False, None
    base_avg_vol = np.mean(pre_gc_vols)
    if base_avg_vol <= 0:
        return False, None
    
    # 找高点（金叉后的最高收盘价位置）
    peak_idx = np.argmax(post_gc_closes)
    if peak_idx < 1:  # 高点至少在金叉后第2天
        return False, None
    
    # 放量上涨段：金叉到高点
    up_closes = post_gc_closes[:peak_idx + 1]
    up_volumes = post_gc_volumes[:peak_idx + 1]
    
    # 检查：价格上涨
    if up_closes[-1] <= up_closes[0]:
        return False, None
    
    # 检查：放量（上涨段均量 > 基准均量的1.5倍）
    up_avg_vol = np.mean([v for v in up_volumes if v > 0]) if any(v > 0 for v in up_volumes) else 0
    vol_amplify = up_avg_vol / base_avg_vol if base_avg_vol > 0 else 0
    if vol_amplify < 1.5:
        return False, None
    
    # Step 3: 缩量回调段：高点到当前
    if peak_idx >= len(post_gc_closes) - 1:
        return False, None  # 今天就是高点，还没回调
    
    pullback_closes = post_gc_closes[peak_idx:]
    pullback_volumes = post_gc_volumes[peak_idx:]
    
    # 检查：价格回调（当前价 < 高点价）
    if pullback_closes[-1] >= pullback_closes[0]:
        return False, None
    
    # 检查：缩量（回调段均量相比上涨段均量的比例，越低越好）
    pb_avg_vol = np.mean([v for v in pullback_volumes if v > 0]) if any(v > 0 for v in pullback_volumes) else 0
    vol_shrink = pb_avg_vol / up_avg_vol if up_avg_vol > 0 else 1.0
    
    if vol_shrink >= 0.6:  # 缩量不够明显（需缩至放量段60%以下）
        return False, None
    
    # 检查：回调期间量逐步递减（地量地价）
    valid_pb_vols = [v for v in pullback_volumes if v > 0]
    vol_declining = True
    if len(valid_pb_vols) >= 2:
        # 后半段均量 < 前半段均量
        mid = len(valid_pb_vols) // 2
        if mid > 0:
            first_half = np.mean(valid_pb_vols[:mid])
            second_half = np.mean(valid_pb_vols[mid:])
            vol_declining = second_half <= first_half
    
    if not vol_declining:
        return False, None
    
    return True, {
        "gc_days_ago": len(closes) - gc_idx,
        "vol_amplify": round(vol_amplify, 2),
        "vol_shrink_pct": round(vol_shrink * 100, 1),
        "pullback_pct": round((pullback_closes[-1] / pullback_closes[0] - 1) * 100, 2),
    }


def scan(quotes_file, output_file, top_n=30):
    """全A股扫描主函数 — 严格按趋势游侠策略过滤"""
    
    # 加载数据
    with open(quotes_file) as f:
        quotes = json.load(f)
    
    kline_cache = load_kline_cache()
    industry_map = load_industry_map()
    
    print(f"Loaded {len(quotes)} quotes, {len(kline_cache)} kline cache, "
          f"{len(industry_map)} industry mappings", file=sys.stderr)
    
    # ===== 第1级：基础过滤（排除垃圾/不可交易） =====
    level1 = {}
    for code, q in quotes.items():
        if q["volume"] <= 0 or q["price"] <= 0:
            continue
        if q["open"] <= 0:
            continue
        if q["price"] < 2:
            continue
        mc = q.get("market_cap", 0)
        if mc > 0 and (mc > 3000 or mc < 10):
            continue
        level1[code] = q
    
    print(f"Level 1 (basic): {len(quotes)} → {len(level1)}", file=sys.stderr)
    
    # ===== 第2级：趋势硬过滤（BBI > MA60 + 价 > BBI） =====
    level2 = {}
    for code, q in level1.items():
        if code not in kline_cache:
            continue
        klines = kline_cache[code]
        if len(klines) < 60:
            continue
        
        closes = np.array([k[2] for k in klines], dtype=float)
        closes[-1] = q["price"]  # 更新为实时价
        
        bbi = calc_bbi(closes)
        ma60 = calc_ma(closes, 60)
        if bbi is None or ma60 is None:
            continue
        
        # 硬条件: BBI > MA60 且 价 > BBI
        if bbi <= ma60 or q["price"] <= bbi:
            continue
        
        level2[code] = (q, klines, bbi, ma60)
    
    print(f"Level 2 (BBI>MA60, P>BBI): {len(level1)} → {len(level2)}", file=sys.stderr)
    
    # ===== 第2.5级：长阴放量排除（大资金撤退） =====
    level2b = {}
    selloff_count = 0
    for code, (q, klines, bbi, ma60) in level2.items():
        has_selloff, selloff_detail = detect_big_selloff(klines)
        if has_selloff:
            selloff_count += 1
            continue  # 直接排除
        level2b[code] = (q, klines, bbi, ma60)
    
    print(f"Level 2.5 (selloff filter): {len(level2)} → {len(level2b)} "
          f"(排除{selloff_count}只长阴放量)", file=sys.stderr)
    
    # ===== 第3级：上升区间 + KDJ硬过滤 =====
    # 主策略候选: N型上升趋势 AND J<13（硬条件）
    # 单针下30候选: MA多头排列上升趋势 + 单针下30条件
    # 金叉缩量回调候选: BBI金叉 + 放量上涨 + 缩量回调（第三通道）
    main_candidates = []
    needle_candidates = []
    pullback_candidates = []
    
    for code, (q, klines, bbi, ma60) in level2b.items():
        closes = np.array([k[2] for k in klines], dtype=float)
        highs = np.array([k[3] for k in klines], dtype=float)
        lows = np.array([k[4] for k in klines], dtype=float)
        volumes = np.array([k[5] if len(k) > 5 else 0 for k in klines], dtype=float)
        closes[-1] = q["price"]
        highs[-1] = max(highs[-1], q["high"]) if q["high"] > 0 else highs[-1]
        lows[-1] = min(lows[-1], q["low"]) if q["low"] > 0 else lows[-1]
        
        # KDJ
        k_val, d_val, j_val = calc_kdj(highs.tolist(), lows.tolist(), closes.tolist())
        if j_val is None:
            continue
        
        # 上升趋势
        is_uptrend, trend_score = detect_uptrend(highs.tolist(), lows.tolist(), closes.tolist())
        
        # BBI金叉（返回索引或None）
        gc_idx = check_bbi_ma60_golden_cross(closes.tolist())
        golden_cross = gc_idx is not None
        
        # 单针下30
        needle = calc_needle30(highs.tolist(), lows.tolist(), closes.tolist())
        needle_triggered = needle["triggered"] if needle else False
        
        # 量比/量价
        vr = q.get("volume_ratio", 0)
        volume_price_up = q["change_pct"] > 0 and vr > 1
        
        # 行业信息
        industry = industry_map.get(code, "未分类")
        
        base_info = {
            "code": code,
            "name": q["name"],
            "price": q["price"],
            "change_pct": q["change_pct"],
            "volume_ratio": vr,
            "market_cap": q.get("market_cap", 0),
            "bbi": round(bbi, 2),
            "ma60": round(ma60, 2),
            "j": round(j_val, 1),
            "k": round(k_val, 1),
            "d": round(d_val, 1),
            "is_uptrend": is_uptrend,
            "trend_score": round(trend_score, 2),
            "golden_cross": golden_cross,
            "needle30": needle_triggered,
            "industry": industry,
            "stop_loss": round(q["price"] - 5 * 0.01, 2),
        }
        
        # ===== 主策略：上升趋势 AND J<13（硬条件） =====
        if is_uptrend and j_val < 13:
            score = 50
            signals = [f"J={j_val:.1f}<13 + 上升趋势"]
            
            if golden_cross:
                score += 20
                signals.append("BBI刚上穿MA60")
            if needle_triggered:
                score += 25
                signals.append("单针下30（双信号）")
            if vr >= 8:
                score += 15
                signals.append(f"量比{vr:.1f}≥8")
            elif vr >= 3:
                score += 8
                signals.append(f"量比{vr:.1f}")
            if volume_price_up:
                score += 5
                signals.append("量价齐升")
            
            entry = {**base_info, "score": score, "signals": signals, "signal_type": "main"}
            main_candidates.append(entry)
        
        # ===== 单针下30（补充策略，MA多头排列上升趋势） =====
        if needle_triggered and detect_ma_uptrend(closes):
            score = 40
            signals = ["单针下30 + MA多头排列"]
            
            if j_val < 13:
                continue  # 已经在主策略里了
            
            if vr >= 8:
                score += 15
                signals.append(f"量比{vr:.1f}≥8")
            elif vr >= 3:
                score += 8
            if volume_price_up:
                score += 5
                signals.append("量价齐升")
            
            entry = {**base_info, "score": score, "signals": signals, "signal_type": "needle30"}
            needle_candidates.append(entry)
        
        # ===== 金叉缩量回调（第三通道，独立信号） =====
        pb_triggered, pb_detail = detect_golden_pullback(closes.tolist(), volumes.tolist())
        if pb_triggered:
            # 避免与主策略/单针下30重复
            already_in = (is_uptrend and j_val < 13) or (needle_triggered and detect_ma_uptrend(closes))
            if not already_in:
                score = 45
                signals = [f"BBI金叉{pb_detail['gc_days_ago']}天前 + 放量{pb_detail['vol_amplify']}x + 缩量回调{pb_detail['vol_shrink_pct']}%"]
                
                if pb_detail["vol_shrink_pct"] <= 50:
                    score += 15
                    signals.append("缩量至半量以下")
                if pb_detail["vol_amplify"] >= 2.0:
                    score += 10
                    signals.append(f"强放量{pb_detail['vol_amplify']}x")
                if vr >= 3:
                    score += 8
                    signals.append(f"量比{vr:.1f}")
                if volume_price_up:
                    score += 5
                    signals.append("量价齐升")
                
                entry = {**base_info, "score": score, "signals": signals, "signal_type": "pullback"}
                pullback_candidates.append(entry)
    
    # 合并候选
    all_candidates = main_candidates + needle_candidates + pullback_candidates
    
    print(f"Level 3 (hard filter): {len(level2b)} → "
          f"{len(main_candidates)} main + {len(needle_candidates)} needle30 "
          f"+ {len(pullback_candidates)} pullback "
          f"= {len(all_candidates)} total", file=sys.stderr)
    
    # ===== 第4级：排序（量比降序，量价齐升优先） =====
    all_candidates.sort(key=lambda x: (-x["score"], -x["volume_ratio"]))
    
    # ===== 第5级：板块聚类 =====
    industry_counts = {}
    for c in all_candidates:
        ind = c["industry"]
        industry_counts[ind] = industry_counts.get(ind, 0) + 1
    
    # 标记板块集中度（同行业3只以上 → 板块启动信号）
    hot_industries = {k for k, v in industry_counts.items() if v >= 3 and k != "未分类"}
    
    for c in all_candidates:
        if c["industry"] in hot_industries:
            c["score"] += 10
            c["signals"].append(f"板块集中:{c['industry']}({industry_counts[c['industry']]}只)")
    
    # 重新排序
    all_candidates.sort(key=lambda x: (-x["score"], -x["volume_ratio"]))
    
    # 输出
    sorted_industries = dict(sorted(industry_counts.items(), key=lambda x: -x[1])[:15])
    
    result = {
        "scan_time": None,
        "total_scanned": len(quotes),
        "level1_pass": len(level1),
        "level2_pass": len(level2),
        "selloff_filtered": selloff_count,
        "level2b_pass": len(level2b),
        "main_signals": len(main_candidates),
        "needle30_signals": len(needle_candidates),
        "pullback_signals": len(pullback_candidates),
        "candidates_count": len(all_candidates),
        "hot_industries": list(hot_industries),
        "top_candidates": all_candidates[:top_n],
        "industry_distribution": sorted_industries,
    }
    
    with open(output_file, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"长阴放量排除: {selloff_count}只", file=sys.stderr)
    print(f"主策略候选（J<13 + N型上升趋势）: {len(main_candidates)}只", file=sys.stderr)
    print(f"单针下30候选（MA多头排列）: {len(needle_candidates)}只", file=sys.stderr)
    print(f"金叉缩量回调候选: {len(pullback_candidates)}只", file=sys.stderr)
    if hot_industries:
        print(f"板块集中: {', '.join(hot_industries)}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    
    print(f"\nTop 10:", file=sys.stderr)
    for c in all_candidates[:10]:
        print(f"  [{c['signal_type']:8s}] {c['code']} {c['name']:6s} ¥{c['price']:.2f} "
              f"J={c['j']:.1f} VR={c['volume_ratio']:.1f} "
              f"[{c['industry']}] "
              f"{' | '.join(c['signals'][:2])}", file=sys.stderr)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="趋势游侠策略全A股扫描")
    parser.add_argument("--quotes", default="/tmp/a_share_quotes.json", help="实时行情JSON")
    parser.add_argument("--output", default="/tmp/dwj_signals.json", help="输出文件")
    parser.add_argument("--top-n", type=int, default=30, help="输出前N只候选")
    args = parser.parse_args()
    
    result = scan(args.quotes, args.output, args.top_n)
    print(json.dumps({
        "total_scanned": result["total_scanned"],
        "candidates": result["candidates_count"],
    }))


if __name__ == "__main__":
    main()
