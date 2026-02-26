#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情绪战神策略 — 情绪周期扫描器 + 龙头识别

数据源：akshare (东方财富涨停复盘)
输出：情绪周期判断 + 龙头候选列表

用法:
  python3 sentiment_scanner.py --date 20260226 --output /tmp/sentiment_scan.json
  python3 sentiment_scanner.py --date 20260226 --output /tmp/sentiment_scan.json --history 5
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# 数据缓存路径
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
SENTIMENT_CACHE = DATA_DIR / "sentiment_history.json"


def fetch_zt_data(date_str):
    """获取指定日期的涨停/跌停/炸板数据"""
    import akshare as ak
    
    result = {"date": date_str, "zt": [], "dt": [], "zb": []}
    
    # 涨停池
    try:
        df = ak.stock_zt_pool_em(date=date_str)
        for _, row in df.iterrows():
            result["zt"].append({
                "code": str(row["代码"]),
                "name": str(row["名称"]),
                "change_pct": float(row["涨跌幅"]) if row["涨跌幅"] else 0,
                "price": float(row["最新价"]) if row["最新价"] else 0,
                "amount": float(row["成交额"]) if row["成交额"] else 0,
                "float_mv": float(row["流通市值"]) if row["流通市值"] else 0,
                "turnover": float(row["换手率"]) if row["换手率"] else 0,
                "seal_fund": float(row["封板资金"]) if row["封板资金"] else 0,
                "first_seal": str(row["首次封板时间"]) if row["首次封板时间"] else "",
                "last_seal": str(row["最后封板时间"]) if row["最后封板时间"] else "",
                "open_times": int(row["炸板次数"]) if row["炸板次数"] else 0,
                "zt_stats": str(row["涨停统计"]) if row["涨停统计"] else "",
                "streak": int(row["连板数"]) if row["连板数"] else 1,
                "industry": str(row["所属行业"]) if row["所属行业"] else "",
            })
    except Exception as e:
        print(f"[WARN] 涨停池获取失败 {date_str}: {e}", file=sys.stderr)
    
    # 跌停池
    try:
        df2 = ak.stock_zt_pool_dtgc_em(date=date_str)
        for _, row in df2.iterrows():
            result["dt"].append({
                "code": str(row.get("代码", "")),
                "name": str(row.get("名称", "")),
            })
    except Exception as e:
        print(f"[WARN] 跌停池获取失败: {e}", file=sys.stderr)
    
    # 炸板池
    try:
        df3 = ak.stock_zt_pool_zbgc_em(date=date_str)
        for _, row in df3.iterrows():
            result["zb"].append({
                "code": str(row.get("代码", "")),
                "name": str(row.get("名称", "")),
            })
    except Exception as e:
        print(f"[WARN] 炸板池获取失败: {e}", file=sys.stderr)
    
    return result


def calc_sentiment_phase(zt_count, dt_count, zb_count, max_streak, ladder):
    """
    情绪周期五阶段判定
    
    返回: (phase, temperature, details)
    phase: ice/startup/ferment/climax/ebb
    temperature: 0-100
    """
    # 涨跌停比
    zt_dt_ratio = zt_count / max(dt_count, 1)
    # 封板成功率
    seal_rate = zt_count / max(zt_count + zb_count, 1) * 100
    # 连板高度
    height = max_streak
    # 高位板数量（3板以上）
    high_streak_count = sum(v for k, v in ladder.items() if k >= 3)
    
    # ===== 五档温度计 =====
    temperature = 0
    
    # 涨停家数贡献 (0-30分)
    if zt_count >= 80:
        temperature += 30
    elif zt_count >= 50:
        temperature += 20
    elif zt_count >= 30:
        temperature += 12
    elif zt_count >= 15:
        temperature += 5
    
    # 涨跌停比贡献 (0-25分)
    if zt_dt_ratio >= 10:
        temperature += 25
    elif zt_dt_ratio >= 5:
        temperature += 18
    elif zt_dt_ratio >= 2:
        temperature += 10
    elif zt_dt_ratio >= 1:
        temperature += 5
    
    # 封板成功率贡献 (0-20分)
    if seal_rate >= 80:
        temperature += 20
    elif seal_rate >= 65:
        temperature += 12
    elif seal_rate >= 50:
        temperature += 5
    
    # 连板高度贡献 (0-15分)
    if height >= 7:
        temperature += 15
    elif height >= 5:
        temperature += 10
    elif height >= 3:
        temperature += 5
    
    # 高位板梯队贡献 (0-10分)
    if high_streak_count >= 5:
        temperature += 10
    elif high_streak_count >= 3:
        temperature += 6
    elif high_streak_count >= 1:
        temperature += 3
    
    # ===== 阶段判定 =====
    if temperature <= 20:
        phase = "ice"          # 冰点
    elif temperature <= 40:
        phase = "startup"      # 启动
    elif temperature <= 60:
        phase = "ferment"      # 发酵
    elif temperature <= 80:
        phase = "climax"       # 高潮
    else:
        phase = "overheated"   # 过热（退潮前兆）
    
    details = {
        "zt_count": zt_count,
        "dt_count": dt_count,
        "zb_count": zb_count,
        "zt_dt_ratio": round(zt_dt_ratio, 2),
        "seal_rate": round(seal_rate, 1),
        "max_streak": height,
        "high_streak_count": high_streak_count,
        "ladder": {str(k): v for k, v in sorted(ladder.items(), reverse=True)},
    }
    
    return phase, temperature, details


def detect_ebb(history):
    """
    退潮检测：连板高度断裂 + 情绪温度连降
    需要至少2天历史数据
    """
    if len(history) < 2:
        return False, ""
    
    today = history[-1]
    yesterday = history[-2]
    
    reasons = []
    
    # 1. 连板高度断裂（最高板比昨天少2个以上）
    height_drop = yesterday.get("max_streak", 0) - today.get("max_streak", 0)
    if height_drop >= 2:
        reasons.append(f"连板高度断裂({yesterday['max_streak']}→{today['max_streak']})")
    
    # 2. 涨停家数腰斩
    zt_drop_pct = 1 - today.get("zt_count", 0) / max(yesterday.get("zt_count", 1), 1)
    if zt_drop_pct >= 0.4:
        reasons.append(f"涨停家数骤降{zt_drop_pct*100:.0f}%")
    
    # 3. 跌停家数飙升
    if today.get("dt_count", 0) >= 10 and today.get("dt_count", 0) >= yesterday.get("dt_count", 0) * 2:
        reasons.append(f"跌停飙升({yesterday.get('dt_count',0)}→{today.get('dt_count',0)})")
    
    # 4. 温度连续下降（如果有3天数据）
    if len(history) >= 3:
        temps = [h.get("temperature", 50) for h in history[-3:]]
        if temps[0] > temps[1] > temps[2] and temps[0] - temps[2] >= 20:
            reasons.append(f"情绪连降3天({temps[0]}→{temps[2]})")
    
    is_ebb = len(reasons) >= 2
    return is_ebb, "; ".join(reasons)


def identify_leaders(zt_data, min_streak=2):
    """
    龙头股识别
    
    排序逻辑：
    1. 连板数（越高越强）
    2. 封板时间（越早越强）
    3. 炸板次数（越少越强）
    4. 封单占比（封板资金/流通市值，越高越强）
    """
    candidates = []
    
    for stock in zt_data:
        streak = stock.get("streak", 1)
        if streak < min_streak:
            continue
        
        # 封单占比
        seal_ratio = 0
        if stock.get("float_mv", 0) > 0:
            seal_ratio = stock.get("seal_fund", 0) / stock["float_mv"] * 100
        
        # 封板时间评分（越早越好，09:25=100, 14:57=0）
        first_seal = stock.get("first_seal", "")
        seal_time_score = 0
        if first_seal:
            try:
                h, m = int(first_seal[:2]), int(first_seal[2:4])
                minutes_from_open = (h - 9) * 60 + (m - 25)
                seal_time_score = max(0, 100 - minutes_from_open / 3.3)  # 0~330分钟映射到100~0
            except:
                seal_time_score = 50
        
        # 综合评分
        score = 0
        score += streak * 30                              # 连板数权重最高
        score += seal_time_score * 0.3                    # 封板时间
        score += max(0, 20 - stock.get("open_times", 0) * 10)  # 炸板惩罚
        score += min(seal_ratio * 5, 30)                  # 封单占比
        
        # 辨识度标记
        tags = []
        if streak >= 5:
            tags.append("绝对龙头")
        elif streak >= 3:
            tags.append("高位龙头")
        if first_seal and first_seal <= "093000":
            tags.append("秒板")
        elif first_seal and first_seal <= "100000":
            tags.append("早封")
        if stock.get("open_times", 0) == 0:
            tags.append("一封不开")
        if seal_ratio >= 10:
            tags.append("巨量封单")
        elif seal_ratio >= 5:
            tags.append("强封单")
        
        candidates.append({
            **stock,
            "seal_ratio": round(seal_ratio, 2),
            "seal_time_score": round(seal_time_score, 1),
            "leader_score": round(score, 1),
            "tags": tags,
        })
    
    candidates.sort(key=lambda x: -x["leader_score"])
    return candidates


def generate_action(phase, temperature, leaders, is_ebb, ebb_reasons, holding=False):
    """
    生成交易动作建议
    
    核心原则：
    - 冰点/退潮 → 绝对空仓
    - 启动期 → 轻仓试错，只买最强龙头
    - 发酵期 → 重仓出击，聚焦绝对龙头
    - 高潮期 → 兑现获利，不追高
    - 过热 → 准备撤退
    """
    action = {
        "phase": phase,
        "temperature": temperature,
        "recommendation": "hold",
        "position_pct": 0,
        "target": None,
        "reason": "",
        "urgency": "low",
    }
    
    if phase == "ice" or is_ebb:
        action["recommendation"] = "clear_all"
        action["position_pct"] = 0
        action["reason"] = f"情绪{'退潮' if is_ebb else '冰点'}，绝对空仓" + (f" ({ebb_reasons})" if ebb_reasons else "")
        action["urgency"] = "high" if holding else "low"
        return action
    
    if phase == "overheated":
        action["recommendation"] = "reduce"
        action["position_pct"] = 30
        action["reason"] = "情绪过热，兑现获利准备撤退"
        action["urgency"] = "high" if holding else "medium"
        return action
    
    if not leaders:
        action["recommendation"] = "wait"
        action["position_pct"] = 0
        action["reason"] = "无合格龙头，等待机会"
        return action
    
    top_leader = leaders[0]
    
    if phase == "startup":
        action["recommendation"] = "light_buy"
        action["position_pct"] = 50
        action["target"] = top_leader
        action["reason"] = f"情绪启动期，轻仓试错{top_leader['name']}({top_leader['streak']}板)"
        action["urgency"] = "medium"
    
    elif phase == "ferment":
        action["recommendation"] = "heavy_buy"
        action["position_pct"] = 100
        action["target"] = top_leader
        action["reason"] = f"情绪发酵期，重仓出击{top_leader['name']}({top_leader['streak']}板)"
        action["urgency"] = "high"
    
    elif phase == "climax":
        if holding:
            action["recommendation"] = "reduce"
            action["position_pct"] = 50
            action["reason"] = "情绪高潮期，兑现一半获利"
            action["urgency"] = "medium"
        else:
            action["recommendation"] = "light_buy"
            action["position_pct"] = 30
            action["target"] = top_leader
            action["reason"] = f"情绪高潮期，轻仓跟随{top_leader['name']}（控制风险）"
            action["urgency"] = "low"
    
    return action


def scan(date_str, history_days=5, output_file=None):
    """主扫描函数"""
    
    print(f"=== 情绪战神扫描 {date_str} ===", file=sys.stderr)
    
    # 获取今日数据
    today_data = fetch_zt_data(date_str)
    zt_count = len(today_data["zt"])
    dt_count = len(today_data["dt"])
    zb_count = len(today_data["zb"])
    
    print(f"涨停: {zt_count}  跌停: {dt_count}  炸板: {zb_count}", file=sys.stderr)
    
    # 连板梯队
    ladder = {}
    for stock in today_data["zt"]:
        s = stock["streak"]
        ladder[s] = ladder.get(s, 0) + 1
    max_streak = max(ladder.keys()) if ladder else 0
    
    # 情绪周期判定
    phase, temperature, phase_details = calc_sentiment_phase(
        zt_count, dt_count, zb_count, max_streak, ladder
    )
    
    print(f"情绪周期: {phase} (温度={temperature})", file=sys.stderr)
    print(f"连板梯队: {dict(sorted(ladder.items(), reverse=True))}", file=sys.stderr)
    
    # 加载历史数据用于退潮检测
    history = []
    if SENTIMENT_CACHE.exists():
        with open(SENTIMENT_CACHE) as f:
            cached = json.load(f)
        history = cached.get("history", [])
    
    # 今日记录
    today_record = {
        "date": date_str,
        "zt_count": zt_count,
        "dt_count": dt_count,
        "zb_count": zb_count,
        "max_streak": max_streak,
        "temperature": temperature,
        "phase": phase,
    }
    
    # 退潮检测
    is_ebb, ebb_reasons = detect_ebb(history + [today_record])
    if is_ebb:
        phase = "ebb"
        print(f"⚠️ 退潮信号: {ebb_reasons}", file=sys.stderr)
    
    # 龙头识别
    leaders = identify_leaders(today_data["zt"], min_streak=2)
    print(f"龙头候选: {len(leaders)}只", file=sys.stderr)
    for l in leaders[:5]:
        tags = " ".join(l["tags"]) if l.get("tags") else ""
        print(f"  {l['code']} {l['name']} {l['streak']}板 "
              f"封单占比{l['seal_ratio']:.1f}% score={l['leader_score']:.0f} [{tags}]",
              file=sys.stderr)
    
    # 交易建议
    action = generate_action(phase, temperature, leaders, is_ebb, ebb_reasons)
    
    phase_names = {
        "ice": "❄️ 冰点", "startup": "🌱 启动", "ferment": "🔥 发酵",
        "climax": "🎆 高潮", "overheated": "🌋 过热", "ebb": "🌊 退潮"
    }
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"情绪周期: {phase_names.get(phase, phase)} (温度={temperature})", file=sys.stderr)
    print(f"建议: {action['recommendation']} 仓位={action['position_pct']}%", file=sys.stderr)
    print(f"理由: {action['reason']}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    
    # 更新缓存
    history.append(today_record)
    history = history[-30:]  # 保留最近30天
    with open(SENTIMENT_CACHE, "w") as f:
        json.dump({"history": history, "updated": date_str}, f, ensure_ascii=False, indent=2)
    
    # 输出结果
    result = {
        "scan_date": date_str,
        "phase": phase,
        "phase_name": phase_names.get(phase, phase),
        "temperature": temperature,
        "is_ebb": is_ebb,
        "ebb_reasons": ebb_reasons,
        "phase_details": phase_details,
        "leaders": leaders[:10],
        "action": action,
        "raw_counts": {
            "zt": zt_count, "dt": dt_count, "zb": zb_count,
        },
        "ladder": {str(k): v for k, v in sorted(ladder.items(), reverse=True)},
        "all_zt": today_data["zt"],  # 完整涨停池
    }
    
    if output_file:
        with open(output_file, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="情绪战神 — 情绪周期扫描")
    parser.add_argument("--date", required=True, help="日期 YYYYMMDD")
    parser.add_argument("--output", default="/tmp/sentiment_scan.json", help="输出文件")
    parser.add_argument("--history", type=int, default=0, help="回溯N天历史（初始化用）")
    args = parser.parse_args()
    
    if args.history > 0:
        # 回溯模式：获取最近N天数据建立历史基线
        print(f"回溯模式：获取最近{args.history}天数据...", file=sys.stderr)
        from datetime import datetime, timedelta
        base = datetime.strptime(args.date, "%Y%m%d")
        for i in range(args.history, 0, -1):
            d = base - timedelta(days=i)
            ds = d.strftime("%Y%m%d")
            # 跳过周末
            if d.weekday() >= 5:
                continue
            try:
                print(f"\n--- {ds} ---", file=sys.stderr)
                scan(ds, output_file=None)
            except Exception as e:
                print(f"Skip {ds}: {e}", file=sys.stderr)
    
    # 扫描目标日期
    result = scan(args.date, output_file=args.output)
    
    # 标准输出摘要
    print(json.dumps({
        "date": result["scan_date"],
        "phase": result["phase"],
        "temperature": result["temperature"],
        "leaders": len(result["leaders"]),
        "action": result["action"]["recommendation"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
