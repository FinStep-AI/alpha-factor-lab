#!/usr/bin/env python3
"""
标准化净值更新脚本 — 收盘后统一刷新所有选手净值

用法：
  python3 scripts/nav_update_all.py --date 2026-03-10
  python3 scripts/nav_update_all.py --date 2026-03-10 --intraday  # 盘中刷新

功能：
  1. 读取所有选手持仓，收集全部股票代码
  2. 批量获取实时/收盘价格（腾讯行情API）
  3. 通过 trading_engine.update_all_navs() 更新净值
  4. 数据一致性自检（positions vs total_value vs nav）
  5. 保存数据
"""

import json
import os
import sys
import argparse
import urllib.request
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
ENGINE_DIR = REPO_DIR / "skills" / "paper-trading" / "scripts"
sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from trading_engine import (load_data, save_data, update_nav, 
                           update_benchmark, update_all_navs,
                           _update_stats, _update_leaderboard)

DEFAULT_DATA = str(REPO_DIR / "paper-trading-data.json")


def fetch_all_prices(codes: list) -> dict:
    """批量获取价格（腾讯行情）"""
    if not codes:
        return {}
    
    def to_tc(code):
        parts = code.split('.')
        if len(parts) == 2:
            num, ex = parts
            return f"{'sh' if ex in ('SH','SS') else 'sz'}{num}"
        return code
    
    prices = {}
    tc_list = [(c, to_tc(c)) for c in codes]
    
    for i in range(0, len(tc_list), 80):
        batch = tc_list[i:i+80]
        url = f"https://qt.gtimg.cn/q={','.join(tc for _, tc in batch)}"
        headers = {"Referer": "https://finance.qq.com"}
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=15)
            text = resp.read().decode('gbk')
        except Exception as e:
            print(f"  ⚠️ 批次{i//80}获取失败: {e}")
            continue
        
        for line in text.strip().split(';'):
            line = line.strip()
            if not line or '~' not in line:
                continue
            parts = line.split('~')
            if len(parts) > 33:
                raw_code = parts[2]
                price = float(parts[3]) if parts[3] else 0
                if price > 0:
                    for orig, tc in batch:
                        if orig.split('.')[0] == raw_code:
                            prices[orig] = price
                            break
    
    return prices


def fetch_benchmark_nav(data: dict) -> float:
    """获取沪深300指数当前点位，计算基准NAV"""
    try:
        url = "https://qt.gtimg.cn/q=sh000300"
        headers = {"Referer": "https://finance.qq.com"}
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req, timeout=10)
        text = resp.read().decode('gbk')
        parts = text.split('~')
        if len(parts) > 33:
            current_price = float(parts[3])
            start_value = data.get("benchmark", {}).get("start_value", 4707.54)
            nav = round(current_price / start_value, 6)
            print(f"  沪深300: {current_price:.2f}, 基准NAV: {nav:.6f}")
            return nav
    except Exception as e:
        print(f"  ⚠️ 获取沪深300失败: {e}")
    return None


def consistency_check(data: dict) -> list:
    """数据一致性自检"""
    issues = []
    initial = data["meta"]["initial_cash"]
    
    for pid, player in data["players"].items():
        port = player["portfolio"]
        cash = port["cash"]
        positions = port.get("positions", {})
        stored_tv = port["total_value"]
        
        # 检查1: volume=0 的幽灵持仓
        ghost = [c for c, p in positions.items() if (p.get("volume", 0) or p.get("shares", 0)) <= 0]
        if ghost:
            issues.append(f"❌ {player['name']}: {len(ghost)} 个volume=0持仓: {ghost}")
            # Auto-fix: remove ghost positions
            for c in ghost:
                del positions[c]
        
        # 检查2: total_value 是否等于 cash + market_value
        pos_value = sum(
            p.get("current_price", p.get("avg_cost", 0)) * (p.get("volume", 0) or p.get("shares", 0))
            for p in positions.values()
        )
        actual_tv = round(cash + pos_value, 2)
        if abs(stored_tv - actual_tv) > 1:
            issues.append(f"⚠️ {player['name']}: total_value不一致 stored={stored_tv:,.2f} actual={actual_tv:,.2f}")
            port["total_value"] = actual_tv
        
        # 检查3: NAV 是否和 total_value 一致
        nh = player.get("nav_history", {})
        if nh.get("nav"):
            latest_nav = nh["nav"][-1]
            expected_nav = round(port["total_value"] / initial, 6)
            if abs(latest_nav - expected_nav) > 0.001:
                issues.append(f"⚠️ {player['name']}: NAV不一致 stored={latest_nav} expected={expected_nav}")
        
        # 检查4: cash 为负
        if cash < 0:
            issues.append(f"❌ {player['name']}: cash为负 {cash:,.2f}")
    
    return issues


def main():
    parser = argparse.ArgumentParser(description="标准化净值更新")
    parser.add_argument("--date", required=True, help="日期 YYYY-MM-DD")
    parser.add_argument("--data", default=DEFAULT_DATA, help="数据文件路径")
    parser.add_argument("--intraday", action="store_true", help="盘中刷新（不追加nav_history）")
    parser.add_argument("--check-only", action="store_true", help="只做一致性检查，不更新")
    
    args = parser.parse_args()
    
    data = load_data(args.data)
    initial = data["meta"]["initial_cash"]
    
    # 一致性自检（先修复再更新）
    print("🔍 数据一致性检查...")
    issues = consistency_check(data)
    if issues:
        print(f"\n  发现 {len(issues)} 个问题:")
        for i in issues:
            print(f"    {i}")
        print("  已自动修复可修复的问题\n")
    else:
        print("  ✅ 数据一致\n")
    
    if args.check_only:
        if issues:
            save_data(data, args.data)
            print("💾 修复结果已保存")
        return
    
    # 收集所有持仓代码
    all_codes = set()
    for pid, player in data["players"].items():
        for code in player["portfolio"].get("positions", {}):
            all_codes.add(code)
    
    print(f"📊 更新净值 ({args.date}{'，盘中' if args.intraday else ''})")
    print(f"  总持仓代码: {len(all_codes)} 只")
    
    # 获取价格
    prices = fetch_all_prices(list(all_codes))
    print(f"  获取到 {len(prices)} 个价格")
    
    if len(prices) < len(all_codes) * 0.5 and all_codes:
        print(f"  ⚠️ 价格获取不足 ({len(prices)}/{len(all_codes)})")
    
    # 更新各选手净值
    for pid in data["players"]:
        player = data["players"][pid]
        positions = player["portfolio"].get("positions", {})
        if not positions:
            # 空仓选手也要更新NAV（保持cash=total_value）
            player["portfolio"]["total_value"] = player["portfolio"]["cash"]
            if not args.intraday:
                nh = player.get("nav_history", {"dates": [], "nav": [], "cash_pct": []})
                nav = round(player["portfolio"]["total_value"] / initial, 6)
                if nh["dates"] and nh["dates"][-1] == args.date:
                    nh["nav"][-1] = nav
                    if nh.get("cash_pct"):
                        nh["cash_pct"][-1] = 100.0
                else:
                    nh["dates"].append(args.date)
                    nh["nav"].append(nav)
                    if "cash_pct" in nh:
                        nh["cash_pct"].append(100.0)
                player["nav_history"] = nh
            continue
        
        # 有持仓的选手
        update_nav(data, pid, prices, args.date, intraday=args.intraday)
    
    # 更新基准
    benchmark_nav = fetch_benchmark_nav(data)
    if benchmark_nav:
        update_benchmark(data, benchmark_nav, args.date)
    
    # 更新排行榜
    for pid in data["players"]:
        _update_stats(data["players"][pid], initial)
    _update_leaderboard(data)
    
    # 最终一致性检查
    print("\n🔍 最终一致性检查...")
    final_issues = consistency_check(data)
    if final_issues:
        print(f"  ⚠️ 仍有 {len(final_issues)} 个问题:")
        for i in final_issues:
            print(f"    {i}")
    else:
        print("  ✅ 全部一致")
    
    # 保存
    save_data(data, args.data)
    print(f"\n💾 数据已保存到 {args.data}")
    
    # 输出排行榜
    print(f"\n{'='*60}")
    print("🏆 排行榜")
    print(f"{'='*60}")
    for r in data.get("leaderboard", {}).get("rankings", []):
        nav = r.get("current_nav", 1.0)
        ret = r.get("total_return_pct", 0)
        print(f"  #{r['rank']} {r['emoji']} {r['name']}: NAV={nav:.4f} ({ret:+.2f}%)")


if __name__ == "__main__":
    main()
