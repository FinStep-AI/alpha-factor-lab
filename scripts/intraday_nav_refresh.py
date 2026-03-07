#!/usr/bin/env python3
"""
盘中实时净值刷新 — 每半小时拉实时行情更新持仓盈亏+净值+排行榜
用法: python3 scripts/intraday_nav_refresh.py [--data paper-trading-data.json]

流程:
1. 从 paper-trading-data.json 提取所有选手持仓股票代码
2. 用腾讯行情API批量获取实时价格
3. 获取沪深300实时点位计算基准净值
4. 更新每只持仓的 current_price/market_value/pnl/pnl_pct
5. 更新每位选手的 portfolio.total_value + 盘中净值（覆盖今天的nav点）
6. 更新排行榜
7. 保存 + git push

网络失败时静默退出(exit 0)，不影响cron调度。
"""

import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

DATA_FILE = "paper-trading-data.json"
INITIAL_CASH = 10_000_000
# 沪深300起始点位 (2026-02-24开盘)
HS300_START = None  # 从data文件读取


def fetch_quotes_tencent(codes):
    """腾讯行情API批量获取实时价格
    codes: ["sh600519", "sz000858", ...]
    返回: {"sh600519": 1800.5, ...}
    """
    if not codes:
        return {}
    
    results = {}
    batch_size = 60
    batches = [codes[i:i+batch_size] for i in range(0, len(codes), batch_size)]
    
    for batch in batches:
        codes_str = ",".join(batch)
        url = f"https://qt.gtimg.cn/q={codes_str}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=10)
            raw = resp.read().decode("gbk", errors="ignore")
            for line in raw.strip().split("\n"):
                if "v_" not in line or "~" not in line:
                    continue
                var_name = line.split("=")[0].strip()
                code = var_name.replace("v_", "")
                fields = line.split("~")
                if len(fields) < 5:
                    continue
                price = _safe_float(fields[3])
                if price > 0:
                    results[code] = price
        except Exception as e:
            print(f"[WARN] batch fetch failed: {e}", file=sys.stderr)
    
    return results


def fetch_hs300_nav():
    """获取沪深300实时点位"""
    try:
        url = "https://qt.gtimg.cn/q=sh000300"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        raw = resp.read().decode("gbk", errors="ignore")
        for line in raw.strip().split("\n"):
            if "~" in line:
                fields = line.split("~")
                if len(fields) >= 4:
                    return _safe_float(fields[3])
    except Exception as e:
        print(f"[WARN] HS300 fetch failed: {e}", file=sys.stderr)
    return None


def _safe_float(s):
    try:
        return float(s)
    except:
        return 0.0


def code_to_tencent(code):
    """转换持仓代码到腾讯格式
    000603.SZ -> sz000603
    600519.SH -> sh600519
    002716 -> sz002716 (自动判断)
    603999 -> sh603999
    """
    if "." in code:
        num, market = code.split(".")
        if market.upper() in ("SH",):
            return f"sh{num}"
        else:
            return f"sz{num}"
    # 无后缀：按代码前缀判断市场
    num = code.lstrip("0123456789") or code  # 取纯数字
    num = code
    if num.startswith("6") or num.startswith("9"):
        return f"sh{num}"
    else:
        return f"sz{num}"


def tencent_to_position(tcode):
    """腾讯格式转回持仓代码
    sz000603 -> 000603.SZ
    sh600519 -> 600519.SH
    """
    if tcode.startswith("sh"):
        return f"{tcode[2:]}.SH"
    elif tcode.startswith("sz"):
        return f"{tcode[2:]}.SZ"
    return tcode


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_FILE)
    parser.add_argument("--no-push", action="store_true", help="不执行git push")
    parser.add_argument("--dry-run", action="store_true", help="只打印不保存")
    args = parser.parse_args()
    
    # 网络检测
    try:
        urllib.request.urlopen("https://qt.gtimg.cn/q=sh000001", timeout=5)
    except:
        print("[SKIP] 网络不通，跳过", file=sys.stderr)
        sys.exit(0)
    
    # 加载数据
    if not os.path.exists(args.data):
        print(f"[ERROR] {args.data} not found", file=sys.stderr)
        sys.exit(1)
    
    with open(args.data, 'r') as f:
        data = json.load(f)
    
    today = datetime.now().strftime("%Y-%m-%d")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 收集所有持仓代码 + 确定每个选手的实际数据位置
    all_codes = set()
    player_pos_refs = {}  # pid -> (positions_dict, cash_value)
    
    for pid, player in data["players"].items():
        # 优先用 portfolio.positions（trader/dwj/quant都有）
        port = player.get("portfolio", {})
        port_pos = port.get("positions", {})
        top_pos = player.get("positions", {})
        
        # 选择非空的那个，优先portfolio
        if port_pos and isinstance(port_pos, dict):
            positions = port_pos
            cash = port.get("cash", player.get("cash", 0))
        elif top_pos and isinstance(top_pos, dict):
            positions = top_pos
            cash = player.get("cash", 0)
        else:
            positions = {}
            cash = player.get("cash", port.get("cash", INITIAL_CASH))
        
        player_pos_refs[pid] = (positions, cash)
        for code in positions.keys():
            all_codes.add(code)
    
    if not all_codes:
        print("[INFO] 所有选手均空仓，仅更新基准", file=sys.stderr)
    
    # 转换代码格式
    code_map = {}  # tencent_code -> position_code
    tencent_codes = []
    for code in all_codes:
        tc = code_to_tencent(code)
        code_map[tc] = code
        tencent_codes.append(tc)
    
    # 获取实时行情
    print(f"[INFO] 获取 {len(tencent_codes)} 只股票实时行情...", file=sys.stderr)
    quotes = fetch_quotes_tencent(tencent_codes)
    
    # 转换回持仓代码格式的价格dict
    prices = {}
    for tc, price in quotes.items():
        pos_code = code_map.get(tc, tencent_to_position(tc))
        prices[pos_code] = price
    
    print(f"[INFO] 获取到 {len(prices)}/{len(all_codes)} 只实时价格", file=sys.stderr)
    
    # 获取沪深300
    hs300_now = fetch_hs300_nav()
    
    # 读取基准起始值
    hs300_start = data.get("benchmark", {}).get("start_value")
    if not hs300_start:
        # 尝试从第一个nav点推算
        bm_navs = data.get("benchmark", {}).get("nav", [])
        bm_dates = data.get("benchmark", {}).get("dates", [])
        if bm_navs and bm_dates:
            # 如果已有历史，保持一致
            pass
        else:
            hs300_start = hs300_now  # 首次
    
    benchmark_nav = round(hs300_now / hs300_start, 6) if hs300_start and hs300_now else None
    
    # 更新每位选手
    for pid, player in data["players"].items():
        positions, cash = player_pos_refs[pid]
        
        total_market_value = 0
        updated_count = 0
        
        for code, pos in positions.items():
            if code in prices:
                new_price = prices[code]
                pos["current_price"] = new_price
                
                shares = pos.get("shares", pos.get("volume", 0))
                cost = pos.get("cost_price", pos.get("avg_cost", 0))
                
                pos["market_value"] = round(new_price * shares, 2)
                if cost > 0:
                    pos["pnl"] = round((new_price - cost) * shares, 2)
                    pos["pnl_pct"] = round((new_price / cost - 1) * 100, 2)
                
                total_market_value += pos["market_value"]
                updated_count += 1
            else:
                # 保持原值
                shares = pos.get("shares", pos.get("volume", 0))
                mv = pos.get("market_value", pos.get("current_price", pos.get("avg_cost", 0)) * shares)
                total_market_value += mv
        
        total_value = cash + total_market_value
        
        # 更新portfolio
        if "portfolio" in player:
            player["portfolio"]["total_value"] = round(total_value, 2)
            player["portfolio"]["last_update"] = now_str
        
        # 计算盘中净值
        nav = round(total_value / INITIAL_CASH, 6)
        cash_pct = round(cash / total_value * 100, 2) if total_value > 0 else 100
        
        # 盘中刷新：只更新portfolio，不追加nav_history
        # nav_history 仅由 nav-update（收盘）写入，保证每天只有一条收盘NAV
        
        # 更新stats（仅更新实时收益率和持仓数）
        stats = player.get("stats", {})
        stats["total_return_pct"] = round((nav - 1) * 100, 2)
        stats["current_positions"] = len(positions)
        player["stats"] = stats
        
        print(f"  {player.get('emoji','?')} {player.get('name', pid)}: NAV={nav:.4f} ({stats['total_return_pct']:+.2f}%), "
              f"updated {updated_count}/{len(positions)} prices", file=sys.stderr)
    
    # 更新基准
    if benchmark_nav is not None:
        bm = data.get("benchmark", {})
        bm_dates = bm.get("dates", [])
        bm_navs = bm.get("nav", [])
        bm_time_label = datetime.now().strftime("%m-%d %H:%M")
        if bm_dates and bm_dates[-1] == bm_time_label:
            bm_navs[-1] = benchmark_nav
        else:
            bm_dates.append(bm_time_label)
            bm_navs.append(benchmark_nav)
        bm["dates"] = bm_dates
        bm["nav"] = bm_navs
        if hs300_start:
            bm["start_value"] = hs300_start
        data["benchmark"] = bm
        print(f"  📈 沪深300: {hs300_now:.2f}, 基准NAV={benchmark_nav:.4f}", file=sys.stderr)
    
    # 更新排行榜
    rankings = []
    for pid, player in data["players"].items():
        nav_list = player.get("nav_history", {}).get("nav", [])
        stats = player.get("stats", {})
        rankings.append({
            "player_id": pid,
            "name": player.get("name", pid),
            "emoji": player.get("emoji", ""),
            "total_return_pct": stats.get("total_return_pct", 0),
            "sharpe_ratio": stats.get("sharpe_ratio", 0),
            "max_drawdown_pct": stats.get("max_drawdown_pct", 0),
            "current_nav": nav_list[-1] if nav_list else 1.0,
            "total_trades": stats.get("total_trades", 0)
        })
    rankings.sort(key=lambda x: x["total_return_pct"], reverse=True)
    for i, r in enumerate(rankings):
        r["rank"] = i + 1
    data["leaderboard"] = {"rankings": rankings, "updated_at": now_str}
    
    # 更新meta时间
    data["meta"]["updated_at"] = now_str
    
    if args.dry_run:
        print(json.dumps(data, ensure_ascii=False, indent=2)[:2000])
        return
    
    # 保存
    with open(args.data, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] 数据已更新 {now_str}", file=sys.stderr)
    
    # Git push
    if not args.no_push:
        data_dir = os.path.dirname(os.path.abspath(args.data)) or '.'
        data_basename = os.path.basename(args.data)
        time_label = datetime.now().strftime("%H:%M")
        # macOS没有timeout，用gtimeout或直接跑
        timeout_cmd = "gtimeout" if os.path.exists("/opt/homebrew/bin/gtimeout") else "timeout"
        cmd = (f"cd {data_dir} && "
               f"git add {data_basename} && "
               f"git diff --cached --quiet || "
               f"(git commit -m 'intraday: nav refresh {time_label}' && "
               f"{timeout_cmd} 30 git push origin main 2>&1)")
        ret = os.system(cmd)
        if ret != 0:
            print("[WARN] git push failed, data saved locally", file=sys.stderr)
    
    # 输出摘要到stdout
    summary = {"time": now_str, "players": {}}
    for pid, player in data["players"].items():
        nav_list = player.get("nav_history", {}).get("nav", [])
        summary["players"][pid] = {
            "nav": nav_list[-1] if nav_list else 1.0,
            "return_pct": player.get("stats", {}).get("total_return_pct", 0)
        }
    if benchmark_nav:
        summary["benchmark_nav"] = benchmark_nav
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
