#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仓位管理器 — 止盈止损检查 + 交易执行

用法:
  # 检查持仓止盈止损
  python3 position_manager.py check --quotes /tmp/a_share_quotes.json

  # 执行买入
  python3 position_manager.py buy --code sh601777 --price 11.39 --reason "J<13+上升趋势"

  # 执行卖出
  python3 position_manager.py sell --code sh601777 --portion 0.5 --reason "止盈7%"
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent.parent.parent / "paper-trading-data.json"
PLAYER_ID = "dwj"
MAX_POSITIONS = 10


def load_data():
    with open(DATA_FILE) as f:
        return json.load(f)


def save_data(data):
    tmp = str(DATA_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, str(DATA_FILE))


def get_player(data):
    return data["players"].get(PLAYER_ID, {})


def format_code(code):
    """sh601777 → 601777.SH"""
    if code.startswith("sh"):
        return f"{code[2:]}.SH"
    elif code.startswith("sz"):
        return f"{code[2:]}.SZ"
    return code


def reverse_code(formatted_code):
    """601777.SH → sh601777, or bare 601777/002040 → sh601777/sz002040"""
    if formatted_code.endswith(".SH"):
        return f"sh{formatted_code[:-3]}"
    elif formatted_code.endswith(".SZ"):
        return f"sz{formatted_code[:-3]}"
    # Handle bare numeric codes (e.g. "002040", "600072")
    if formatted_code.isdigit():
        if formatted_code.startswith("6"):
            return f"sh{formatted_code}"
        else:
            return f"sz{formatted_code}"
    return formatted_code


def check_positions(quotes_file):
    """检查现有持仓的止盈止损"""
    data = load_data()
    player = get_player(data)
    positions = player.get("portfolio", {}).get("positions", {})
    
    if not positions:
        print("No positions to check.", file=sys.stderr)
        return {"actions": []}
    
    with open(quotes_file) as f:
        quotes = json.load(f)
    
    actions = []
    
    for pos_code, pos in positions.items():
        # 转换代码格式查找行情
        qq_code = reverse_code(pos_code)
        q = quotes.get(qq_code)
        
        if not q:
            print(f"[WARN] No quote for {pos_code} ({qq_code})", file=sys.stderr)
            continue
        
        current_price = q["price"]
        cost_price = pos.get("cost_price", 0) or pos.get("avg_cost", 0)
        shares = pos.get("shares", 0) or pos.get("volume", 0)
        
        if cost_price <= 0 or shares <= 0:
            continue
        
        pnl_pct = (current_price / cost_price - 1) * 100
        
        # === 止损检查 ===
        # 规则1: 买入价往下5个价位
        stop_loss = round(cost_price - 5 * 0.01, 2)
        if current_price <= stop_loss:
            actions.append({
                "code": pos_code,
                "name": pos.get("name", ""),
                "action": "sell_all",
                "reason": f"止损：价格{current_price}≤止损线{stop_loss}（成本{cost_price}下5个价位）",
                "current_price": current_price,
                "pnl_pct": round(pnl_pct, 2),
            })
            continue
        
        # 规则2: 利润全亏完（曾有浮盈但现在亏损）
        # 用高点回撤近似：如果盈利曾>3%但现在<0%
        if pnl_pct < 0 and pos.get("max_pnl_pct", 0) >= 3:
            actions.append({
                "code": pos_code,
                "name": pos.get("name", ""),
                "action": "sell_all",
                "reason": f"利润全亏：曾盈{pos.get('max_pnl_pct',0):.1f}%，现亏{pnl_pct:.1f}%",
                "current_price": current_price,
                "pnl_pct": round(pnl_pct, 2),
            })
            continue
        
        # === 止盈检查 ===
        # 规则1: 累计盈利≥7%，卖出1/2
        if pnl_pct >= 7 and not pos.get("half_profit_taken", False):
            actions.append({
                "code": pos_code,
                "name": pos.get("name", ""),
                "action": "sell_half",
                "reason": f"止盈第一档：盈利{pnl_pct:.1f}%≥7%，卖出1/2",
                "current_price": current_price,
                "pnl_pct": round(pnl_pct, 2),
            })
            continue
        
        # 规则2: 破BBI清仓
        # 这里需要BBI值，从K线缓存计算
        # 简化处理：由scanner在扫描时提供BBI值
        
        # 更新max_pnl_pct
        if pnl_pct > pos.get("max_pnl_pct", 0):
            pos["max_pnl_pct"] = round(pnl_pct, 2)
    
    # 保存更新的max_pnl_pct
    save_data(data)
    
    print(f"Checked {len(positions)} positions, {len(actions)} actions needed", file=sys.stderr)
    for a in actions:
        print(f"  {a['action']}: {a['code']} {a['name']} ({a['reason']})", file=sys.stderr)
    
    return {"actions": actions}


def check_bbi_stop(quotes_file):
    """检查BBI止损（需要K线缓存）"""
    import numpy as np
    
    kline_cache_file = Path(__file__).parent.parent.parent.parent / "data" / "a_share_kline_cache.json"
    if not kline_cache_file.exists():
        return {"actions": []}
    
    with open(kline_cache_file) as f:
        kline_cache = json.load(f)
    
    data = load_data()
    player = get_player(data)
    positions = player.get("portfolio", {}).get("positions", {})
    
    with open(quotes_file) as f:
        quotes = json.load(f)
    
    actions = []
    
    for pos_code, pos in positions.items():
        qq_code = reverse_code(pos_code)
        if qq_code not in kline_cache:
            continue
        
        klines = kline_cache[qq_code]
        if len(klines) < 24:
            continue
        
        closes = [k[2] for k in klines]
        
        # 更新为实时价格
        q = quotes.get(qq_code)
        if q:
            closes[-1] = q["price"]
        
        # 计算BBI
        bbi = (sum(closes[-3:])/3 + sum(closes[-6:])/6 + 
               sum(closes[-12:])/12 + sum(closes[-24:])/24) / 4
        
        current = closes[-1]
        
        # 破BBI清仓
        if current < bbi and pos.get("half_profit_taken", False):
            # 已经止盈过一次的票，破BBI就清
            actions.append({
                "code": pos_code,
                "name": pos.get("name", ""),
                "action": "sell_all",
                "reason": f"破BBI清仓：价格{current:.2f} < BBI={bbi:.2f}",
                "current_price": current,
                "bbi": round(bbi, 2),
            })
        elif current < bbi:
            # 没止盈过的票，破BBI预警
            actions.append({
                "code": pos_code,
                "name": pos.get("name", ""),
                "action": "warn_bbi",
                "reason": f"BBI预警：价格{current:.2f} < BBI={bbi:.2f}，关注是否清仓",
                "current_price": current,
                "bbi": round(bbi, 2),
            })
    
    return {"actions": actions}


def execute_buy(data, code, name, price, shares, reason, stop_loss=None):
    """执行买入"""
    player = data["players"][PLAYER_ID]
    portfolio = player["portfolio"]
    
    amount = price * shares
    fee = round(amount * 0.001, 2)  # 0.1%手续费
    total_cost = amount + fee
    
    if total_cost > portfolio["cash"]:
        print(f"[ERROR] Not enough cash: need {total_cost:.0f}, have {portfolio['cash']:.0f}", file=sys.stderr)
        return False
    
    pos_code = format_code(code)
    
    # 扣减现金
    portfolio["cash"] -= total_cost
    
    # 添加持仓
    portfolio["positions"][pos_code] = {
        "name": name,
        "shares": shares,
        "cost_price": price,
        "current_price": price,
        "market_value": amount,
        "stop_loss": stop_loss or round(price - 5 * 0.01, 2),
        "buy_date": datetime.now().strftime("%Y-%m-%d"),
        "max_pnl_pct": 0,
        "half_profit_taken": False,
    }
    
    # 更新总市值
    total_mv = sum(p.get("market_value", 0) for p in portfolio["positions"].values())
    portfolio["total_value"] = portfolio["cash"] + total_mv
    
    # 记录交易
    player.setdefault("trades", []).append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M"),
        "code": pos_code,
        "name": name,
        "action": "buy",
        "price": price,
        "shares": shares,
        "amount": amount,
        "fee": fee,
        "reason": reason,
    })
    
    return True


def execute_sell(data, pos_code, portion, price, reason):
    """执行卖出 (portion: 0.5=卖一半, 1.0=全卖)"""
    player = data["players"][PLAYER_ID]
    portfolio = player["portfolio"]
    pos = portfolio["positions"].get(pos_code)
    
    if not pos:
        print(f"[ERROR] Position {pos_code} not found", file=sys.stderr)
        return False
    
    sell_shares = int(pos["shares"] * portion / 100) * 100  # 整手
    if sell_shares <= 0:
        sell_shares = pos["shares"]  # 不足100股全卖
    
    if portion >= 0.99:
        sell_shares = pos["shares"]  # 全卖
    
    amount = price * sell_shares
    fee = max(round(amount * 0.001, 2), 5)  # 至少5元
    
    # 增加现金
    portfolio["cash"] += amount - fee
    
    # 更新持仓
    remaining = pos["shares"] - sell_shares
    if remaining <= 0:
        del portfolio["positions"][pos_code]
    else:
        pos["shares"] = remaining
        pos["current_price"] = price
        pos["market_value"] = remaining * price
        if portion >= 0.45 and portion <= 0.55:
            pos["half_profit_taken"] = True
    
    # 更新总市值
    total_mv = sum(p.get("market_value", 0) for p in portfolio["positions"].values())
    portfolio["total_value"] = portfolio["cash"] + total_mv
    
    # 记录交易
    player.setdefault("trades", []).append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M"),
        "code": pos_code,
        "name": pos.get("name", ""),
        "action": "sell",
        "price": price,
        "shares": sell_shares,
        "amount": amount,
        "fee": fee,
        "reason": reason,
    })
    
    return True


def main():
    parser = argparse.ArgumentParser(description="仓位管理器")
    sub = parser.add_subparsers(dest="action")
    
    p_check = sub.add_parser("check", help="检查止盈止损")
    p_check.add_argument("--quotes", required=True)
    
    p_buy = sub.add_parser("buy", help="执行买入")
    p_buy.add_argument("--code", required=True)
    p_buy.add_argument("--name", default="")
    p_buy.add_argument("--price", type=float, required=True)
    p_buy.add_argument("--shares", type=int, required=True)
    p_buy.add_argument("--reason", default="")
    
    p_sell = sub.add_parser("sell", help="执行卖出")
    p_sell.add_argument("--code", required=True)
    p_sell.add_argument("--portion", type=float, default=1.0, help="卖出比例 0.5=一半 1.0=全部")
    p_sell.add_argument("--price", type=float, required=True)
    p_sell.add_argument("--reason", default="")
    
    args = parser.parse_args()
    
    if args.action == "check":
        result1 = check_positions(args.quotes)
        result2 = check_bbi_stop(args.quotes)
        all_actions = result1["actions"] + result2["actions"]
        print(json.dumps({"actions": all_actions}, ensure_ascii=False, indent=2))
    
    elif args.action == "buy":
        data = load_data()
        success = execute_buy(data, args.code, args.name, args.price, args.shares, args.reason)
        if success:
            save_data(data)
            print(f"BUY OK: {args.code} x{args.shares} @{args.price}")
        else:
            print("BUY FAILED", file=sys.stderr)
            sys.exit(1)
    
    elif args.action == "sell":
        data = load_data()
        success = execute_sell(data, args.code, args.portion, args.price, args.reason)
        if success:
            save_data(data)
            print(f"SELL OK: {args.code} x{args.portion:.0%} @{args.price}")
        else:
            print("SELL FAILED", file=sys.stderr)
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
