#!/usr/bin/env python3
"""
AI投资竞赛 — 模拟交易引擎
负责：下单/撮合/费用计算/净值更新/排行榜/前端JSON生成
"""

import json
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import copy

# ─── 常量 ───
DEFAULT_DATA_FILE = "paper-trading-data.json"
COMMISSION_RATE = 0.0003      # 佣金万三（双向）
STAMP_DUTY_RATE = 0.0005      # 印花税万五（仅卖出）
TRANSFER_FEE_RATE = 0.00001   # 过户费十万分之一
SLIPPAGE_RATE = 0.001          # 滑点千一
MIN_COMMISSION = 5.0           # 最低佣金

PLAYERS = {
    "quant": {
        "name": "因子猎人",
        "emoji": "🔢",
        "style": "量化因子选股 · 中证1000 · 周度调仓",
        "description": "每天读研报挖因子，用统计显著的因子排名选股，系统化交易"
    },
    "trader": {
        "name": "技术猎手",
        "emoji": "📊",
        "style": "技术面择时 · 单标的 · 日内决策",
        "description": "盯一只高波动高换手标的，MACD/KDJ/RSI/BOLL综合分析，择时进出"
    },
    "value": {
        "name": "巴菲特门徒",
        "emoji": "🏛️",
        "style": "价值投资 · 集中持仓 · 低换手",
        "description": "每天读新闻做基本面分析，寻找护城河深的优质公司，长期持有"
    }
}

# ─── 数据初始化 ───
def init_data(start_date: str, initial_cash: float = 10_000_000) -> dict:
    """初始化竞赛数据结构"""
    data = {
        "meta": {
            "name": "AI投资竞赛 · 三大流派对决",
            "start_date": start_date,
            "initial_cash": initial_cash,
            "benchmark": "000300",
            "benchmark_name": "沪深300",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "players": {},
        "benchmark": {
            "dates": [],
            "nav": []
        },
        "leaderboard": {
            "rankings": [],
            "updated_at": None
        }
    }
    
    for pid, pinfo in PLAYERS.items():
        data["players"][pid] = {
            "name": pinfo["name"],
            "emoji": pinfo["emoji"],
            "style": pinfo["style"],
            "description": pinfo["description"],
            "portfolio": {
                "cash": initial_cash,
                "positions": {},
                "total_value": initial_cash,
                "last_update": start_date
            },
            "nav_history": {
                "dates": [],
                "nav": [],
                "cash_pct": []
            },
            "trades": [],
            "decisions": [],
            "rebalances": [],
            "stats": {
                "total_return_pct": 0.0,
                "annualized_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "calmar_ratio": 0.0,
                "win_rate": 0.0,
                "profit_loss_ratio": 0.0,
                "total_trades": 0,
                "total_fees": 0.0,
                "current_positions": 0,
                "turnover_rate": 0.0
            }
        }
    
    return data


def load_data(filepath: str) -> dict:
    """加载竞赛数据"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_data(data: dict, filepath: str):
    """保存竞赛数据"""
    data["meta"]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] 数据已保存: {filepath}")


# ─── 费用计算 ───
def calc_fees(price: float, volume: int, direction: str) -> dict:
    """计算交易费用"""
    amount = price * volume
    
    # 佣金（双向）
    commission = max(amount * COMMISSION_RATE, MIN_COMMISSION)
    
    # 印花税（仅卖出）
    stamp_duty = amount * STAMP_DUTY_RATE if direction == "sell" else 0
    
    # 过户费
    transfer_fee = amount * TRANSFER_FEE_RATE
    
    # 滑点
    slippage_cost = amount * SLIPPAGE_RATE
    
    total = commission + stamp_duty + transfer_fee + slippage_cost
    
    return {
        "commission": round(commission, 2),
        "stamp_duty": round(stamp_duty, 2),
        "transfer_fee": round(transfer_fee, 2),
        "slippage_cost": round(slippage_cost, 2),
        "total": round(total, 2)
    }


# ─── 交易执行 ───
def execute_trade(data: dict, player_id: str, code: str, name: str, 
                  price: float, volume: int, direction: str, date: str,
                  reason: str = "") -> dict:
    """执行一笔模拟交易"""
    player = data["players"][player_id]
    portfolio = player["portfolio"]
    
    fees = calc_fees(price, volume, direction)
    amount = price * volume
    
    if direction == "buy":
        # 检查资金够不够
        total_cost = amount + fees["total"]
        if total_cost > portfolio["cash"]:
            return {"status": "error", "message": f"资金不足: 需要{total_cost:.2f}, 可用{portfolio['cash']:.2f}"}
        
        # 扣钱
        portfolio["cash"] -= total_cost
        
        # 更新持仓（如已有则加仓）
        if code in portfolio["positions"]:
            pos = portfolio["positions"][code]
            old_cost = pos["avg_cost"] * pos["volume"]
            new_cost = old_cost + amount
            pos["volume"] += volume
            pos["avg_cost"] = new_cost / pos["volume"]
        else:
            portfolio["positions"][code] = {
                "code": code,
                "name": name,
                "volume": volume,
                "avg_cost": price,
                "current_price": price,
                "market_value": amount,
                "pnl": 0,
                "pnl_pct": 0,
                "entry_date": date
            }
    
    elif direction == "sell":
        if code not in portfolio["positions"]:
            return {"status": "error", "message": f"无持仓: {code}"}
        
        pos = portfolio["positions"][code]
        if volume > pos["volume"]:
            return {"status": "error", "message": f"持仓不足: 持有{pos['volume']}, 卖出{volume}"}
        
        # 收钱
        portfolio["cash"] += amount - fees["total"]
        
        # 更新持仓
        pos["volume"] -= volume
        if pos["volume"] == 0:
            del portfolio["positions"][code]
        else:
            pos["market_value"] = pos["current_price"] * pos["volume"]
    
    # 记录交易
    trade_record = {
        "date": date,
        "code": code,
        "name": name,
        "direction": direction,
        "price": price,
        "volume": volume,
        "amount": round(amount, 2),
        "fees": fees,
        "reason": reason
    }
    player["trades"].append(trade_record)
    player["stats"]["total_trades"] += 1
    player["stats"]["total_fees"] += fees["total"]
    
    return {"status": "ok", "trade": trade_record}


def execute_order_list(data: dict, player_id: str, orders: list, date: str) -> list:
    """批量执行订单列表
    orders: [{"code": "000001", "name": "平安银行", "price": 10.5, "volume": 1000, "direction": "buy", "reason": "..."}]
    """
    results = []
    # 先执行卖出，再执行买入（释放资金）
    sells = [o for o in orders if o["direction"] == "sell"]
    buys = [o for o in orders if o["direction"] == "buy"]
    
    for order in sells + buys:
        result = execute_trade(
            data, player_id,
            code=order["code"],
            name=order["name"],
            price=order["price"],
            volume=order["volume"],
            direction=order["direction"],
            date=date,
            reason=order.get("reason", "")
        )
        results.append(result)
    
    return results


# ─── 净值更新 ───
def update_nav(data: dict, player_id: str, prices: dict, date: str):
    """
    用最新价格更新某选手的净值
    prices: {"000001": 10.5, "600519": 1800.0, ...}
    """
    player = data["players"][player_id]
    portfolio = player["portfolio"]
    
    # 更新每个持仓的当前价格和盈亏
    total_market_value = 0
    for code, pos in portfolio["positions"].items():
        if code in prices:
            pos["current_price"] = prices[code]
            pos["market_value"] = round(pos["current_price"] * pos["volume"], 2)
            pos["pnl"] = round((pos["current_price"] - pos["avg_cost"]) * pos["volume"], 2)
            pos["pnl_pct"] = round((pos["current_price"] / pos["avg_cost"] - 1) * 100, 2) if pos["avg_cost"] > 0 else 0
        total_market_value += pos.get("market_value", pos["avg_cost"] * pos["volume"])
    
    # 计算总资产和净值
    total_value = portfolio["cash"] + total_market_value
    portfolio["total_value"] = round(total_value, 2)
    portfolio["last_update"] = date
    
    initial_cash = data["meta"]["initial_cash"]
    nav = round(total_value / initial_cash, 6)
    cash_pct = round(portfolio["cash"] / total_value * 100, 2) if total_value > 0 else 100
    
    # 追加净值历史
    nav_history = player["nav_history"]
    nav_history["dates"].append(date)
    nav_history["nav"].append(nav)
    nav_history["cash_pct"].append(cash_pct)
    
    # 更新统计指标
    _update_stats(player, initial_cash)
    
    player["stats"]["current_positions"] = len(portfolio["positions"])


def update_benchmark(data: dict, benchmark_nav: float, date: str):
    """更新基准净值"""
    data["benchmark"]["dates"].append(date)
    data["benchmark"]["nav"].append(round(benchmark_nav, 6))


def update_all_navs(data: dict, all_prices: dict, benchmark_nav: float, date: str):
    """一次性更新所有选手净值和基准"""
    for pid in data["players"]:
        update_nav(data, pid, all_prices, date)
    update_benchmark(data, benchmark_nav, date)
    _update_leaderboard(data)


# ─── 统计指标 ───
def _update_stats(player: dict, initial_cash: float):
    """更新选手统计指标"""
    stats = player["stats"]
    nav_list = player["nav_history"]["nav"]
    
    if len(nav_list) == 0:
        return
    
    # 累计收益
    stats["total_return_pct"] = round((nav_list[-1] - 1) * 100, 2)
    
    # 年化收益
    n_days = len(nav_list)
    if n_days > 1:
        total_return = nav_list[-1] / nav_list[0] if nav_list[0] > 0 else 1
        years = n_days / 252
        if years > 0 and total_return > 0:
            stats["annualized_return_pct"] = round((total_return ** (1 / years) - 1) * 100, 2)
    
    # 最大回撤
    peak = nav_list[0]
    max_dd = 0
    for nav in nav_list:
        if nav > peak:
            peak = nav
        dd = (peak - nav) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    stats["max_drawdown_pct"] = round(max_dd * 100, 2)
    
    # Sharpe（简化版：假设无风险利率2%，年化）
    if len(nav_list) > 5:
        import numpy as np
        nav_arr = np.array(nav_list)
        returns = np.diff(nav_arr) / nav_arr[:-1]
        if returns.std() > 0:
            daily_rf = 0.02 / 252
            sharpe = (returns.mean() - daily_rf) / returns.std() * (252 ** 0.5)
            stats["sharpe_ratio"] = round(sharpe, 2)
    
    # Calmar
    if max_dd > 0 and n_days > 20:
        stats["calmar_ratio"] = round(stats.get("annualized_return_pct", 0) / (max_dd * 100), 2)
    
    # 胜率和盈亏比
    trades = player["trades"]
    if trades:
        # 只统计已平仓的（简化：按每笔卖出交易算）
        sell_trades = [t for t in trades if t["direction"] == "sell"]
        # 这里简化处理，后续可以精确匹配买卖对
        winning = sum(1 for t in sell_trades if t.get("pnl", 0) > 0)
        stats["win_rate"] = round(winning / len(sell_trades) * 100, 2) if sell_trades else 0


def _update_leaderboard(data: dict):
    """更新排行榜"""
    rankings = []
    for pid, player in data["players"].items():
        nav_list = player["nav_history"]["nav"]
        rankings.append({
            "player_id": pid,
            "name": player["name"],
            "emoji": player["emoji"],
            "total_return_pct": player["stats"]["total_return_pct"],
            "sharpe_ratio": player["stats"]["sharpe_ratio"],
            "max_drawdown_pct": player["stats"]["max_drawdown_pct"],
            "current_nav": nav_list[-1] if nav_list else 1.0,
            "total_trades": player["stats"]["total_trades"]
        })
    
    # 按累计收益排名
    rankings.sort(key=lambda x: x["total_return_pct"], reverse=True)
    for i, r in enumerate(rankings):
        r["rank"] = i + 1
    
    data["leaderboard"]["rankings"] = rankings
    data["leaderboard"]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ─── 决策日志 ───
def add_decision(data: dict, player_id: str, date: str, 
                 decision_type: str, summary: str, detail: str = "",
                 action: str = "hold"):
    """记录选手决策
    decision_type: "factor_scan" | "technical_analysis" | "fundamental_analysis" | "rebalance" | "trade"
    action: "buy" | "sell" | "hold" | "rebalance"
    """
    player = data["players"][player_id]
    decision = {
        "date": date,
        "time": datetime.now().strftime("%H:%M"),
        "type": decision_type,
        "action": action,
        "summary": summary,
        "detail": detail
    }
    player["decisions"].append(decision)
    
    # 只保留最近100条决策
    if len(player["decisions"]) > 100:
        player["decisions"] = player["decisions"][-100:]


# ─── 调仓记录 ───
def add_rebalance(data: dict, player_id: str, date: str,
                  sells: int, buys: int, total_fees: float, 
                  note: str = "", target_holdings: list = None):
    """记录调仓"""
    player = data["players"][player_id]
    portfolio = player["portfolio"]
    
    rebalance = {
        "date": date,
        "total_value_before": portfolio["total_value"],
        "sells": sells,
        "buys": buys,
        "total_fees": round(total_fees, 2),
        "note": note,
        "target_holdings": target_holdings or []
    }
    player["rebalances"].append(rebalance)


# ─── 查询接口 ───
def get_player_summary(data: dict, player_id: str) -> str:
    """获取选手摘要（给AI决策用）"""
    player = data["players"][player_id]
    portfolio = player["portfolio"]
    stats = player["stats"]
    
    lines = [
        f"## {player['emoji']} {player['name']}",
        f"总资产: ¥{portfolio['total_value']:,.2f}",
        f"可用资金: ¥{portfolio['cash']:,.2f}",
        f"累计收益: {stats['total_return_pct']}%",
        f"最大回撤: {stats['max_drawdown_pct']}%",
        f"持仓数: {len(portfolio['positions'])}只",
        ""
    ]
    
    if portfolio["positions"]:
        lines.append("### 当前持仓")
        for code, pos in portfolio["positions"].items():
            lines.append(f"- {pos['name']}({code}): {pos['volume']}股 @ ¥{pos['avg_cost']:.2f}, "
                        f"现价¥{pos['current_price']:.2f}, 盈亏{pos['pnl_pct']:+.2f}%")
    
    # 最近5条决策
    recent_decisions = player["decisions"][-5:]
    if recent_decisions:
        lines.append("\n### 最近决策")
        for d in recent_decisions:
            lines.append(f"- [{d['date']} {d.get('time','')}] {d['summary']}")
    
    return "\n".join(lines)


def get_competition_summary(data: dict) -> str:
    """获取竞赛总览"""
    lines = [
        "# 🏆 AI投资竞赛 · 三大流派对决",
        f"起始日期: {data['meta']['start_date']}",
        f"初始资金: ¥{data['meta']['initial_cash']:,.0f}",
        f"基准: {data['meta']['benchmark_name']}",
        ""
    ]
    
    rankings = data["leaderboard"].get("rankings", [])
    if rankings:
        lines.append("## 排行榜")
        medals = ["🥇", "🥈", "🥉"]
        for r in rankings:
            medal = medals[r["rank"]-1] if r["rank"] <= 3 else f"#{r['rank']}"
            lines.append(f"{medal} {r['emoji']} {r['name']}: {r['total_return_pct']:+.2f}% "
                        f"(Sharpe={r['sharpe_ratio']}, MDD={r['max_drawdown_pct']}%)")
    
    lines.append("")
    for pid in data["players"]:
        lines.append(get_player_summary(data, pid))
        lines.append("")
    
    return "\n".join(lines)


# ─── CLI ───
def main():
    parser = argparse.ArgumentParser(description="AI投资竞赛 · 模拟交易引擎")
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # init
    sp_init = subparsers.add_parser("init", help="初始化竞赛数据")
    sp_init.add_argument("--start-date", required=True, help="起始日期 YYYY-MM-DD")
    sp_init.add_argument("--cash", type=float, default=10_000_000, help="初始资金")
    sp_init.add_argument("--output", default=DEFAULT_DATA_FILE, help="输出文件路径")
    
    # trade
    sp_trade = subparsers.add_parser("trade", help="执行交易")
    sp_trade.add_argument("--player", required=True, choices=["quant","trader","value","dwj"])
    sp_trade.add_argument("--code", required=True)
    sp_trade.add_argument("--name", required=True)
    sp_trade.add_argument("--price", type=float, required=True)
    sp_trade.add_argument("--volume", type=int, required=True)
    sp_trade.add_argument("--direction", required=True, choices=["buy","sell"])
    sp_trade.add_argument("--date", required=True)
    sp_trade.add_argument("--reason", default="")
    sp_trade.add_argument("--data", default=DEFAULT_DATA_FILE)
    
    # nav
    sp_nav = subparsers.add_parser("nav", help="更新净值（从JSON价格文件）")
    sp_nav.add_argument("--prices-file", required=True, help="价格JSON: {code: price}")
    sp_nav.add_argument("--benchmark-nav", type=float, required=True)
    sp_nav.add_argument("--date", required=True)
    sp_nav.add_argument("--data", default=DEFAULT_DATA_FILE)
    
    # summary
    sp_summary = subparsers.add_parser("summary", help="输出竞赛摘要")
    sp_summary.add_argument("--player", choices=["quant","trader","value","dwj"], help="指定选手（留空=全部）")
    sp_summary.add_argument("--data", default=DEFAULT_DATA_FILE)
    
    # decision
    sp_dec = subparsers.add_parser("decision", help="记录决策")
    sp_dec.add_argument("--player", required=True, choices=["quant","trader","value","dwj"])
    sp_dec.add_argument("--date", required=True)
    sp_dec.add_argument("--type", required=True)
    sp_dec.add_argument("--summary", required=True)
    sp_dec.add_argument("--detail", default="")
    sp_dec.add_argument("--action", default="hold")
    sp_dec.add_argument("--data", default=DEFAULT_DATA_FILE)
    
    args = parser.parse_args()
    
    if args.command == "init":
        data = init_data(args.start_date, args.cash)
        save_data(data, args.output)
        print(f"[OK] 竞赛初始化完成: {args.start_date}, ¥{args.cash:,.0f}/人")
    
    elif args.command == "trade":
        data = load_data(args.data)
        result = execute_trade(data, args.player, args.code, args.name,
                              args.price, args.volume, args.direction, args.date, args.reason)
        if result["status"] == "ok":
            save_data(data, args.data)
            t = result["trade"]
            print(f"[OK] {args.player} {t['direction']} {t['name']}({t['code']}) "
                  f"{t['volume']}股 @ ¥{t['price']}, 费用¥{t['fees']['total']}")
        else:
            print(f"[ERROR] {result['message']}", file=sys.stderr)
            sys.exit(1)
    
    elif args.command == "nav":
        data = load_data(args.data)
        with open(args.prices_file, 'r') as f:
            prices = json.load(f)
        update_all_navs(data, prices, args.benchmark_nav, args.date)
        save_data(data, args.data)
        print(f"[OK] 净值已更新: {args.date}")
        for pid, player in data["players"].items():
            nav = player["nav_history"]["nav"][-1] if player["nav_history"]["nav"] else 1.0
            print(f"  {player['emoji']} {player['name']}: NAV={nav:.4f}, "
                  f"收益={player['stats']['total_return_pct']:+.2f}%")
    
    elif args.command == "summary":
        data = load_data(args.data)
        if args.player:
            print(get_player_summary(data, args.player))
        else:
            print(get_competition_summary(data))
    
    elif args.command == "decision":
        data = load_data(args.data)
        add_decision(data, args.player, args.date, args.type, args.summary, args.detail, args.action)
        save_data(data, args.data)
        print(f"[OK] 决策已记录: {args.player} @ {args.date}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
