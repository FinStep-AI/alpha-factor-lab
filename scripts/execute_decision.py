#!/usr/bin/env python3
"""
确定性交易执行器 — cron AI 只输出决策文件，本脚本负责执行

设计原则：
  1. AI agent 只负责分析，输出决策到 JSON 文件
  2. 本脚本读取决策文件，通过 trading_engine 执行交易
  3. 所有交易经过 trading_engine.execute_trade()，保证数据一致性
  4. 决策记录自动写入 player.decisions
  5. 执行完自动 git commit + push

用法：
  # 执行单笔交易
  python3 scripts/execute_decision.py --decision /tmp/decision.json

  # 执行调仓（整体换仓）
  python3 scripts/execute_decision.py --decision /tmp/decision.json --rebalance

  # 只记录日志不交易
  python3 scripts/execute_decision.py --decision /tmp/decision.json --log-only

决策文件格式：
{
  "player": "trader",           # 必填：选手ID
  "date": "2026-03-11",         # 必填：日期
  "action": "buy|sell|hold|rebalance",  # 必填：动作
  "trades": [                    # 交易列表（action=buy/sell时）
    {"code": "600519.SH", "name": "贵州茅台", "volume": 100, "reason": "KDJ金叉"}
  ],
  "target_codes": ["600519.SH", "000001.SZ"],  # 目标持仓（action=rebalance时）
  "topn": 25,                    # 目标持仓数量（rebalance时）
  "summary": "早盘分析摘要",      # 决策摘要
  "detail": "详细分析过程",       # 详细说明
  "recommendation": "BUY 600519.SH"  # 建议
}
"""

import json
import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
ENGINE_DIR = REPO_DIR / "skills" / "paper-trading" / "scripts"
sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from trading_engine import load_data, save_data, execute_trade
from rebalance import fetch_prices_tencent

DEFAULT_DATA = str(REPO_DIR / "paper-trading-data.json")


def validate_decision(decision: dict) -> list:
    """验证决策文件格式，返回错误列表"""
    errors = []
    if "player" not in decision:
        errors.append("缺少 player 字段")
    if "date" not in decision:
        errors.append("缺少 date 字段")
    if "action" not in decision:
        errors.append("缺少 action 字段")
    
    action = decision.get("action", "")
    if action in ("buy", "sell"):
        trades = decision.get("trades", [])
        if not trades:
            errors.append(f"action={action} 但 trades 为空")
        for i, t in enumerate(trades):
            if "code" not in t:
                errors.append(f"trades[{i}] 缺少 code")
            if "volume" not in t:
                errors.append(f"trades[{i}] 缺少 volume")
            if t.get("volume", 0) <= 0:
                errors.append(f"trades[{i}] volume 必须 > 0")
    elif action == "rebalance":
        if not decision.get("target_codes"):
            errors.append("action=rebalance 但 target_codes 为空")
    elif action == "hold":
        pass  # hold 不需要额外字段
    else:
        errors.append(f"未知 action: {action}")
    
    # 检查字段类型
    for field in ("summary", "detail", "recommendation"):
        val = decision.get(field)
        if val is not None and not isinstance(val, str):
            errors.append(f"{field} 必须是 string 类型，当前是 {type(val).__name__}")
    
    return errors


def execute_trades(data: dict, decision: dict) -> dict:
    """执行买卖交易"""
    player_id = decision["player"]
    date = decision["date"]
    trades = decision.get("trades", [])
    action = decision["action"]
    
    results = {"success": [], "failed": [], "skipped": []}
    
    # 收集所有需要的代码 — 标准化为 .SH/.SZ 格式供行情API使用
    def _normalize_code(c: str) -> str:
        """纯数字代码自动加后缀: 6开头→.SH, 其他→.SZ"""
        if '.' not in c:
            return f"{c}.SH" if c.startswith('6') else f"{c}.SZ"
        return c

    def _bare_code(c: str) -> str:
        """去掉 .SH/.SZ 后缀，返回纯数字代码"""
        return c.split('.')[0]

    codes_for_api = [_normalize_code(t["code"]) for t in trades]
    prices_raw = fetch_prices_tencent(codes_for_api)
    # 建立 bare_code → price_info 映射，兼容两种格式查找
    prices = {}
    for k, v in prices_raw.items():
        prices[k] = v
        prices[_bare_code(k)] = v
    
    for trade in trades:
        code = trade["code"]
        name = trade.get("name", "")
        volume = trade["volume"]
        reason = trade.get("reason", decision.get("summary", ""))
        
        price_info = prices.get(code, prices.get(_normalize_code(code), {}))
        price = price_info.get("price", 0)
        if not name and price_info.get("name"):
            name = price_info["name"]
        
        if price <= 0:
            results["failed"].append({"code": code, "error": f"无法获取价格"})
            print(f"  ❌ {code} 无法获取价格，跳过")
            continue
        
        # volume 取整到100的倍数
        volume = (volume // 100) * 100
        if volume <= 0:
            results["skipped"].append({"code": code, "reason": "volume < 100"})
            print(f"  ⏭️  {code} 数量不足100股，跳过")
            continue
        
        # 使用带后缀的 code（如 600141.SH）匹配 portfolio positions key
        trade_code = _normalize_code(code)
        result = execute_trade(data, player_id, trade_code, name, price, volume, action, date, reason)
        
        if result.get("status") == "ok":
            results["success"].append({
                "code": code, "name": name, "volume": volume,
                "price": price, "direction": action
            })
            print(f"  ✅ {action} {code} {name} {volume}股 @ {price:.2f}")
        else:
            results["failed"].append({"code": code, "error": result.get("message", "未知错误")})
            print(f"  ❌ {code}: {result.get('message', '未知错误')}")
    
    return results


def execute_rebalance(data: dict, decision: dict) -> dict:
    """执行调仓（委托给 rebalance.py）"""
    player_id = decision["player"]
    date = decision["date"]
    target_codes = decision["target_codes"]
    topn = decision.get("topn", 25)
    reason = decision.get("summary", "rebalance")
    
    # 调用 rebalance.py 的核心逻辑
    from rebalance import rebalance as do_rebalance
    
    result = do_rebalance(
        data=data,
        player_id=player_id,
        target_codes=target_codes[:topn],
        date=date,
        equal_weight=True,
        reason=reason
    )
    
    return result


def record_decision(data: dict, decision: dict, exec_results: dict = None):
    """记录决策日志到 player.decisions"""
    player_id = decision["player"]
    player = data["players"][player_id]
    
    if "decisions" not in player:
        player["decisions"] = []
    
    record = {
        "date": decision["date"],
        "action": decision["action"],
        "summary": str(decision.get("summary", "")),
        "detail": str(decision.get("detail", "")),
        "recommendation": str(decision.get("recommendation", "")),
    }
    
    if exec_results:
        # 只存摘要，不存完整结果
        success_count = len(exec_results.get("success", []))
        failed_count = len(exec_results.get("failed", []))
        record["exec_summary"] = f"成功{success_count}笔，失败{failed_count}笔"
    
    player["decisions"].append(record)
    
    # 保留最近30条
    if len(player["decisions"]) > 30:
        player["decisions"] = player["decisions"][-30:]


def git_commit_push(date: str, player_id: str, action: str):
    """git commit and push"""
    try:
        os.chdir(str(REPO_DIR))
        subprocess.run(["git", "add", "paper-trading-data.json"], check=True, capture_output=True)
        msg = f"{player_id}: {action} {date}"
        subprocess.run(["git", "commit", "-m", msg], check=True, capture_output=True)
        subprocess.run(["git", "push", "origin", "main"], check=True, capture_output=True, timeout=30)
        print(f"📤 git push 成功: {msg}")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  git操作失败: {e.stderr.decode() if e.stderr else str(e)}")
    except subprocess.TimeoutExpired:
        print("⚠️  git push 超时")


def main():
    parser = argparse.ArgumentParser(description="确定性交易执行器")
    parser.add_argument("--decision", required=True, help="决策JSON文件路径")
    parser.add_argument("--data", default=DEFAULT_DATA, help="数据文件路径")
    parser.add_argument("--rebalance", action="store_true", help="执行调仓模式")
    parser.add_argument("--log-only", action="store_true", help="只记录日志不交易")
    parser.add_argument("--no-push", action="store_true", help="不执行git push")
    parser.add_argument("--dry-run", action="store_true", help="只显示计划不执行")
    args = parser.parse_args()
    
    # 读取决策文件
    with open(args.decision) as f:
        decision = json.load(f)
    
    # 验证
    errors = validate_decision(decision)
    if errors:
        print("❌ 决策文件验证失败：")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    
    player_id = decision["player"]
    action = decision["action"]
    date = decision["date"]
    
    print(f"📋 决策: {player_id} / {action} / {date}")
    print(f"   摘要: {decision.get('summary', 'N/A')}")
    
    # 加载数据
    data = load_data(args.data)
    
    if player_id not in data["players"]:
        print(f"❌ 未知选手: {player_id}")
        sys.exit(1)
    
    exec_results = None
    
    if args.log_only or action == "hold":
        print("📝 仅记录决策日志")
    elif args.dry_run:
        print("🔍 Dry run 模式，不执行交易")
        if action in ("buy", "sell"):
            for t in decision.get("trades", []):
                print(f"  [DRY] {action} {t['code']} {t.get('name','')} {t['volume']}股")
        elif action == "rebalance":
            print(f"  [DRY] 调仓目标: {decision.get('target_codes', [])}")
    else:
        if action in ("buy", "sell"):
            print(f"\n🔄 执行 {action} 交易...")
            exec_results = execute_trades(data, decision)
        elif action == "rebalance":
            print(f"\n🔄 执行调仓...")
            exec_results = execute_rebalance(data, decision)
    
    # 记录决策
    record_decision(data, decision, exec_results)
    
    # 保存
    save_data(data, args.data)
    print(f"\n💾 数据已保存")
    
    # git push
    if not args.no_push:
        git_commit_push(date, player_id, action)
    
    print("✅ 完成")


if __name__ == "__main__":
    main()
