#!/usr/bin/env python3
"""
标准化调仓脚本 — 解决 cron AI 代理绕过 trading_engine 直接写 JSON 导致的数据不一致

用法：
  python3 scripts/rebalance.py --player quant --date 2026-03-10 \
      --targets '{"600519.SH":0.04, "000001.SZ":0.04}' \
      [--data paper-trading-data.json]

  python3 scripts/rebalance.py --player quant --date 2026-03-10 \
      --target-codes 600519.SH,000001.SZ,002714.SZ \
      [--equal-weight] [--topn 25]
      
  python3 scripts/rebalance.py --player quant --date 2026-03-10 \
      --target-file /tmp/target_stocks.json

功能：
  1. 读取当前持仓
  2. 对比目标持仓，计算买卖差异
  3. 通过 trading_engine.execute_trade() 执行所有交易（先卖后买）
  4. 自动获取实时价格（腾讯行情 API）
  5. 输出调仓明细

设计原则：
  - 所有交易必须通过 trading_engine.execute_trade()
  - 不直接修改 paper-trading-data.json 的 positions/cash/trades
  - 保证 total_value / nav_history / positions 一致性
"""

import json
import os
import sys
import argparse
import urllib.request
from pathlib import Path

# Add trading_engine to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
ENGINE_DIR = REPO_DIR / "skills" / "paper-trading" / "scripts"
sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from trading_engine import load_data, save_data, execute_trade, execute_order_list

DEFAULT_DATA = str(REPO_DIR / "paper-trading-data.json")


def fetch_prices_tencent(codes: list) -> dict:
    """从腾讯行情API批量获取实时/收盘价格"""
    if not codes:
        return {}
    
    def to_tencent_code(code: str) -> str:
        # Handles: 'sh688710.SH'->'sh688710', '688710.SH'->'sh688710', '300881.SZ'->'sz300881'
        parts = code.split('.')
        if len(parts) == 2:
            num, ex = parts
            # Strip existing sh/sz prefix if present (e.g. 'sh688710' -> '688710')
            num_clean = num[2:] if (num.startswith('sh') or num.startswith('sz')) else num
            prefix = 'sh' if ex in ('SH', 'SS') else 'sz'
            return f"{prefix}{num_clean}"
        return code
    
    tc_codes = [to_tencent_code(c) for c in codes]
    
    # 每次最多80只（URL长度限制）
    prices = {}
    for i in range(0, len(tc_codes), 80):
        batch = tc_codes[i:i+80]
        batch_original = codes[i:i+80]
        
        url = f"https://qt.gtimg.cn/q={','.join(batch)}"
        headers = {"Referer": "https://finance.qq.com"}
        req = urllib.request.Request(url, headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=15)
            text = resp.read().decode('gbk')
        except Exception as e:
            print(f"  ⚠️ 腾讯行情API失败: {e}")
            continue
        
        for line in text.strip().split(';'):
            line = line.strip()
            if not line or '~' not in line:
                continue
            parts = line.split('~')
            if len(parts) > 33:
                name = parts[1]
                code_raw = parts[2]
                price = float(parts[3]) if parts[3] else 0
                
                # Map back to original code format
                # code_raw from Tencent is bare numeric code (e.g. 688710)
                # orig may have .SH/.SZ suffix, so strip prefix too
                for orig in batch_original:
                    bare = orig.split('.')[0].lstrip('sh').lstrip('sz')
                    if bare == code_raw:
                        if price > 0:
                            prices[orig] = {"price": price, "name": name}
                        break
    
    return prices


def fetch_prices_fintool(codes: list) -> dict:
    """从fintool API获取价格（备选）"""
    try:
        from fintool_client import call_tool
        prices = {}
        for code in codes:
            result = call_tool("market_quote", "get_snapshot", {"keyword": code.split('.')[0]})
            if result and isinstance(result, list) and len(result) > 0:
                item = result[0]
                price = float(item.get("latest_price", 0) or item.get("current", 0) or 0)
                name = item.get("name", "")
                if price > 0:
                    prices[code] = {"price": price, "name": name}
        return prices
    except Exception as e:
        print(f"  ⚠️ fintool失败: {e}")
        return {}


def rebalance(data: dict, player_id: str, target_codes: list, date: str,
              equal_weight: bool = True, target_weights: dict = None,
              reason: str = "rebalance") -> dict:
    """
    执行调仓：从当前持仓变换到目标持仓
    
    Args:
        data: paper-trading-data.json 的完整数据
        player_id: 选手ID (quant/trader/value/dwj/sentiment/hotspot)
        target_codes: 目标持仓代码列表 (e.g. ["600519.SH", "000001.SZ"])
        date: 调仓日期 YYYY-MM-DD
        equal_weight: 是否等权配置
        target_weights: 自定义权重 {"600519.SH": 0.1, ...}，覆盖 equal_weight
        reason: 调仓原因
    
    Returns:
        {"status": "ok/error", "sells": [...], "buys": [...], "summary": "..."}
    """
    player = data["players"][player_id]
    portfolio = player["portfolio"]
    initial_cash = data["meta"]["initial_cash"]
    
    current_positions = portfolio.get("positions", {})
    current_codes = set(current_positions.keys())
    target_set = set(target_codes)
    
    # 计算差异
    to_sell = current_codes - target_set
    to_buy = target_set - current_codes
    to_keep = current_codes & target_set
    
    print(f"\n{'='*60}")
    print(f"📊 {player['name']} 调仓计划 ({date})")
    print(f"{'='*60}")
    print(f"  当前持仓: {len(current_codes)} 只")
    print(f"  目标持仓: {len(target_codes)} 只")
    print(f"  卖出: {len(to_sell)} 只")
    print(f"  买入: {len(to_buy)} 只")
    print(f"  保留: {len(to_keep)} 只")
    
    # 获取所有需要的价格
    all_codes = list(to_sell | to_buy | to_keep)
    print(f"\n  获取 {len(all_codes)} 只股票实时价格...")
    prices = fetch_prices_tencent(all_codes)
    
    if len(prices) < len(all_codes) * 0.8:
        print(f"  ⚠️ 只获取到 {len(prices)}/{len(all_codes)} 个价格，尝试fintool补充...")
        missing = [c for c in all_codes if c not in prices]
        fintool_prices = fetch_prices_fintool(missing[:10])  # 限制数量
        prices.update(fintool_prices)
    
    print(f"  ✅ 获取到 {len(prices)} 个价格")
    
    # ═══ 第一步：卖出 ═══
    sell_results = []
    sell_proceeds = 0
    print(f"\n--- 卖出 {len(to_sell)} 只 ---")
    for code in sorted(to_sell):
        pos = current_positions[code]
        vol = pos.get("volume", 0)
        if vol <= 0:
            print(f"  ⚠️ {code} volume=0，跳过")
            continue
        
        price_info = prices.get(code, {})
        sell_price = price_info.get("price", pos.get("current_price", pos.get("avg_cost", 0)))
        name = price_info.get("name", pos.get("name", ""))
        
        if sell_price <= 0:
            print(f"  ⚠️ {code} 价格为0，跳过")
            continue
        
        result = execute_trade(data, player_id, code, name, sell_price, vol, "sell", date, reason)
        sell_results.append(result)
        if result.get("status") != "error":
            sell_proceeds += sell_price * vol
            print(f"  ✅ 卖出 {code} {name} {vol}股 @ {sell_price:.2f}")
        else:
            print(f"  ❌ {code}: {result.get('message','')}")
    
    # ═══ 第二步：计算买入金额 ═══
    available_cash = portfolio["cash"]
    total_value = portfolio["total_value"]
    
    print(f"\n  卖出后现金: {available_cash:,.2f}")
    print(f"  卖出后总资产: {total_value:,.2f}")
    
    # 计算目标配置
    if target_weights:
        weight_map = target_weights
    elif equal_weight:
        n = len(target_codes)
        w = 1.0 / n if n > 0 else 0
        weight_map = {c: w for c in target_codes}
    else:
        # Default: equal weight
        n = len(target_codes)
        w = 1.0 / n if n > 0 else 0
        weight_map = {c: w for c in target_codes}
    
    # ═══ 第2.5步：保留股仓位调整 ═══
    keep_results = []
    if to_keep:
        print(f"\n--- 保留股调整 {len(to_keep)} 只 ---")
        for code in sorted(to_keep):
            pos = current_positions[code]
            cur_vol = pos.get("volume", 0)
            cur_price_pos = pos.get("current_price", pos.get("avg_cost", 0))
            
            price_info = prices.get(code, {})
            cur_price = price_info.get("price", cur_price_pos)
            name = price_info.get("name", pos.get("name", ""))
            
            if cur_price <= 0:
                print(f"  ⚠️ {code} 价格为0，保持不动")
                continue
            
            target_value = total_value * weight_map.get(code, 0)
            target_vol = int(target_value / cur_price / 100) * 100
            
            diff = target_vol - cur_vol
            if abs(diff) < 100:
                print(f"  ➡️  {code} {name} 保持 {cur_vol}股（目标{target_vol}，差异<100股）")
                continue
            
            if diff < 0:
                # 减仓
                sell_vol = abs(diff)
                result = execute_trade(data, player_id, code, name, cur_price, sell_vol, "sell", date, reason + " (trim)")
                keep_results.append(result)
                if result.get("status") != "error":
                    print(f"  📉 减仓 {code} {name} {sell_vol}股 @ {cur_price:.2f} ({cur_vol}→{cur_vol-sell_vol})")
                else:
                    print(f"  ❌ 减仓 {code}: {result.get('message','')}")
            else:
                # 加仓 — 在买入阶段统一处理资金分配，这里先标记
                print(f"  📈 {code} {name} 需加仓 {diff}股（{cur_vol}→{target_vol}），随买入阶段执行")
                to_buy.add(code)  # 加入买入列表，买入时会根据target_weight计算
    
    # ═══ 第三步：买入（含保留股加仓）═══
    buy_results = []
    print(f"\n--- 买入 {len(to_buy)} 只 ---")
    
    for code in sorted(to_buy):
        price_info = prices.get(code, {})
        buy_price = price_info.get("price", 0)
        name = price_info.get("name", "")
        
        if buy_price <= 0:
            print(f"  ⚠️ {code} 价格为0，跳过")
            continue
        
        target_value = total_value * weight_map.get(code, 0)
        
        # 计算目标股数（100股整数倍）
        target_shares = int(target_value / buy_price / 100) * 100
        
        # 如果是保留股加仓，只买差额部分
        existing_vol = 0
        if code in current_positions:
            existing_vol = current_positions[code].get("volume", 0)
        
        shares = target_shares - existing_vol
        if shares <= 0:
            if existing_vol > 0:
                print(f"  ➡️  {code} {name} 已持有{existing_vol}股≥目标{target_shares}股，无需加仓")
            else:
                print(f"  ⚠️ {code} 计算股数为0（目标金额{target_value:,.0f}，价格{buy_price}），跳过")
            continue
        
        # 取整到100股
        shares = int(shares / 100) * 100
        if shares <= 0:
            print(f"  ➡️  {code} {name} 差额不足100股，保持不变")
            continue
        
        # 检查剩余资金
        cost = buy_price * shares * 1.002  # 含费用估算
        if cost > portfolio["cash"]:
            # 减少股数
            shares = int(portfolio["cash"] / buy_price / 100 / 1.002) * 100
            if shares <= 0:
                print(f"  ⚠️ {code} 资金不足，跳过（需{cost:,.0f}，剩{portfolio['cash']:,.0f}）")
                continue
        
        label = "加仓" if existing_vol > 0 else "买入"
        result = execute_trade(data, player_id, code, name, buy_price, shares, "buy", date, reason)
        buy_results.append(result)
        if result.get("status") != "error":
            print(f"  ✅ {label} {code} {name} {shares}股 @ {buy_price:.2f}" +
                  (f" ({existing_vol}→{existing_vol+shares})" if existing_vol > 0 else ""))
        else:
            print(f"  ❌ {code}: {result.get('message','')}")
    
    # ═══ 最终状态 ═══
    final_positions = len(portfolio["positions"])
    final_cash = portfolio["cash"]
    final_total = portfolio["total_value"]
    nav = round(final_total / initial_cash, 6)
    
    print(f"\n{'='*60}")
    print(f"📊 调仓完成")
    print(f"  持仓: {final_positions} 只")
    print(f"  现金: {final_cash:,.2f} ({final_cash/final_total*100:.1f}%)")
    print(f"  总资产: {final_total:,.2f}")
    print(f"  NAV: {nav:.6f}")
    print(f"{'='*60}")
    
    return {
        "status": "ok",
        "sells": len(sell_results),
        "buys": len(buy_results),
        "final_positions": final_positions,
        "final_cash": final_cash,
        "final_total": final_total,
        "nav": nav,
        "summary": f"卖出{len(sell_results)}只,买入{len(buy_results)}只,最终{final_positions}只持仓,NAV={nav:.4f}"
    }


def main():
    parser = argparse.ArgumentParser(description="标准化调仓脚本")
    parser.add_argument("--player", required=True, help="选手ID (quant/trader/value/dwj/sentiment/hotspot)")
    parser.add_argument("--date", required=True, help="调仓日期 YYYY-MM-DD")
    parser.add_argument("--data", default=DEFAULT_DATA, help="数据文件路径")
    parser.add_argument("--reason", default="rebalance", help="调仓原因")
    
    # 目标持仓输入方式（三选一）
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target-codes", help="逗号分隔的目标股票代码 (e.g. 600519.SH,000001.SZ)")
    group.add_argument("--target-file", help="目标股票JSON文件路径 (list of codes 或 {code: weight})")
    group.add_argument("--targets", help="JSON格式目标 (e.g. '{\"600519.SH\":0.04}')")
    
    parser.add_argument("--equal-weight", action="store_true", default=True, help="等权配置（默认）")
    parser.add_argument("--topn", type=int, help="只取前N只")
    parser.add_argument("--dry-run", action="store_true", help="只计算不执行")
    parser.add_argument("--save", action="store_true", default=True, help="保存结果到数据文件")
    
    args = parser.parse_args()
    
    # 加载数据
    data = load_data(args.data)
    if args.player not in data["players"]:
        print(f"❌ 未知选手: {args.player}")
        print(f"   可选: {', '.join(data['players'].keys())}")
        sys.exit(1)
    
    # 解析目标持仓
    target_weights = None
    if args.target_codes:
        target_codes = [c.strip() for c in args.target_codes.split(',') if c.strip()]
    elif args.target_file:
        with open(args.target_file) as f:
            target_data = json.load(f)
        if isinstance(target_data, list):
            target_codes = target_data
        elif isinstance(target_data, dict):
            target_codes = list(target_data.keys())
            target_weights = target_data
        else:
            print(f"❌ 目标文件格式不支持: {type(target_data)}")
            sys.exit(1)
    elif args.targets:
        target_data = json.loads(args.targets)
        if isinstance(target_data, list):
            target_codes = target_data
        elif isinstance(target_data, dict):
            target_codes = list(target_data.keys())
            target_weights = target_data
        else:
            print(f"❌ --targets 格式不支持")
            sys.exit(1)
    
    # 限制数量
    if args.topn and len(target_codes) > args.topn:
        target_codes = target_codes[:args.topn]
        if target_weights:
            target_weights = {k: v for k, v in target_weights.items() if k in target_codes}
    
    # 确保代码格式正确（带后缀）
    validated = []
    for code in target_codes:
        if '.' not in code:
            # 尝试自动补后缀
            if code.startswith('6'):
                code = f"{code}.SH"
            else:
                code = f"{code}.SZ"
        validated.append(code)
    target_codes = validated
    
    print(f"目标持仓: {len(target_codes)} 只")
    
    if args.dry_run:
        player = data["players"][args.player]
        current = set(player["portfolio"]["positions"].keys())
        target = set(target_codes)
        print(f"[DRY RUN] 卖出 {len(current - target)} 只, 买入 {len(target - current)} 只, 保留 {len(current & target)} 只")
        sys.exit(0)
    
    # 执行调仓
    result = rebalance(
        data, args.player, target_codes, args.date,
        equal_weight=args.equal_weight,
        target_weights=target_weights,
        reason=args.reason
    )
    
    if result["status"] == "ok" and args.save:
        save_data(data, args.data)
        print(f"\n💾 数据已保存到 {args.data}")
    
    # 输出JSON结果（方便cron解析）
    print(f"\n📋 RESULT_JSON: {json.dumps(result, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
