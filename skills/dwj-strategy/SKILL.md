---
name: dwj-strategy
description: 趋势游侠策略 — 全A股板块轮动+趋势跟踪选股系统。BBI+MA60趋势判断，KDJ低位买入，单针下30补充信号。
---

# 趋势游侠策略 (DWJ Strategy)

板块轮动+趋势跟踪选股系统，用于AI投资竞赛"趋势游侠"选手。

## 竞赛角色

- 选手名：趋势游侠
- Emoji：🏄
- 风格：趋势跟踪 + 板块轮动 + 中长线波段
- 股票池：全A股（约5000只）
- 持仓数量：5-10只

## 策略规格

详见 `references/strategy_spec.md`

## 执行流程

### 每日晨扫（09:35工作日）

```bash
WORK=~/.openclaw/workspace/alpha-factor-lab
SCRIPTS=$WORK/skills/dwj-strategy/scripts

# 1. 获取全A实时行情（~16秒）
python3 $SCRIPTS/realtime_quotes.py --output /tmp/a_share_quotes.json

# 2. 更新K线缓存（追加当天数据，<1秒）
python3 $SCRIPTS/kline_cache.py update --quotes /tmp/a_share_quotes.json

# 3. 全A扫描（~15秒）
python3 $SCRIPTS/scanner.py --quotes /tmp/a_share_quotes.json --output /tmp/dwj_signals.json

# 4. 检查现有持仓止盈止损
python3 $SCRIPTS/position_manager.py check --quotes /tmp/a_share_quotes.json
```

### 决策逻辑

1. **先检查持仓**：有止损/止盈信号 → 优先执行卖出
2. **再看候选**：扫描结果中 score > 30 的标的 → 评估是否买入
3. **仓位控制**：
   - 最大10只持仓
   - 单只=可用资金/剩余可开仓位数
   - 无加仓，买入即该标的满配
4. **板块优先**：同板块出现3只以上信号 → 该板块优先

### 买入执行
- **日内量价确认**（9:33-9:37）：候选票在开盘后第4-8分钟，需同时满足"单分钟量比递增"+"价格上升"才确认买入，否则放弃
- 买入价：确认后在9:37附近价格买入
- 止损：买入价往下5个价位
- 在 paper-trading-data.json 中记录交易

### 止盈止损（尾盘扫描检查）
- 止损：买入价-5tick，破位全清
- 止盈1：累计+7%，卖1/2
- 止盈2：不破BBI持有，破BBI清仓
- 利润保护：浮盈全亏完 → 全清

### 数据写入

更新 paper-trading-data.json 的 dwj player：
- portfolio（cash/positions/total_value）
- trades（交易记录）
- decisions（决策日志）
- nav_history（净值曲线）

然后 git add + commit + push。

## K线缓存

### 首次初始化（一次性，约5分钟）
```bash
python3 $SCRIPTS/kline_cache.py init --days 80 --workers 10
```

### 每日更新（<1秒）
从实时行情追加当天K线，无需重新拉取。

## 工具脚本

| 脚本 | 功能 | 耗时 |
|------|------|------|
| realtime_quotes.py | 全A实时行情批量获取 | ~16s |
| kline_cache.py | K线缓存管理（init/update/get） | init 5min, update <1s |
| scanner.py | 全A扫描+信号排序 | ~15s |
| position_manager.py | 仓位管理+止盈止损+交易执行 | <1s |

## 数据源

- 实时行情：腾讯 qt.gtimg.cn（批量查询，稳定可靠）
- 日K线：腾讯 web.ifzq.gtimg.cn（前复权，80天缓存）
- 板块数据：TODO（暂用代码前缀粗分）
