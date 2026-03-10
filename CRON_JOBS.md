# 定时任务总览

> 自动更新时间：2026-03-10 | 全部按工作日（周一~周五）运行（除 daily-log-push 和 factor-mining-weekend） | 时区：Asia/Shanghai | 共 23 个任务

## ⚠️ 数据一致性规则

**所有 cron 任务必须遵守以下铁律：**
1. **交易执行** → 必须通过 `scripts/rebalance.py` 或 `trading_engine.execute_trade()`，**绝对不要直接修改 JSON**
2. **净值更新** → 必须通过 `scripts/nav_update_all.py`，不要手动写 nav_history
3. **调仓** → 必须通过 `scripts/rebalance.py --player <id> --date <date> --target-codes <codes>`
4. `trading_engine.save_data()` 已内置保存前自检（幽灵持仓清除 + total_value 强制重算）

## 一、盘中选手任务（10个）

| 时间 | 任务名 | 选手 | 内容 | 超时 |
|------|--------|------|------|------|
| 08:30 | `value-news-scan` | 巴菲特门徒 🏛️ | 新闻扫描 + 交易建议 | 600s |
| 09:00 | `quant-factor-scan` | 因子猎人 🔢 | 研报扫描 + 因子回测；**周一额外：LightGBM调仓（必须用 rebalance.py）** | 2400s |
| 09:15 | `hotspot-morning-scan` | 热点猎手 📰 | 热点新闻扫描 + 建仓 | 900s |
| 09:20 | `trader-morning` | 技术猎手 📊 | 早盘技术分析 + 仓位决策 | 900s |
| 09:25 | `sentiment-morning-scan` | 情绪战神 ⚡ | 情绪温度计 + 龙头识别 + 仓位决策 | 600s |
| 09:35 | `dwj-morning-scan` | 趋势游侠 🏄 | 全A扫描 + 止盈止损 + 买入 | 600s |
| 13:30 | `trader-afternoon` | 技术猎手 📊 | 午盘验证 + 止损止盈检查 | 900s |
| 14:50 | `dwj-afternoon-scan` | 趋势游侠 🏄 | 尾盘扫描 + 极强信号追入 | 900s |
| 14:50 | `sentiment-afternoon-scan` | 情绪战神 ⚡ | 封板确认 + 龙头炸板检查 | 900s |
| 14:50 | `hotspot-afternoon-scan` | 热点猎手 📰 | 热点追踪 + 退出判断 | 900s |

## 二、收盘流程（6个，串行依赖）

| 时间 | 任务名 | 内容 | 超时 |
|------|--------|------|------|
| 15:05 | `sentiment-close-update` | 情绪战神收盘更新 + 交易 | 900s |
| 15:10 | `dwj-close-update` | 趋势游侠收盘行情 + K线缓存 + 交易 | 900s |
| 15:10 | `hotspot-close-update` | 热点猎手收盘净值确认 | 900s |
| 15:15 | `nav-update` | **标准化脚本** `nav_update_all.py` — 全选手净值 + 基准 + 排行榜 + 一致性自检 | 900s |
| 15:30 | `frontend-deploy` | git push → GitHub Pages 自动部署 | 900s |
| 15:45 | `daily-review` | 收盘战报飞书卡片 → 竞赛群 | 900s |

## 三、因子挖掘（4个）

| 时间 | 任务名 | 内容 | 超时 |
|------|--------|------|------|
| 12:00 工作日 | `factor-mining-noon` | 午间因子挖掘 | 2400s |
| 15:30 工作日 | `factor-mining-afternoon` | 盘后因子挖掘 | 2400s |
| 21:00 工作日 | `daily-factor-mining` | 晚间因子挖掘 | 1800s |
| 10:00/15:00/20:00 周末 | `factor-mining-weekend` | 周末论文复现 + 因子挖掘 | 3600s |

## 四、盘中辅助（2个）

| 时间 | 任务名 | 内容 | 超时 |
|------|--------|------|------|
| 每30分钟(09:00-14:30) | `intraday-nav-refresh` | **标准化脚本** `nav_update_all.py --intraday` 盘中净值刷新 | 600s |
| 15:30 工作日 | `kline-data-refresh` | K线数据更新 | 600s |

## 五、日常维护（1个）

| 时间 | 任务名 | 内容 | 超时 |
|------|--------|------|------|
| 23:59 每天 | `daily-log-push` | logs/ 目录 git push | 600s |

## 任务依赖关系

```
盘中选手任务（互相独立，并行运行）
  ├── value-news-scan (08:30)
  ├── quant-factor-scan (09:00) ← 周一含LightGBM调仓(rebalance.py)
  ├── hotspot-morning-scan (09:15)
  ├── trader-morning (09:20)
  ├── sentiment-morning-scan (09:25)
  ├── dwj-morning-scan (09:35)
  ├── factor-mining-noon (12:00)
  ├── trader-afternoon (13:30)
  ├── dwj/sentiment/hotspot-afternoon-scan (14:50)
  └── intraday-nav-refresh (每30min) ← nav_update_all.py --intraday

因子挖掘（独立）
  ├── factor-mining-noon (12:00)
  ├── factor-mining-afternoon (15:30)
  ├── daily-factor-mining (21:00)
  └── factor-mining-weekend (10/15/20, 周末)

收盘流程（串行依赖）
  sentiment-close-update (15:05)
    → dwj-close-update (15:10) + hotspot-close-update (15:10)
      → nav-update (15:15) ← nav_update_all.py（含一致性自检）
        → frontend-deploy (15:30)
          → daily-review (15:45)

日常维护
  └── daily-log-push (23:59)
```

## 关键文件

| 文件 | 用途 |
|------|------|
| `paper-trading-data.json` | 竞赛核心数据（选手持仓、净值、交易、决策日志） |
| `scripts/rebalance.py` | ⚠️ 标准化调仓脚本（所有调仓必须走这里） |
| `scripts/nav_update_all.py` | ⚠️ 标准化净值更新脚本（含一致性自检） |
| `skills/paper-trading/scripts/trading_engine.py` | 交易引擎（save_data含保存前自检） |
| `paper-trading-state.json` | 任务状态管理 |
| `paper-trading.html` | 竞赛前端页面 |
| `factors.json` | 因子库元数据 |
| `scripts/feature_engine.py` | LightGBM特征工程（99+特征） |
| `scripts/ml_stock_selector.py` | LightGBM模型训练/预测 |
| `scripts/risk_control.py` | 风控模块（ST/减持/行业/集中度） |

## 数据一致性 Bug 历史（前车之鉴）

| 日期 | Bug | 根因 | 修复 |
|------|-----|------|------|
| 3/2 | positions volume=0 | execute_trade被绕过 | 手动重建 |
| 3/6 | 空仓NAV=1.0 | total_value未重算 | engine加即时重算 |
| 3/6 | 基准NAV偏高 | 起始值用盘中值 | 修正为收盘价 |
| 3/9 | 因子猎人24只全volume=0 | ML脚本字段名不一致+调仓目标引用旧持仓 | rebalance.py标准化 |
| 3/10 | 技术猎手total_value=10M | 卖出后total_value未重算 | save_data加自检 |
