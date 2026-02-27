# 定时任务总览

> 自动更新时间：2026-02-27 | 全部按工作日（周一~周五）运行 | 时区：Asia/Shanghai | 共 15 个任务

## 一、竞赛选手任务（盘中）

| 时间 | 任务名 | 选手 | 内容 | 超时 |
|------|--------|------|------|------|
| 08:30 | `value-news-scan` | 巴菲特门徒 🏛️ | 新闻扫描（web_search / 东方财富快讯 API），分析持仓影响，生成交易建议 | 300s |
| 09:00 | `quant-factor-scan` | 因子猎人 🔢 | 研报扫描 → 因子构建 → 回测验证 → 入库/淘汰；**周一额外执行调仓**（双因子 TOP25 等权） | 1800s |
| 09:20 | `trader-morning` | 技术猎手 📊 | 早盘技术分析：选标的 / MACD·KDJ·RSI·BOLL 打分 / 仓位决策 | 600s |
| 09:25 | `sentiment-morning-scan` | 情绪战神 🔥 | 情绪温度计+龙头识别+连板梯队+仓位决策（周期驱动） | 300s |
| 09:35 | `dwj-morning-scan` | 趋势游侠 🏄 | 全 A 实时行情 → 五级漏斗扫描 → 止盈止损 → 买入交易 | 300s |
| 13:30 | `trader-afternoon` | 技术猎手 📊 | 午盘二次分析，验证早盘判断，检查止损（亏 7% 清仓）止盈（赚 20% 减半） | 600s |
| 14:50 | `dwj-afternoon-scan` | 趋势游侠 🏄 | 尾盘扫描，止盈止损检查，极强信号（score≥65 + 板块集中）可追入 | 300s |
| 14:50 | `sentiment-afternoon-scan` | 情绪战神 🔥 | 封板确认+龙头炸板检查+风控（断板次日砍） | 300s |

## 二、收盘流程（15:00 ~ 16:00）

按顺序依次执行：

| 时间 | 任务名 | 内容 | 超时 |
|------|--------|------|------|
| 15:05 | `sentiment-close-update` | 情绪战神收盘情绪温度更新 + 止损止盈交易 + 净值更新 | 300s |
| 15:10 | `dwj-close-update` | 趋势游侠收盘行情拉取 + K 线缓存更新 + 止盈止损交易 + 净值确认 | 300s |
| 15:15 | `nav-update` | 全部选手净值计算（收盘价）+ 沪深 300 基准净值 + 排行榜更新 | 600s |
| 15:30 | `frontend-deploy` | `paper-trading-data.json` + 前端文件 git push → GitHub Pages 自动部署 | 600s |
| 15:45 | `daily-review` | 生成收盘战报飞书卡片 → 发送至竞赛群 `oc_8970b8b266cfc574adf6431d1720d387` | 600s |

## 三、盘中辅助

| 时间 | 任务名 | 内容 | 超时 |
|------|--------|------|------|
| 每 30 分钟（09:00–14:30） | `intraday-nav-refresh` | 盘中实时净值刷新，静默运行（不发消息） | 120s |

## 四、日常维护

| 时间 | 任务名 | 内容 | 超时 |
|------|--------|------|------|
| 23:59（**每天**，含周末） | `daily-log-push` | `logs/` 目录变更 git push 到 GitHub，网络不通则跳过 | 90s |

## 任务依赖关系

```
盘中选手任务（互相独立，并行运行）
  ├── value-news-scan (08:30)
  ├── quant-factor-scan (09:00)
  ├── trader-morning (09:20)
  ├── sentiment-morning-scan (09:25)
  ├── dwj-morning-scan (09:35)
  ├── trader-afternoon (13:30)
  ├── dwj-afternoon-scan (14:50)
  ├── sentiment-afternoon-scan (14:50)
  └── intraday-nav-refresh (每30min)

收盘流程（串行依赖）
  sentiment-close-update (15:05)
    → dwj-close-update (15:10)
    → nav-update (15:15)       ← 依赖所有选手当日交易完成
      → frontend-deploy (15:30) ← 依赖净值数据更新
        → daily-review (15:45)  ← 依赖前端数据已 push

日常维护
  └── daily-log-push (23:59)   ← 独立，无依赖
```

## 关键文件

| 文件 | 用途 |
|------|------|
| `paper-trading-data.json` | 竞赛核心数据（选手持仓、净值、交易、决策日志） |
| `paper-trading-state.json` | 任务状态管理（should-run / done 控制） |
| `paper-trading.html` | 竞赛前端页面 |
| `data/a_share_kline_cache.json` | 趋势游侠 K 线缓存（5396 只，80 天） |
| `data/sentiment_history.json` | 情绪战神情绪温度历史 |
| `factors.json` | 因子库元数据 |
| `scripts/intraday_nav_refresh.py` | 盘中净值刷新脚本 |
| `scripts/feishu_send_card.py` | 飞书卡片发送工具 |

## 选手策略参考文档

| 选手 | 参考文档 |
|------|----------|
| 因子猎人 🔢 | `references/player_quant.md` |
| 技术猎手 📊 | `references/player_trader.md` |
| 巴菲特门徒 🏛️ | `references/player_value.md` |
| 趋势游侠 🏄 | `references/player_dwj.md` |
| 情绪战神 🔥 | `references/player_sentiment.md` |
