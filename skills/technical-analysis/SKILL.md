---
name: technical-analysis
description: 股票技术分析工具，生成专业K线图表（含均线、BOLL、MACD、KDJ、RSI、OBV）和技术分析报告。支持A股、港股、美股。当用户请求分析股票走势、技术形态、买卖时机时使用。
---

# 技术分析 (Technical Analysis)

股票技术分析工具，自动生成6面板专业图表，输出到阿尔法工坊前端。支持A股、港股、美股。

## 市场识别

| 市场 | 代码格式 | 示例 |
|------|---------|------|
| A股 | 6位数字 或 带.SH/.SZ后缀 | 600519、000858.SZ |
| 港股 | 4-5位数字 或 带.HK后缀 | 0700、09988.HK |
| 美股 | 英文字母ticker | AAPL、TSLA、NVDA |

## 执行流程

### 步骤1: 获取K线数据

**A股** — 使用 fintool MCP 或东方财富 API:
```bash
python3 ~/openclaw/skills/technical-analysis/scripts/fetch_kline.py --market A --code 600519 --days 120
# 输出: /tmp/ta_{code}_kline.json
```

**美股/港股** — 使用 yfinance:
```bash
python3 ~/openclaw/skills/technical-analysis/scripts/fetch_kline.py --market US --code AAPL --days 120
python3 ~/openclaw/skills/technical-analysis/scripts/fetch_kline.py --market HK --code 0700 --days 120
# 输出: /tmp/ta_{code}_kline.json
```

### 步骤2: 生成图表
```bash
python3 ~/openclaw/skills/technical-analysis/scripts/stock_chart.py \
    /tmp/ta_{code}_kline.json \
    "{stock_name}" "{stock_code}" \
    "/tmp/ta_{code}_chart.png"
```

图表包含6个面板：
| 面板 | 内容 |
|------|------|
| 主图 | K线 + MA5/10/20/60/120 + BOLL布林带 + 买卖点(B/S) |
| 副图1 | 成交量 + MA5/MA10 + OBV能量潮 |
| 副图2 | MACD (DIF/DEA/柱状图) |
| 副图3 | KDJ (K/D/J + 超买超卖区) |
| 副图4 | RSI (RSI6/RSI12 + 超买超卖区) |
| 底部 | 日期轴 + 近期信号汇总 |

### 步骤3: 撰写技术分析报告

按照 `references/report-template.md` 中的模板撰写分析报告。重点包括：
- 趋势研判（短期/中期）
- 买卖点信号
- 量能分析
- 各技术指标状态
- 关键支撑位/压力位
- 操作建议

### 步骤4: 写入阿尔法工坊 + 推送

1. 将图表上传到 GitHub repo（`alpha-factor-lab/charts/` 目录）
2. 将分析结果写入 `alpha-factor-lab/technical-reports.json`
3. git add + commit + push

**JSON 结构：**
```json
{
  "name": "贵州茅台",
  "code": "600519.SH",
  "market": "A",
  "date": "2026-02-22",
  "price": 1485.30,
  "change_pct": -0.09,
  "trend": "震荡偏强",
  "signal": "持有观望",
  "chart": "charts/600519_chart.png",
  "summary": "一句话总结...",
  "analysis": {
    "trend": { "short": "...", "mid": "...", "ma_status": "..." },
    "signals": [
      { "type": "MACD金叉", "date": "02/18", "strength": "中等" }
    ],
    "volume": { "summary": "...", "obv_trend": "..." },
    "indicators": {
      "macd": "...",
      "kdj": "...",
      "rsi": "...",
      "boll": "..."
    },
    "key_levels": {
      "strong_resistance": 1600,
      "resistance": 1550,
      "support": 1450,
      "strong_support": 1400
    },
    "action": {
      "buy_strategy": "...",
      "sell_strategy": "...",
      "stop_loss": "..."
    }
  },
  "risks": ["..."]
}
```

**对于已有相同 code 的报告，覆盖更新。**

### 步骤5: 在飞书发送简要结果

发送简要的技术分析结论（趋势/信号/关键价位/操作建议），不需要发完整报告。

## 自动识别的买卖信号

- **MACD金叉/死叉**: DIF与DEA交叉
- **KDJ金叉/死叉**: K与D线交叉（低位/高位更有效）
- **均线交叉**: MA5与MA10交叉
- **BOLL突破**: 股价突破/跌破布林带上下轨
- **背离信号**: 股价与指标方向不一致

## 数据源

| 市场 | 主数据源 | 备用数据源 |
|------|---------|-----------|
| A股 | 东方财富 API | fintool MCP |
| 美股 | yfinance | Alpha Vantage |
| 港股 | yfinance | — |

## 输出

- 图表: `alpha-factor-lab/charts/{code}_chart.png`
- 数据: `alpha-factor-lab/technical-reports.json`
- 前端: `https://oscar2sun.github.io/alpha-factor-lab/technical.html`

## 更多信息

- 分析框架和报告模板见 `references/report-template.md`
- 技术指标详解见 `references/indicators.md`
