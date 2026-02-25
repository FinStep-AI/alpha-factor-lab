# Alpha Factor Lab — 量化因子研究工作流 v2.2

## 描述
量化因子研究（Quant Factor Research）完整工作流。从研报/新闻/投资 idea 输入，到因子构建、衰减分析、单因子回测评估的完整链路。模仿专业量化公司 Quant Researcher 的日常研究流程。

## 触发条件
当用户提到以下关键词时激活本技能：
- 因子研究、因子挖掘、alpha 挖掘
- 因子回测、单因子测试、因子检验
- 因子衰减、因子 IC、因子分层
- 量化研究、alpha factor
- 美股因子、US factor

## 工作流程

### Step 1: Alpha Idea 输入
接受用户输入的投资 idea，可以是：
- 一段研报摘要
- 一条新闻 / 市场观察
- 一个直觉（如「高研发投入的公司长期收益更好」）

**输出：** 将 idea 提炼为一句话描述。

### Step 2: 因子定义
将 idea 转化为可量化的因子：
- **因子名称**（英文缩写，如 `rd_intensity`）
- **描述**（中文）
- **计算公式**（Python 表达式，参考 `references/factor-guide.md`）
- **数据需求**（需要哪些字段、哪些 MCP 工具）
- **预期方向**（因子值越大/越小，预期收益越高）
- **是否静态**（截面因子 vs 时变因子——影响 IC 计算和衰减分析）

### Step 3: 数据获取

根据目标市场选择数据获取方式：

#### A股路径（fintool MCP）

通过 mcporter 调用 MCP 金融工具获取数据。PATH 需包含 `/home/node/.local/bin`。

```bash
export PATH="/home/node/.local/bin:$PATH"

# K线
mcporter call fintool-quote.get_kline keyword=平安银行 kline_type=day kline_num=100 end_date=2025-12-31

# 研发费用（需要 end_date）
mcporter call fintool-company.get_research_development_expense keyword=海康威视 end_date=2025-12-31

# 指数成分股
mcporter call fintool-index.get_index_constituent index_code=000300
```

**⚠️ K线分批拉取：** MCP 的 `get_kline` 单次最多返回 100 条。如需拉取长周期数据（如 3 年 ≈ 750 个交易日），需要分批拉取：
1. 第一批：`end_date=2026-02-14, kline_num=100`
2. 取本批最早日期，前一天作为下一批的 `end_date`
3. 重复直到数据量足够，注意去重
4. 建议支持**断点续传**（记录已完成的股票），避免大批量任务被中断后重头开始

参考实现：`scripts/fetch_kline_3y.py`

#### 美股路径（yfinance / us-market skill）

使用 us-market skill 或直接用 yfinance 获取美股数据：

```bash
# 获取历史K线（用 us-market skill）
python3 skills/us-market/scripts/us_market_query.py --type history --symbol AAPL --period 5y --interval 1d

# 获取财务数据
python3 skills/us-market/scripts/us_market_query.py --type financials --symbol AAPL --statement income
```

**批量获取美股数据（推荐直接用 yfinance）：**

```python
import yfinance as yf

# 批量下载多只股票K线
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
data = yf.download(tickers, period='3y', interval='1d', group_by='ticker')

# 获取单只股票财务数据
t = yf.Ticker('AAPL')
income = t.income_stmt        # 利润表
balance = t.balance_sheet     # 资产负债表
cashflow = t.cashflow         # 现金流量表
```

**美股指数成分股获取：**
- S&P 500：从 Wikipedia 抓取 `https://en.wikipedia.org/wiki/List_of_S%26P_500_companies`
- NASDAQ-100：`https://en.wikipedia.org/wiki/Nasdaq-100`
- 也可用 FMP API：`https://financialmodelingprep.com/stable/sp500-constituent?apikey=KEY`

**⚠️ yfinance 速率限制：** Yahoo Finance 有隐式限速，批量拉取时建议：
- 每只股票间隔 0.5-1 秒
- 一次 download 不超过 50 只
- 被限速后等待 5-10 分钟自动恢复

### Step 4: 因子计算
```bash
python3 scripts/factor_calculator.py \
  --formula "net_profit / market_cap" \
  --data data/stock_data.csv \
  --output data/factor_values.csv \
  [--neutralize market_cap,log_market_cap] \
  [--industry-col industry] \
  [--winsorize 3.0] \
  [--winsorize-method mad|percentile] \
  [--log-transform] \
  [--rank-transform] \
  [--no-zscore]
```

**v2 新增参数：**
- `--industry-col`：行业列名，做行业中性化（OLS 回归取残差）
- `--neutralize`：支持多变量逗号分隔（如 `market_cap,log_market_cap`）
- `--winsorize-method mad`：默认 MAD 方法（更鲁棒）
- `--log-transform`：sign(x) × log(1 + |x|)，处理右偏分布
- `--rank-transform`：截面百分位排名 (0~1)

**安全改进：** 公式执行使用 AST 白名单验证，禁止 import/exec/eval。

### Step 5: 因子衰减分析
```bash
# 标准模式（时变因子）
python3 scripts/factor_decay.py \
  --factor data/factor_values.csv \
  --max-lag 20 \
  --output-report data/decay_report.json \
  --output-chart output/decay_chart.png

# IC 衰减模式（静态因子推荐）
python3 scripts/factor_decay.py \
  --factor data/factor_values.csv \
  --returns data/returns.csv \
  --ic-decay \
  --max-lag 60 \
  --ic-step 5 \
  --output-report data/ic_decay_report.json
```

**v2 改进：**
- 自动检测静态因子并警告（自相关恒 ≈ 1 无意义）
- 新增 `--ic-decay` 模式：分析不同前瞻窗口的 IC 衰减
- tau 拟合上限动态调整为 5 × max_lag

**⚡ 联动回测：** 衰减分析的输出报告可直接传给 Step 6 的 `--decay-report`，自动设置最优调仓周期，无需手动猜测。推荐流程：先跑衰减 → 再跑回测。

### Step 6: 单因子回测
```bash
# 方式一：手动指定调仓周期
python3 scripts/factor_backtest.py \
  --factor data/factor_values.csv \
  --returns data/returns.csv \
  --n-groups 5 \
  --rebalance-freq 20 \
  --forward-days 20 \
  --cost 0.002 \
  --output-report data/backtest_report.json \
  --output-dir output/ \
  [--static-factor]

# 方式二（推荐）：从衰减报告自动设置调仓周期
python3 scripts/factor_backtest.py \
  --factor data/factor_values.csv \
  --returns data/returns.csv \
  --n-groups 5 \
  --decay-report data/decay_report.json \
  --cost 0.002 \
  --output-report data/backtest_report.json \
  --output-dir output/ \
  [--static-factor]
```

**`--decay-report` 联动逻辑：**
- 如果衰减报告含 IC 衰减数据（`--ic-decay` 模式产出），取 |IC| 最大的前瞻窗口作为调仓周期
- 否则取自相关半衰期（`half_life`），四舍五入为整数天
- 同时自动设置 `--forward-days`（IC 前瞻天数）保持一致
- 如果用户手动指定了 `--rebalance-freq`（非默认值 20），则优先用户指定，不覆盖

**v2 新增参数：**
- `--forward-days`：IC 前瞻天数（默认 20），基本面因子建议 ≥ 20
- `--cost`：单边交易成本（如 0.002 = 0.2%）
- `--static-factor`：强制标记为静态因子

**v2 核心改进：**
- **静态 vs 动态 IC**：静态因子使用不重叠窗口，避免 IC 自相关膨胀
- **前瞻收益修正**：正确使用 t+1 到 t+N 的累计收益
- **Newey-West t 检验**：修正自相关后检验 IC 统计显著性
- **年化收益安全计算**：处理负收益溢出和极端幂次
- **交易成本扣除**：调仓时按换手比例双边扣除

### Step 7: 可视化
```bash
python3 scripts/visualizer.py \
  --backtest-report data/backtest_report.json \
  --decay-report data/decay_report.json \
  --factor data/factor_values.csv \
  --output-dir output/ \
  --factor-name "R&D Efficiency"
```

**⚠️ 中文字体：** 标题建议用英文，多数服务器环境无 CJK 字体，中文会显示为方块。

**路径查找逻辑：** visualizer 会在 `--backtest-report` 所在目录和 `--output-dir` 两个位置搜索中间数据文件（`cumulative_returns.json`、`ic_series.json`）。如果 backtest-report 和 output-dir 不在同一目录，确保中间数据在其中之一即可。

**💡 小技巧：** 如果 `--backtest-report` 和 `--output-dir` 不同，可以将 backtest report 复制到 output-dir 下再跑 visualizer，确保所有文件在一起。

生成图表：
| 图表 | 文件名 | 数据依赖 | 说明 |
|------|--------|----------|------|
| 分层净值曲线 | `quintile_returns.png` | `cumulative_returns.json` | 各组 + 多空的资金曲线，最直观的因子效果图 |
| IC 时序图 | `ic_series.png` | `ic_series.json` | IC 柱状 + 滚动均值 + 累计 IC，含 NW 显著性标注 |
| 回测摘要 | `backtest_summary.png` | backtest report JSON | 四宫格：分层收益、指标表、Sharpe、MDD |
| 因子分布 | `factor_distribution.png` | factor CSV | 直方图 + QQ 图，诊断因子分布形态 |
| 衰减图 | `factor_decay.png` | decay report JSON | 自相关衰减 + 指数拟合 + 半衰期标注 |

## 评估标准速查
| 指标 | 优秀 | 良好 | 一般 | 差 |
|------|------|------|------|----|
| |IC Mean| | > 0.05 | 0.03~0.05 | 0.02~0.03 | < 0.02 |
| IR | > 0.5 | 0.3~0.5 | 0.2~0.3 | < 0.2 |
| NW t-stat | > 2.58 (1%) | > 1.96 (5%) | > 1.65 (10%) | < 1.65 |
| 多空 Sharpe | > 1.5 | 1.0~1.5 | 0.5~1.0 | < 0.5 |
| Half-life | > 20天 | 10~20天 | 5~10天 | < 5天 |
| Turnover | < 0.2 | 0.2~0.35 | 0.35~0.5 | > 0.5 |
| 单调性 | > 0.9 | 0.7~0.9 | 0.5~0.7 | < 0.5 |

## 常见陷阱 Checklist（每次回测前过一遍）
- [ ] 行业中性化了吗？（`--industry-col`）
- [ ] 市值中性化了吗？（`--neutralize log_market_cap`）
- [ ] 因子分布右偏吗？需要 `--log-transform` 吗？
- [ ] 静态因子用了正确的 IC 计算方式吗？（`--static-factor`）
- [ ] 样本量足够吗？（建议 ≥ 100 只股票）
- [ ] 前瞻收益窗口合理吗？（日频技术因子用 1-5d，基本面用 20d+）
- [ ] 考虑了交易成本吗？（`--cost`）
- [ ] 存在存活偏差吗？（用历史成分股而非当前成分股）

## Step 8: 写入阿尔法工坊前端

每次因子回测完成后，**必须**将结果写入前端展示：

1. 读取 `factors.json`
2. 按下面的 **标准格式** 追加/更新一条记录（按 `id` 匹配）
3. 回测产出的 `cumulative_returns.json` 和 `ic_series.json` 放在 `output/{factor_id}/` 目录下
4. 在因子记录中设置 `nav_data` 和 `ic_data` 路径指向这些文件
5. 写入后 commit 并 push 到 GitHub

**阿尔法工坊地址：** https://finstep-ai.github.io/alpha-factor-lab/factor-backtest.html

### factors.json 标准字段格式

⚠️ **严格遵循此格式，否则前端无法正常渲染！**

```json
{
  "id": "factor_name_v1",
  "name": "因子中文名",
  "name_en": "Factor English Name",
  "formula": "简短公式描述",
  "description": "因子详细描述（1-2句话）",
  "category": "流动性|风险|动量|价值|基本面|技术",
  "stock_pool": "中证1000",
  "period": "2022-10 ~ 2026-02",
  "rebalance_freq": 20,
  "forward_days": 20,
  "cost": 0.002,
  "direction": "positive|negative",
  "expected_direction": "正向|负向",
  "factor_type": "量价|基本面",
  "hypothesis": "因子逻辑假设（1句话）",
  "rating": "★★★★ 强|★★★ 可用|★★☆ 弱|☆☆☆ 失效",
  "conclusion": "回测结论（2-3句话，包含关键指标和最终判断）",
  "lessons_learned": ["教训1", "教训2", "教训3"],
  "upgrade_notes": "升级说明（如有）",
  "nav_data": "output/factor_name_v1/cumulative_returns.json",
  "ic_data": "output/factor_name_v1/ic_series.json",
  "created": "2026-02-24",
  "updated": "2026-02-24",
  "metrics": {
    "ic_mean": 0.028,
    "ic_std": 0.113,
    "ic_t": 2.76,
    "ic_positive_ratio": 0.60,
    "ic_significant": true,
    "rank_ic": 0.048,
    "ir": 0.244,
    "long_short_total": 0.608,
    "long_short_annual": 0.160,
    "long_short_annual_return": 0.160,
    "long_short_sharpe": 1.14,
    "long_short_mdd": -0.186,
    "turnover": 0.245,
    "monotonicity": 1.0,
    "group_returns_annualized": [0.018, 0.063, 0.096, 0.136, 0.214],
    "group_sharpe": [0.06, 0.24, 0.40, 0.58, 0.94],
    "group_mdd": [-0.49, -0.42, -0.35, -0.36, -0.32]
  }
}
```

**关键注意事项：**
- `metrics` 必须是嵌套对象，指标不能平铺在顶层
- `group_returns_annualized`、`group_sharpe`、`group_mdd` 必须是**数组**（[G1, G2, ..., G5]），不能是对象 `{G1: v}`
- `lessons_learned` 必须是**字符串数组**，不能是单个字符串
- `conclusion` 是必填字段，前端详情页底部显示
- 收益/回撤等指标用**小数**（0.16 而非 16%），前端会自动格式化
- `long_short_annual_return` 和 `long_short_annual` 保持相同值（兼容性）

## 目录结构
```
alpha-factor-lab/
├── SKILL.md                    # 本文件
├── factors.json                # 因子数据（前端读取）
├── fundamental-reports.json    # 基本面报告数据
├── output/                     # 回测产出（净值曲线/IC序列等）
├── logs/                       # 模型调用日志（按日期）
├── references/
│   └── factor-guide.md         # 因子分类、公式规范、评价标准
├── scripts/
│   ├── factor_calculator.py    # 因子计算引擎 v2
│   ├── factor_decay.py         # 因子衰减分析 v2
│   ├── factor_backtest.py      # 单因子回测引擎 v2.1
│   └── visualizer.py           # 可视化模块 v2.1
```

## 更新日志

### v2.2 (2026-02-24)
- **SKILL.md:**
  - 新增 factors.json 标准字段格式规范（完整模板 + 注意事项）
  - 明确 metrics 嵌套结构、数组类型、必填字段要求
  - 避免新因子写入时格式不一致导致前端渲染失败

### v2.1 (2026-02-16)
- **factor_backtest.py:**
  - 新增 `--decay-report` 参数，自动从衰减报告读取半衰期设置调仓周期和 IC 前瞻天数
  - 修复 numpy bool/int/float 类型的 JSON 序列化错误
- **visualizer.py:**
  - 修复中间数据路径查找：同时搜索 backtest-report 目录和 output-dir 目录
  - 分层净值曲线（资金曲线）现在能可靠生成
- **SKILL.md:**
  - 新增 K 线分批拉取和断点续传说明
  - 新增衰减分析 → 回测联动流程说明
  - 新增可视化图表清单和路径查找说明
  - 新增中文字体注意事项
