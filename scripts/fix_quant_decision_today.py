import json, sys

p = "paper-trading-data.json"
d  = json.load(open(p))
q  = d["players"]["quant"]

# drop the corrupted last entry (today)
if q["decisions"] and q["decisions"][-1]["date"] == "2026-05-27":
    q["decisions"].pop()

CLEAN = {
    "date": "2026-05-27",
    "title": "周三维度因子扫描",
    "summary": (
        "gap_contrib_v1 复现失败 IC=-0.004 t=-0.48 Sharpe 0.04 mono 0.1；"
        "负向变体对称失效；不入因子库，最强仍为 amihud_illiq_v2"
    ),
    "detail": (
        "今日扫描走4条文献路径：\n"
        "  1. QuantaAlpha arXiv 2602.07085v3 提出'日内可预测性退化后隔夜成为主导信号'，"
        "据此按gap贡献率gap_contrib_v1 = MA20(|ovret|)/MA20(|idret|) 构造日频因子，"
        "成交额OLS中性化 → MAD winsor → z-score。\n"
        "  2. 国海20260128金工系列委托挂单手数方向、方正金工成交量激增方向两路均因数据位级不足未能落地复现。\n"
        "回测结果：截面830×1000，IC=-0.0038,t=-0.48，多空Sharpe=0.04，单调性=0.10；"
        "反项gap_contrib_neg_v1对称证伪IC=+0.004,t=+0.48,Sharpe=-0.104。"
        "分组 G3最高19.3% G1/G5均约9%，凸形非单调，截面无排序列选股能力。"
        "结论：不入factor库，仅记factor-research.json已处理。非调仓日，不执行rebalance。"
    ),
    "source": "paper_scan 2026-05-27",
}

q["decisions"].append(CLEAN)
json.dump(d, open(p, "w"), ensure_ascii=False, indent=2)
print("OK quant decisions=", len(q["decisions"]), flush=True)
