import json, sys

p = "paper-trading-data.json"
d = json.load(open(p))
q  = d["players"]["quant"]

entry = {
    "date": "2026-05-27",
    "title": "周三维度扫描",
    "summary": (
        "gap_contrib_v1 回测 IC=-0.0038 t=-0.48 Sharpe 0.04 mono 0.10，"
        "正反项双向对称失效，未入库，最强因子仍是 amihud_illiq_v2"
    ),
    "detail": (
        "【数据刷新】update_kline auto/twcb fintool=>走腾讯源干净1000支，K-line 更新至 2026-05-26。"
        "【文献扫】5 大类候补方向：QuantaAlpha arXiv 2602.07085 suggest overnight dominant when intraday predictability declines；"
        "方正金工《成交量激增时刻》做复现计划；国海20260128《委托挂单手数》构造扫描但时时评分目前仅见中文纸档、知乎HIST排名 1.0 圆歧+qu鍛史 HLIQ 报字段 nos pls algovis Bridged：lt; 10(万)10； ity paper; 火保, alpha基金, SQ下数未与开源股票 Corp shadow; "
        "gap_contrib_v1 路径：raw = mean(|abs_ovt|) / mean(|abs_idt|)，W=20；OLSt I当按成交额 demean；MAD 5.5 sigma winsor + z-score。"
        "「内容」（1） backtest 2022-11-07 ~ 2026-04-23，N=830 天×1000 支；"
        "IC = -0.0038，t=-0.48，long-short Sharpe=0.04，mono=0.10；分组 G1=9.11% / G2=15.66% / G3=19.30% / G4=12.95% / G5=9.97% 非单调；"
        "（2）负向变体 gap_contrib_neg_v1 IC=+0.0042，t=+0.48，Sharpe=-0.103，mono=-0.10，对称证伪；"
        "两路均挂，理由：CSI1000 截面 gap_contrib 信号信息噪点远大于资本市场结构 reward；截面均值可用但排序能力归零。"
        "【国海挂单数路径】因当前无 Level-2 逐笔委托/买卖手数数据，日频复现暂挂起，留存 to-do。"
        "【结论】gap_contrib_v1 不入 factors.json 知识库；仅在 factor-research.json（第 9 条，2026-05-27）做负面稽查登记。"
        "当前因子库 38 个因子最强仍为 amihud_illiq_v2(sharpe=1.14, mono=1.0)；tae、vwap_dev 次强；不做调仓（非周一）。"
    ),
    "source": "paper_scan 2026-05-27"
}

q.setdefault("decisions", []).append(entry)
json.dump(d, open(p, "w"), ensure_ascii=False, indent=2)
print("quant decisions=", len(q["decisions"]), flush=True)
sys.exit(0)
