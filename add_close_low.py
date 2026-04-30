#!/usr/bin/env python3
"""Add close_low_v1 to factors.json"""
import json

factors = json.load(open('factors.json'))

new_entry = {
    "id": "close_low_v1",
    "name": "低点收盘位置因子 v1",
    "name_en": "Low Close Location v1",
    "category": "反转/日内效应",
    "description": "过去20日日内低点位置：(最高价-收盘价)/(最高价-最低价)的20日均值。高因子值=持续收在日内低点=日内卖压累积，次日倾向于均值回复反弹。正交于vwap_dev(尾盘均价)和shadow_pressure(上下影线比)。",
    "hypothesis": "收盘持续触及日内最低的股票，日内卖方抛压(散户恐慌/机构撤退)已充分释放，次日(5日)倾向于均值回复。A股中证1000日内反转效应强(尾盘卖→隔夜涨)。BARRA Reversal方向。",
    "expected_direction": "正向（高low_clv = 收在日内下方 → 次日正向收益）",
    "formula": "MA20((high-close)/(high-low+1e-8)), 成交额OLS中性化+3σ MAD Winsorize+z-score",
    "direction": 1,
    "stock_pool": "中证1000",
    "period": "2022-10-11 ~ 2026-04-30",
    "n_groups": 5,
    "rebalance_freq": 20,
    "forward_days": 5,
    "cost": 0.003,
    "factor_type": "日内反转因子",
    "source_type": "自研(日内反转Paper启发)",
    "source_title": "日内反转效应 + CLV低点提取因子本土化",
    "source_url": "https://zhuanlan.zhihu.com/p/644380633",
    "barra_style": "Reversal",
    "correlations": {},
    "metrics": {
        "ic_mean": 0.0276,
        "ic_t_stat": 3.23,
        "rank_ic_mean": 0.0433,
        "ir": 0.2650,
        "long_short_sharpe": 1.431,
        "long_short_mdd": -0.130,
        "g5_sharpe": 1.351,
        "g5_ann_return": 0.3552,
        "g1_ann_return": 0.0656,
        "g2_ann_return": 0.1524,
        "g3_ann_return": 0.1964,
        "g4_ann_return": 0.2393,
        "monotonicity": 1.0,
        "turnover_mean": 0.179,
        "ic_positive_pct": 0.593,
        "max_ic_correlation": 0.064
    },
    "nav_data": "output/close_low_v1/cumulative_returns.json",
    "ic_data": "output/close_low_v1/ic_series.json",
    "conclusion": "低点收盘位置因子 IC=0.028(t=3.23,1%显著✓), G5 Sharpe=1.35✓, LS Sharpe=1.43, 单调性=1.0✓。换手率仅17.9%极低。日均相关性最高仅为0.064→新因子独立。Barra Style: Reversal。日内卖压释放完毕→次日反弹。",
    "lessons_learned": [
        "CLV低点方向(做多收在日内下限的股票)在5天前瞻窗口极有效——不需要long-only_integrator配合日内反转效应。",
        "低换手率说明该因子信号稳定且不与高换手因子矛盾; 适合与vwap_dev(高换手77%)、shadow_pressure(高换手)形成互补。",
        "Barra: Reversal增加覆盖率; high_volatility因子主要捕捉Windows重构,但中心位置因子都捕获主要方向针对证券的异常收益的重要因素。"
    ],
}

factors.append(new_entry)
with open('factors.json', 'w', encoding='utf-8') as f:
    json.dump(factors, f, indent=2, ensure_ascii=False)
print(f'factors.json updated: {len(factors)} factors total')
