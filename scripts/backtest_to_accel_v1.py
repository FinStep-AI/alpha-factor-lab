#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: to_accel_v1 — 换手率加速因子 v1
================================================
构造:
  1. 换手率 = volume / amount * 100
  2. ma_short = MA10(换手率), ma_long = MA40(换手率)
  3. factor_raw = log(ma_short / ma_long)  → 正值=换手加速，负值=换手减速
  4. 成交额OLS中性化 (MAD winsorize + z-score)
  5. 不做额外方向反转（原始方向: 换手加速→做多）

逻辑:
- 换手率加速(MA10 > MA40相对水平)意味着:
  ① 价格关注度提升(Merton投资者认知假说)
  ② 信息传播加速(Lee&Swaminathan 2000)
  ③ 知情交易者活跃度提升 → 后续动能延续
- 与turnover_decel_v1(做多换手减速)方向接近但角度不同:
  - turnover_decel: 短期短期动量反转(放量→回吐)
  - to_accel: 中期趋势延续(换手持续上升→趋势确立)
- 换手加速在中证1000小盘股上更显著: 小盘股流动性差，换手率放大往往反映信息驱动

来源: 自研 | 参考: Lee & Swaminathan (2000) JFE
"""

import json
import sys
import time
import warnings
from pathlib import Path
from numpy.linalg import lstsq

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ────────────────── 参数 ──────────────────
SHORT_W = 10
LONG_W = 40
FORWARD_DAYS = 20
REBALANCE_FREQ = 20
N_GROUPS = 5
COST = 0.002
WINSORIZE_PCT = 0.05
FACTOR_ID = "to_accel_v1"

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "skills" / "alpha-factor-lab" / "scripts"
DATA_PATH = BASE_DIR / "data" / "csi1000_kline_raw.csv"
OUTPUT_DIR = BASE_DIR / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

t0 = time.time()

# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
print(f"   数据范围: {df['date'].min().date()} ~ {df['date'].max().date()}")

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")
volume_piv = df.pivot_table(index="date", columns="stock_code", values="volume")
ret_piv = close_piv.pct_change()

# 换手率 = volume / amount * 100 (注意单位)
turnover_piv = (volume_piv / amount_piv * 100).clip(lower=0.01)

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股  ({time.time()-t0:.1f}s)")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造换手率加速因子 (MA{SHORT_W}/MA{LONG_W})...")
t0_factor = time.time()

ma_short = turnover_piv.rolling(SHORT_W, min_periods=int(SHORT_W*0.7)).mean()
ma_long = turnover_piv.rolling(LONG_W, min_periods=int(LONG_W*0.7)).mean()

ratio = ma_short / ma_long.clip(lower=0.01)
factor_raw = np.log(ratio.clip(lower=0.01))

print(f"   换手率加速均值: {factor_raw.stack().mean():.4f}  std: {factor_raw.stack().std():.4f}")
print(f"   非空率: {factor_raw.notna().mean().mean():.2%}  ({time.time()-t0_factor:.1f}s)")

# ────────────────── 缩尾 (截面MAD) ──────────────────
print(f"[3] 截面缩尾 ({WINSORIZE_PCT*100:.0f}%)...")
t0 = time.time()
for date in dates:
    row = factor_raw.loc[date].dropna()
    if len(row) < 10:
        continue
    lo = row.quantile(WINSORIZE_PCT)
    hi = row.quantile(1 - WINSORIZE_PCT)
    factor_raw.loc[date] = factor_raw.loc[date].clip(lo, hi)
print(f"   完成  ({time.time()-t0:.1f}s)")

# ────────────────── 成交额中性化 ──────────────────
log_amt_20 = np.log(amount_piv.rolling(20).mean().clip(lower=1))
print(f"[4] 成交额中性化 (OLS + MAD + z-score)...")
factor_neutral = factor_raw.copy()
n_neutral = 0
t0 = time.time()
for date in dates:
    f_map = factor_raw.loc[date].dropna()
    m_map = log_amt_20.loc[date].reindex(f_map.index).dropna()
    common = f_map.index.intersection(m_map.index)
    if len(common) < 30:
        continue
    f_c = f_map[common].values.astype(float)
    m_c = m_map[common].values.astype(float)
    X = np.column_stack([np.ones(len(m_c)), m_c])
    try:
        beta, _, _, _ = lstsq(X, f_c, rcond=None)
        resid = f_c - X @ beta
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        if mad > 1e-10:
            resid = np.clip(resid, med - 5 * 1.4826 * mad, med + 5 * 1.4826 * mad)
        r_mean = np.nanmean(resid)
        r_std = np.nanstd(resid)
        if r_std > 1e-10:
            factor_neutral.loc[date, common] = (resid - r_mean) / r_std
            n_neutral += 1
    except Exception:
        pass
print(f"   完成中性化: ~{n_neutral} 天  ({time.time()-t0:.1f}s)")

# ────────────────── 导入回测引擎 ──────────────────
sys.path.insert(0, str(SCRIPTS_DIR))
from factor_backtest import (
    compute_group_returns,
    compute_ic_dynamic,
    compute_metrics,
    save_backtest_data,
)

# ────────────────── 回测 ──────────────────
print(f"[5] 分层回测: {N_GROUPS}组, {REBALANCE_FREQ}天 freq, {COST*100:.1f}%成本...")
common_dates = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
common_stocks = sorted(factor_neutral.columns.intersection(ret_piv.columns))
factor_aligned = factor_neutral.loc[common_dates, common_stocks]
return_aligned = ret_piv.loc[common_dates, common_stocks]

print(f"[6] IC (forward={FORWARD_DAYS}d)...")
ic_series = compute_ic_dynamic(factor_aligned, return_aligned, FORWARD_DAYS, "pearson")
rank_ic_series = compute_ic_dynamic(factor_aligned, return_aligned, FORWARD_DAYS, "spearman")

group_returns, turnovers, holdings_info = compute_group_returns(
    factor_aligned, return_aligned, N_GROUPS, REBALANCE_FREQ, COST
)
metrics = compute_metrics(
    group_returns, ic_series, rank_ic_series, turnovers, N_GROUPS,
    holdings_info=holdings_info
)

# ────────────────── 与 turnover_decel_v1 相关性 ──────────────────
print(f"[7] turnover_decel_v1 相关性...")
try:
    # 构建 turnover_decel = -log(MA5/MA20) 用于相关
    to_decel_raw = -np.log((turnover_piv.rolling(5).mean() / turnover_piv.rolling(20).mean().clip(lower=0.01)).clip(lower=0.01))
    corrs_td = []
    for j, dt in enumerate(common_dates[:40]):
        f1 = factor_neutral.loc[dt].dropna()
        f2 = to_decel_raw.loc[dt].reindex(f1.index).dropna()
        c = f1.index.intersection(f2.index)
        if len(c) > 50:
            corr, _ = sp_stats.spearmanr(f1[c], f2[c])
            if not np.isnan(corr):
                corrs_td.append(corr)
    avg_corr_td = float(np.mean(corrs_td)) if corrs_td else None
    print(f"   与turnover_decel_v1截面Spearman: {avg_corr_td:.3f}")
except Exception as e:
    avg_corr_td = None
    print(f"   相关性计算失败: {e}")

# ────────────────── 输出报告 ──────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_backtest_data(group_returns, ic_series, rank_ic_series, str(OUTPUT_DIR))

def nan_to_none(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [nan_to_none(v) for v in obj]
    return obj

# 与turnover_decel相关
report = {
    "factor_id": FACTOR_ID,
    "factor_name": "换手率加速 v1",
    "factor_name_en": "Turnover Acceleration v1",
    "category": "流动性/情绪趋势",
    "description": f"log(MA{SHORT_W}换手率 / MA{LONG_W}换手率)，直接使用（正值=换手加速=高因子值→做多）。换手加速意味着价格关注度提升+信息传播加速+知情交易者活跃度提升，后续动能延续。成交额中性化。",
    "hypothesis": "换手率持续加速（短期换手率显著高于长期基准）表明投资者关注度快速提升、知情交易活跃、信息得以充分传播，后续价格动能延续——这是Lee&Swaminathan (2000)量价动量逻辑在换手率维度上的变体。",
    "expected_direction": "正向（换手加速=高因子值=高预期收益）",
    "factor_type": "流动性趋势/中期动量",
    "formula": f"neutralize(log(MA{SHORT_W}(turnover) / MA{LONG_W}(turnover)), log_amount_20d)",
    "direction": 1,
    "stock_pool": "中证1000",
    "period": f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}",
    "n_dates": len(common_dates),
    "n_stocks": len(common_stocks),
    "n_groups": N_GROUPS,
    "rebalance_freq": REBALANCE_FREQ,
    "forward_days": FORWARD_DAYS,
    "cost": COST,
    "data_cutoff": str(df["date"].max()),
    "barra_style": "Sentiment",
    "source_type": "自研(论文启发)",
    "source_title": "换手率趋势加速因子 (Lee & Swaminathan 2000)",
    "source_url": "https://doi.org/10.1111/j.1540-6261.2000.tb02099.x",
    "correlations": {
        "turnover_decel_v1": avg_corr_td,
    },
    "corr_note": f"与turnover_decel相关={avg_corr_td:.3f}" if avg_corr_td else "待测",
    "lessons_learned": [],
    "upgrade_notes": "v1初版。MA10/MA40组合，20日前瞻，20日调仓。探索性初版，与turnover_decel方向接近但窗口/方向不同。",
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(nan_to_none(report), f, indent=2, ensure_ascii=False)

# ────────────────── 打印摘要 ──────────────────
print(f"\n{'═'*60}")
print(f"  Turnover Acceleration v1 回测结果")
print(f"  (方向: 做多换手加速 / MA{SHORT_W}/MA{LONG_W})")
print(f"{'═'*60}")
print(f"  区间:      {report['period']}")
print(f"  股票/日期: {len(common_stocks)} / {len(common_dates)}")
ic_mean = abs(metrics.get("ic_mean", 0) or 0)
ic_t = abs(metrics.get("ic_t_stat", 0) or 0)
ls_sh = metrics.get("long_short_sharpe", 0) or 0
ls_md = metrics.get("long_short_mdd", 0) or 0
mono = metrics.get("monotonicity", 0) or 0
is_big = ic_mean > 0.015
is_t = ic_t > 2
is_sh = abs(ls_sh) > 0.5
icon_big, icon_t, icon_sh = ("✓" if is_big else "✗"), ("✓" if is_t else "✗"), ("✓" if is_sh else "✗")
n_pass = int(is_big) + int(is_t) + int(is_sh)
print(f"  IC均值:    {metrics.get('ic_mean', 0)*1e4:.1f}bp (t={metrics.get('ic_t_stat', 0):.2f})")
print(f'  IC显著:    5%{"✓" if metrics.get("ic_significant_5pct") else "✗"} 1%{"✓" if metrics.get("ic_significant_1pct") else "✗"}')
print(f"  Rank IC:   {metrics.get('rank_ic_mean', 0)*1e4:.1f}bp (t={metrics.get('rank_ic_t_stat', 0):.2f})")
print(f"  IR:        {metrics.get('ir', 0):.4f}")
print(f"  多空Sharpe: {ls_sh:.4f}")
print(f"  多空MDD:    {ls_md:.2%}")
print(f"  单调性:     {mono:.4f}")
print(f"  换手率:     {metrics.get('turnover_mean', 0):.2%}")
print(f"{'─'*60}")
print(f"  分层年化收益:")
for i, r in enumerate(metrics.get("group_returns_annualized", []), 1):
    r_str = f"{r:.2%}" if r is not None else "N/A"
    print(f"    G{i}: {r_str}")
print(f"{'═'*60}")
msg = "全部达标 ✓✓✓" if n_pass == 3 else ("两项达标 ✓✓" if n_pass == 2 else ("一项达标 ✓" if n_pass == 1 else "未达标 ✗"))
print(f"  达标准则: |IC|>0.015 | t>2 | |Sharpe|>0.5")
print(f"  {icon_big} |IC|{'>' if is_big else '<'}0.015  {icon_t} t{'>' if is_t else '<'}2  {icon_sh} |Sharpe|{'>' if is_sh else '<'}0.5  >> {msg}")
if avg_corr_td:
    print(f"  与turnover_decel相关系数: {avg_corr_td:.3f}")
print(f"\n  总耗时: {time.time()-t0:.1f}s")
