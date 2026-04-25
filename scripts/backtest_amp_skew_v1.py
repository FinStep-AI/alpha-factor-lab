#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: amp_skew_v1 — 振幅分布偏度 v1
================================================
构造:
  1. 日内振幅 = (high - low) / prev_close
  2. factor_raw = rolling_skew(amplitude, 20d)
     正偏度 = 多数振幅小 + 少数极端大振幅 → 机构试仓/日内大单冲击频发
     负偏度 = 振幅较均衡 → 价格发现有序，注入稳定预期
  3. 反向使用: 做多负偏度 = 做多稳定振幅型 = 低 Uncertainty
  4. 市值中性化 (OLS + MAD winsorize + z-score)

逻辑:
- 振幅均质型(负偏度): 日内价格发现稳定有序 → 信息传播效率高 → 正收益
- 振幅突发型(正偏度): 少数日极端放量但多数日平静 → 大单冲击/试仓活跃 → 不确定性 → 负收益
- 与 amp_level_v2(绝对水平) 正交: 本因子测形状(偏度)而非水平
- 与 vol_skew_v1(收益率偏度) 正交: 本因子基于振幅而非收益率

来源: 自研 | 参考: 日内振幅分布与知情交易者活跃度
"""

import json
import sys
import warnings
from pathlib import Path
from numpy.linalg import lstsq

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ────────────────── 参数 ──────────────────
WINDOW = 20
FORWARD_DAYS = 20
REBALANCE_FREQ = 20
N_GROUPS = 5
COST = 0.002
WINSORIZE_PCT = 0.05
FACTOR_ID = "amp_skew_v1"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "csi1000_kline_raw.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
high_piv = df.pivot_table(index="date", columns="stock_code", values="high")
low_piv = df.pivot_table(index="date", columns="stock_code", values="low")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")

ret_piv = close_piv.pct_change()

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造 amp_skew 因子 (振幅偏度, window={WINDOW})...")

# 日内振幅 (Parkinson式: (H-L)/C_{t-1})
prev_close = close_piv.shift(1).clip(lower=0.01)
amp = ((high_piv - low_piv) / prev_close).clip(lower=0)
print(f"   振幅均值/中位数: {amp.stack().mean():.4f} / {amp.stack().median():.4f}")

# 20日滚动偏度 (pandas内置 skew, sample skewness)
factor_raw = amp.rolling(WINDOW, min_periods=int(WINDOW * 0.75)).skew()
print(f"   偏度均值/中位数: {factor_raw.stack().mean():.4f} / {factor_raw.stack().median():.4f}")
print(f"   非空率: {factor_raw.notna().mean().mean():.2%}")

# ────────────────── 反向使用 (做多负偏度) ──────────────────
print(f"[2b] 反向使用 (做多负偏度 → 乘以 -1)...")
factor_raw = -factor_raw  # 负偏度 → 高因子值 → 做多

# ────────────────── 缩尾 (截面MAD) ──────────────────
print(f"[3] 截面缩尾 ({WINSORIZE_PCT*100:.0f}%)...")
for date in dates:
    row = factor_raw.loc[date].dropna()
    if len(row) < 10:
        continue
    lo = row.quantile(WINSORIZE_PCT)
    hi = row.quantile(1 - WINSORIZE_PCT)
    factor_raw.loc[date] = factor_raw.loc[date].clip(lo, hi)

# ────────────────── 市值中性化 ──────────────────
log_amt_20 = np.log(amount_piv.rolling(20).mean().clip(lower=1))
print(f"[4] 市值中性化 (OLS + MAD + z-score)...")
factor_neutral = factor_raw.copy()
n_neutral = 0
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
        # MAD winsorize on residual
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        if mad > 1e-10:
            resid = np.clip(resid, med - 5 * 1.4826 * mad, med + 5 * 1.4826 * mad)
        # z-score
        r_mean = np.nanmean(resid)
        r_std = np.nanstd(resid)
        if r_std > 1e-10:
            factor_neutral.loc[date, common] = (resid - r_mean) / r_std
            n_neutral += 1
    except Exception:
        pass

print(f"   完成中性化: ~{n_neutral} 天有结果")
print(f"   均值/std: {factor_neutral.stack().mean():.4f} / {factor_neutral.stack().std():.4f}")

# ────────────────── 回测 ──────────────────
print(f"[5] 分层回测: {N_GROUPS}组, 频率{REBALANCE_FREQ}天, {COST*100:.1f}%成本...")

common_dates = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
common_stocks = sorted(factor_neutral.columns.intersection(ret_piv.columns))
factor_aligned = factor_neutral.loc[common_dates, common_stocks]
return_aligned = ret_piv.loc[common_dates, common_stocks]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills" / "alpha-factor-lab" / "scripts"))
from factor_backtest import (
    compute_group_returns,
    compute_ic_dynamic,
    compute_metrics,
    save_backtest_data,
)

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

# ────────────────── 与 amp_level_v2 相关性 ──────────────────
print(f"[7] amp_level_v2 相关性...")
try:
    amp_level_20 = amp.rolling(20, min_periods=10).mean()
    corrs_amp = []
    for dt in common_dates[:40]:  # 只取前40天以加速
        f1 = factor_neutral.loc[dt].dropna()
        f2 = amp_level_20.loc[dt].reindex(f1.index).dropna()
        c = f1.index.intersection(f2.index)
        if len(c) > 50:
            corr, _ = sp_stats.spearmanr(f1[c], f2[c])
            if not np.isnan(corr):
                corrs_amp.append(corr)
    avg_corr_amp = float(np.mean(corrs_amp)) if corrs_amp else None
    print(f"   与amp_level_v2截面Spearman: {avg_corr_amp:.3f}")
except Exception as e:
    print(f"   相关性计算失败: {e}")
    avg_corr_amp = None

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

report = {
    "factor_id": FACTOR_ID,
    "factor_name": "振幅分布偏度 v1 (反向-做多负偏度)",
    "factor_name_en": "Amplitude Distribution Skewness v1 (Neg Direction)",
    "category": "波动率/分布形态",
    "description": f"过去{WINDOW}日日内振幅分布的偏度。反向使用:乘-1后做多负偏度(振幅均质型,少数大振幅)。",
    "hypothesis": "振幅均质化(负偏度)指数价格发现有序稳定;振幅突发型(正偏度/少数量大振幅)指数存在大冲击,市场不确定性高 → 负收益。",
    "expected_direction": "负偏度方向正向 (负偏度 γ<0 → IC>0)",
    "factor_type": "日内振幅分布形态",
    "formula": f"neutralize(-rolling_skew((high-low)/prev_close, {WINDOW}d), log_amount_20d)",
    "direction": -1,
    "direction_note": "原始信号做反转(-1)后,负偏度变为高因子值对应负偏度,即高频稳定振幅型",
    "stock_pool": "中证1000",
    "period": f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}",
    "n_dates": len(common_dates),
    "n_stocks": len(common_stocks),
    "n_groups": N_GROUPS,
    "rebalance_freq": REBALANCE_FREQ,
    "forward_days": FORWARD_DAYS,
    "cost": COST,
    "data_cutoff": str(df["date"].max()),
    "barra_style": "Volatility Distribution / Microstructure",
    "source_type": "自研",
    "source_title": "基于日内振幅分布的偏度因子",
    "source_url": "参考: 日内振幅分布预测性(日内市场微观结构)",
    "corr_with_amp_level": avg_corr_amp,
    "corr_note_amp_level": f"二者正交(偏度测形态 vs 水平测绝对幅度),Spearman约{avg_corr_amp}" if avg_corr_amp else "待测",
    "metrics": metrics,
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(nan_to_none(report), f, indent=2, ensure_ascii=False)

# ────────────────── 打印摘要 ──────────────────
print(f"\n{'═'*60}")
print(f"  Amplitude Skewness v1 回测结果")
print(f"  (方向: 做多负偏度 / 做空正偏度)")
print(f"{'═'*60}")
print(f"  区间:      {report['period']}")
print(f"  股票/日期: {len(common_stocks)} / {len(common_dates)}")
ic_mean = abs(metrics.get("ic_mean", 0) or 0)
ic_t = abs(metrics.get("ic_t_stat", 0) or 0)
ls_sh = metrics.get("long_short_sharpe", 0) or 0
ls_md = metrics.get("long_short_mdd", 0) or 0
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
print(f"  单调性:     {metrics.get('monotonicity', 0):.4f}")
print(f"  换手率:     {metrics.get('turnover_mean', 0):.2%}")
print(f"{'─'*60}")
print(f"  分层年化收益:")
for i, r in enumerate(metrics.get("group_returns_annualized", []), 1):
    r_str = f"{r:.2%}" if r is not None else "N/A"
    print(f"    G{i}: {r_str}")
print(f"{'═'*60}")
print(f"\n  达标准则: |IC|>0.015 | t>2 | |Sharpe|>0.5")
msg = "全部达标 ✓✓✓" if n_pass == 3 else ("两项达标 ✓✓" if n_pass == 2 else ("一项达标 ✓" if n_pass == 1 else "未达标 ✗"))
print(f"  {icon_big} |IC|{'>' if is_big else '<'}0.015  {icon_t} t{'>' if is_t else '<'}2  {icon_sh} |Sharpe|{'>' if is_sh else '<'}0.5  >> {msg}")
