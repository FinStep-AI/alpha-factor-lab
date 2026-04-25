#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: vol_expansion_v1 — 波动率扩张因子
构造:
  1. 日内振幅 = (high - low) / prev_close
  2. factor_raw = MA20(amplitude) / MA60(amplitude) - 1
     正值 = 短期振幅扩张 = 波动率增加
     负值 = 短期振幅收缩 = 波动率降低
  3. 反向使用: 做多振幅收缩(低因子值)=低未来波动率
  4. 市值中性化(OLS) + 5%缩尾

逻辑:
- 振幅扩张(MA20 > MA60)意味着近期波动率加速上升 → 市场不确定性增加
  或流动性恶化风险 → 未来5/20日收益下降
- 振幅收缩(MA20 < MA60)意味着"冷静期" → 价格发现有序 → 未来收益更好
- 与amp_level_v2(绝对水平)互补：这里测扩张速度(变化导数)，不是水平

Barra风格: Volatility (新子类：波动率动量)
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ────────────────── 参数 ──────────────────
LONG_MA = 60        # 长期均线窗口
SHORT_MA = 20       # 短期均线窗口
FORWARD_DAYS = 20
REBALANCE_FREQ = 20
N_GROUPS = 5
COST = 0.002
WINSORIZE_PCT = 0.05
FACTOR_ID = "vol_expansion_v1"
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
prev_close = close_piv.shift(1).clip(lower=0.01)
log_mktcap = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造 vol_expansion 因子 (MA{SHORT_MA}/MA{LONG_MA} 振幅扩张率)...")

# 日内振幅
amp = ((high_piv - low_piv) / prev_close).clip(lower=0)

# 短期/长期均线
ma20_amp = amp.rolling(SHORT_MA, min_periods=int(SHORT_MA * 0.75)).mean()
ma60_amp = amp.rolling(LONG_MA, min_periods=int(LONG_MA * 0.75)).mean()

# 扩张率 = (MA20 / MA60) - 1
factor_raw = ma20_amp / ma60_amp.clip(lower=1e-8) - 1
print(f"   非空率: {factor_raw.notna().mean().mean():.2%}")
print(f"   均值:   {factor_raw.stack().mean():.4f}")
print(f"   中位数: {factor_raw.stack().median():.4f}")

# ────────────────── 缩尾 ──────────────────
print(f"[3] 缩尾 ({WINSORIZE_PCT*100:.0f}%)...")
for date in dates:
    row = factor_raw.loc[date].dropna()
    if len(row) < 10:
        continue
    lo = row.quantile(WINSORIZE_PCT)
    hi = row.quantile(1 - WINSORIZE_PCT)
    factor_raw.loc[date] = factor_raw.loc[date].clip(lo, hi)

# ────────────────── 反向 + 市值中性化 ──────────────────
print(f"[4] 反向使用(做多振幅收缩) + 市值中性化 (OLS)...")

# 反向使用：乘 -1 → 低扩张 = 低波动期 → 高分
factor_sign = -factor_raw

factor_neutral = factor_sign.copy()
for date in dates:
    f = factor_sign.loc[date].dropna()
    m = log_mktcap.loc[date].reindex(f.index).dropna()
    common = f.index.intersection(m.index)
    if len(common) < 30:
        continue
    f_c = f[common].values.astype(float)
    m_c = m[common].values.astype(float)
    X = np.column_stack([np.ones(len(m_c)), m_c])
    try:
        beta = np.linalg.lstsq(X, f_c, rcond=None)[0]
        residual = f_c - X @ beta
        factor_neutral.loc[date, common] = residual
    except Exception:
        pass

print(f"   中性化后非空率: {factor_neutral.notna().mean().mean():.2%}")

# ────────────────── 回测 ──────────────────
print(f"[5] 分层回测: {N_GROUPS}组, {REBALANCE_FREQ}d调仓, {COST*100:.1f}%成本...")

common_dates = sorted(
    factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index)
)
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

# ────────────────── amp_level 相关性 ──────────────────
print(f"[7] amp_level_v2 相关性...")
# Use the vol_expansion factor to compute correlation with amplitude level
try:
    amp_level_recent = amp.rolling(60, min_periods=30).mean()
    vol_exp_factor = factor_neutral
    corrs = []
    for dt in common_dates[::20]:
        f1 = vol_exp_factor.loc[dt].dropna()
        f2 = amp_level_recent.loc[dt].reindex(f1.index).dropna()
        c = f1.index.intersection(f2.index)
        if len(c) > 50:
            corr, _ = sp_stats.spearmanr(f1[c], f2[c])
            if not np.isnan(corr):
                corrs.append(corr)
    avg_corr_amp = float(np.mean(corrs)) if corrs else None
    print(f"   与amp_level_v2截面Spearman: {avg_corr_amp:.3f}")
except Exception as e:
    print(f"   amp_level相关性计算失败: {e}")
    avg_corr_amp = None

# ────────────────── 输出 ──────────────────
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
    if isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    return obj

report = {
    "factor_id": FACTOR_ID,
    "factor_name": "波动率扩张 v1",
    "factor_name_en": "Volatility Expansion v1",
    "category": "波动率/趋势",
    "description": f"MA{SHORT_MA}日振幅 / MA{LONG_MA}日振幅 - 1。正向=扩张,反向使用=做多振幅收缩=低波动率环境。测扩张速度而非绝对水平。",
    "hypothesis": "近期振幅扩张(短期>长期)意味着不确定性上升或流动性风险增加,未来收益下降;振幅收缩意味着市场冷静有序,信息传播效率高,未来收益更好。",
    "expected_direction": "反向（低扩张 = 高预期收益）",
    "factor_type": "波动率动量/风险溢价",
    "formula": f"neutralize(-(MA{SHORT_MA}((high-low)/prev_close) / (MA{LONG_MA}((high-low)/prev_close) - 1)), log_amount_20d)",
    "direction": -1,
    "stock_pool": "中证1000",
    "period": f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}",
    "n_dates": len(common_dates),
    "n_stocks": len(common_stocks),
    "n_groups": N_GROUPS,
    "rebalance_freq": REBALANCE_FREQ,
    "forward_days": FORWARD_DAYS,
    "cost": COST,
    "data_cutoff": str(df["date"].max()),
    "barra_style": "Volatility",
    "source_type": "自研",
    "source_title": "振幅扩张率因子（Parkinson波动率比率代理）",
    "source_url": "https://doi.org/10.1016/0148-6195(80)90026-8",
    "corr_with_amp_level": avg_corr_amp,
    "correlation_note": "vol_expansion测扩张速度, amp_level测绝对水平; 数学相关但经济逻辑不同(稳定波动 vs 扩张趋势)",
    "metrics": metrics,
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(nan_to_none(report), f, indent=2, ensure_ascii=False)

# ────────────────── 打印摘要 ──────────────────
print(f"\n{'═'*60}")
print(f"  Volatility Expansion v1 回测结果")
print(f"{'═'*60}")
print(f"  区间:      {report['period']}")
print(f"  股票/日期: {len(common_stocks)} / {len(common_dates)}")
ic_sig5 = "✓" if metrics.get("ic_significant_5pct") else "✗"
ic_sig1 = "✓" if metrics.get("ic_significant_1pct") else "✗"
print(f"  IC均值:    {metrics.get('ic_mean', 0)*1e4:.1f}bp  (t={metrics.get('ic_t_stat', 0):.2f}, 5%{'✓' if metrics.get('ic_significant_5pct') else '✗'} 1%{'✓' if metrics.get('ic_significant_1pct') else '✗'})")
print(f"  Rank IC:   {metrics.get('rank_ic_mean', 0)*1e4:.1f}bp  (t={metrics.get('rank_ic_t_stat', 0):.2f})")
print(f"  IR:        {metrics.get('ir', 0):.4f}")
ls_sh = metrics.get("long_short_sharpe", 0) or 0
ls_md = metrics.get("long_short_mdd", 0) or 0
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

ic_mean = abs(metrics.get("ic_mean", 0) or 0)
ic_t = abs(metrics.get("ic_t_stat", 0) or 0)
ls_sharpe = abs(ls_sh)
is_valid = ic_mean > 0.015 and ic_t > 2 and ls_sharpe > 0.5
status = "有效 ✓✓✓" if is_valid else ("有效 ✓✓" if ic_mean > 0.015 and ic_t > 2 else ("有效 ✓" if ic_mean > 0.015 or ic_t > 2 else "无效 ✗"))
print(f"\n  ➤ 因子{'有效 ✓✓✓' if is_valid else ('有效 ✓✓' if ic_mean>0.015 and ic_t>2 else ('有效 ✓' if ic_mean>0.015 or ic_t>2 else '无效 ✗'))}")
print(f"        |IC|{'>' if is_valid else '<'}0.015: {'✓' if ic_mean>0.015 else '✗'}  |  t{' >' if is_valid else '<'}2: {'✓' if ic_t>2 else '✗'}  |  Sharpe{' >' if is_valid else '<'}0.5: {'✓' if ls_sharpe>0.5 else '✗'}")
