#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: low_clv_v1 — 日内低点收盘位置因子 v1
==========================================================
构造:
  1. low_position = (high - close) / (high - low + 1e-8)
     衡量收盘价在日内区间的位置: 1.0=收最低, 0.0=收最高
  2. factor_raw = MA(low_position, 20d)
     20日均值: 持续收低的股票 = 日内持续卖压 = 短期反弹蓄积
  3. 对数成交额中性化 (OLS + MAD winsorize + z-score)

逻辑:
- 高low_position = 持续收日内低点 = 日内主力持续卖出
  → 次日开盘/隔夜倾向于反弹(中国A股日内反转效应)
  → A股中证1000小盘股: 日内散户恐慌卖出 → 隔夜机构买入 → 反转收益
- 与vwap_dev(尾盘偏离)有区别: low_clv衡量收盘在日内区间的绝对位置
  (低点/高点), vwap_dev衡量收盘相对于全天量价均价
- 与shadow_pressure(上影线)有区别: shadow_pressure用上/下影线比率
  (多个价格点), 本因子只关心收盘位置

来源: 自研 | 内在逻辑=日内反转 + 丈夫因子概念
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
FORWARD_DAYS = 5          # 短周期 Irene (match overnight_intraday reversal)
REBALANCE_FREQ = 20
N_GROUPS = 5
COST = 0.003
WINSORIZE_PCT = 0.03
FACTOR_ID = "low_clv_v1"
FACTOR_NAME = "低点收盘位置"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "csi1000_kline_raw.csv"
RETURNS_PATH = Path(__file__).resolve().parent.parent / "data" / "csi1000_returns.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / FACTOR_ID
FACTOR_CSV = Path(__file__).resolve().parent.parent / "data" / f"factor_{FACTOR_ID}.csv"
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据: {DATA_PATH}")
print(f"[1a] Kline...")
df_k = pd.read_csv(DATA_PATH)
df_k["date"] = pd.to_datetime(df_k["date"])
df_k = df_k.sort_values(["stock_code", "date"]).reset_index(drop=True)
print(f"   {len(df_k)} 行, {df_k['stock_code'].nunique()} 股, {df_k['date'].nunique()} 日")

print(f"[1b] Returns...")
df_r = pd.read_csv(RETURNS_PATH)
df_r["date"] = pd.to_datetime(df_r["date"])
ret_piv = df_r.pivot_table(index="date", columns="stock_code", values="return")
ret_piv = ret_piv.sort_index()
print(f"   {ret_piv.shape} 收益率矩阵")

# ────────────────── pivot fields ──────────────────
print(f"[2] Pivot OHLCV...")
close_piv = df_k.pivot_table(index="date", columns="stock_code", values="close").sort_index()
high_piv  = df_k.pivot_table(index="date", columns="stock_code", values="high").sort_index()
low_piv   = df_k.pivot_table(index="date", columns="stock_code", values="low").sort_index()
amount_piv= df_k.pivot_table(index="date", columns="stock_code", values="amount").sort_index()

# ────────────────── 因子构造 ──────────────────
print(f"[3] 构造 low_clv 因子 (window={WINDOW})...")

# 分层位置: 1.0 = 收在最低, 0.0 = 收在最高
low_position_raw = (high_piv - close_piv) / (high_piv - low_piv + 1e-8)
low_position_raw = low_position_raw.clip(0, 1)
print(f"   low_position 均值: {low_position_raw.stack().mean():.3f}")

# 20日滚动均值
factor_raw = low_position_raw.rolling(WINDOW, min_periods=int(WINDOW*0.75)).mean()
print(f"   factor_raw 非空率: {factor_raw.notna().mean().mean():.2%}")
print(f"   factor_raw 均值/中位数: {factor_raw.stack().mean():.4f} / {factor_raw.stack().median():.4f}")

# ────────────────── 缩尾 (截面MAD 3σ) ──────────────────
print(f"[4] 截面缩尾 (3σ MAD)...")
dates_arr = factor_raw.dropna(how="all").index.tolist()
stocks_arr = factor_raw.columns.tolist()
for dt in dates_arr:
    row = factor_raw.loc[dt].dropna()
    if len(row) < 20:
        continue
    med = row.median()
    mad = np.median(np.abs(row - med))
    if mad < 1e-12:
        continue
    sigma = 1.4826 * mad
    factor_raw.loc[dt] = factor_raw.loc[dt].clip(med - 3*sigma, med + 3*sigma)

# ────────────────── 市值中性化 ──────────────────
# log_amount_20d 作为流通市值代理
log_amt_20 = np.log(amount_piv.rolling(20).mean().clip(lower=1))
print(f"[5] 市值中性化 (OLS + MAD + z-score)...")
factor_neutral = factor_raw.copy()
n_neutralized = 0
all_neu_vals = []

for dt in dates_arr:
    f_map = factor_raw.loc[dt].dropna()
    m_map = log_amt_20.loc[dt].reindex(f_map.index).dropna()
    common = f_map.index.intersection(m_map.index)
    if len(common) < 30:
        continue
    f_v = f_map[common].values.astype(float)
    m_v = m_map[common].values.astype(float)
    X = np.column_stack([np.ones(len(m_v)), m_v])
    try:
        beta, _, _, _ = lstsq(X, f_v, rcond=None)
        resid = f_v - X @ beta
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        if mad > 1e-10:
            resid = np.clip(resid, med - 5*1.4826*mad, med + 5*1.4826*mad)
        r_mean = np.nanmean(resid)
        r_std  = np.nanstd(resid)
        if r_std > 1e-10:
            std_resid = (resid - r_mean) / r_std
            factor_neutral.loc[dt, common] = std_resid
            all_neu_vals.append(std_resid)
            n_neutralized += 1
    except Exception as e:
        pass

print(f"   完成中性化: {n_neutralized} 天")
if all_neu_vals:
    all_arr = np.concatenate(all_neu_vals)
    print(f"   中性化后: mean={all_arr.mean():.4f} std={all_arr.std():.4f}")

# ────────────────── 输出因子CSV ──────────────────
out_records = []
for dt in dates_arr:
    for stk in stocks_arr:
        v = factor_neutral.loc[dt, stk]
        if not np.isnan(v):
            out_records.append({"date": dt.strftime("%Y-%m-%d"), "stock_code": stk, FACTOR_ID: float(v)})

out_df = pd.DataFrame(out_records)
out_df.to_csv(FACTOR_CSV, index=False, encoding="utf-8")
print(f"\n   因子值已保存 → {FACTOR_CSV}")
print(f"   有效行数: {len(out_df)}")

# ────────────────── IC / 分层回测 ──────────────────
print(f"\n[6] 回测: {N_GROUPS}组, rebalance={REBALANCE_FREQ}d, forward={FORWARD_DAYS}d, cost={COST}...")

# 公共日期/股票
common_dates = sorted(
    factor_neutral.dropna(how="all").index
    .intersection(ret_piv.dropna(how="all").index)
)
common_stocks = sorted(factor_neutral.columns.intersection(ret_piv.columns))
factor_mat = factor_neutral.loc[common_dates, common_stocks]
ret_mat    = ret_piv.loc[common_dates, common_stocks]
print(f"   公共: {len(common_dates)} 日 × {len(common_stocks)} 股")

# 加入回测路径到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills/alpha-factor-lab/scripts"))
from factor_backtest import (
    compute_group_returns,
    compute_ic_dynamic,
    compute_metrics,
    save_backtest_data,
)

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

ic_series  = compute_ic_dynamic(factor_mat, ret_mat, FORWARD_DAYS, "pearson")
rank_ic_series = compute_ic_dynamic(factor_mat, ret_mat, FORWARD_DAYS, "spearman")

group_returns, turnovers, holdings_info = compute_group_returns(
    factor_mat, ret_mat, N_GROUPS, REBALANCE_FREQ, COST
)
metrics = compute_metrics(
    group_returns, ic_series, rank_ic_series, turnovers, N_GROUPS,
    holdings_info=holdings_info
)

# ────────────────── IC相关性 (与已有因子) ──────────────────
print(f"\n[7] 计算 IC 时间序列相关性...")
corrs = {}
existing_factors = [
    "amihud_illiq_v2", "shadow_pressure_v1", "overnight_momentum_v1",
    "gap_momentum_v1", "tail_risk_cvar_v1", "neg_day_freq_v1",
    "turnover_level_v1", "tae_v1", "amp_level_v2", "ma_disp_v1",
    "vol_cv_neg_v1", "turnover_decel_v1", "pv_corr_v1", "vwap_dev_v1",
    "vssignal_v1", "price_sync_v3_fwd5",
]
for fid in existing_factors:
    try:
        fpath = Path(__file__).resolve().parent.parent / "output" / fid / "ic_series.json"
        if not fpath.exists():
            fpath = Path(__file__).resolve().parent.parent / "output" / fid / "ic_series.csv"
        if not fpath.exists():
            corrs[fid] = None
            continue
        if fpath.suffix == ".json":
            import json
            ic_d = json.loads(fpath.read_text())
            ic_s = pd.Series({pd.to_datetime(k): v for k, v in ic_d.items() if v is not None}).sort_index()
        else:
            ic_df = pd.read_csv(fpath)
            ic_s = pd.Series(
                {pd.to_datetime(r.iloc[0]): r.iloc[1] for _, r in ic_df.iterrows() if not np.isnan(r.iloc[1])}
            ).sort_index()
        common_idx = ic_series.dropna().index.intersection(ic_s.dropna().index)
        if len(common_idx) > 10:
            corr_val = float(sp_stats.spearmanr(ic_series.loc[common_idx], ic_s.loc[common_idx]).corr)
            corrs[fid] = round(corr_val, 4)
        else:
            corrs[fid] = None
    except Exception as e:
        corrs[fid] = None

print("   IC相关系数:")
for k, v in corrs.items():
    print(f"     {k}: {v}")

max_abs_corr = max(abs(v) for v in corrs.values() if v is not None) if any(v is not None for v in corrs.values()) else 0
print(f"   最大|IC相关|: {max_abs_corr:.3f} {'⚠️ 冗余!' if max_abs_corr > 0.7 else '✓ 独立'}")

# ────────────────── 保存结果 ──────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_backtest_data(group_returns, ic_series, rank_ic_series, str(OUTPUT_DIR))

ic_mean   = metrics.get("ic_mean", 0) or 0
ic_t      = metrics.get("ic_t_stat", 0) or 0
ls_sh     = metrics.get("long_short_sharpe", 0) or 0
mono      = metrics.get("monotonicity", 0) or 0
g5_s      = metrics.get("group_sharpe", [None]*5)[4]
g5_ann    = metrics.get("group_returns_annualized", [None]*5)[4]

barra_style = "Reversal"
barra_explanation = "Reversal: 日内低点反转——收盘接近日内最低意味着日内卖压释放完毕，次日开盘倾向于回调（隔夜均值回复）。"

report = {
    "factor_id":   FACTOR_ID,
    "factor_name": FACTOR_NAME,
    "factor_name_en": "Low Close Location v1",
    "category":    "反转/隔夜效应",
    "description": "20日平均 (最高价-收盘价)/(最高价-最低价)。高值=持续收在日内低点=日内卖压持续释放。正向使用(做多高值): 日内卖压耗尽→隔夜/次日开盘倾向于反弹(日内反转效应)。成交额OLS中性化+MAD Winsorize+z-score。",
    "hypothesis": "收盘持续触及日内低点的股票,日内卖压(散户恐慌/机构撤退)释放较重,次日开盘倾向于均回复(知情信息驱动后的反转)。A股中证1000小盘股日内反转效应强,日内承压→隔夜受益(过度抛售→次日修复)。",
    "expected_direction": "正向 (高low_clv=收低点=高预期收益)",
    "factor_type": "日内反转/隔夜效应",
    "formula": "neutralize(MA20((high-close)/(high-low+1e-8)), log_amount_20d)",
    "direction": 1,
    "stock_pool": "中证1000",
    "period": f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}",
    "n_dates": len(common_dates),
    "n_stocks": len(common_stocks),
    "n_groups": N_GROUPS,
    "rebalance_freq": REBALANCE_FREQ,
    "forward_days":  FORWARD_DAYS,
    "cost": COST,
    "data_cutoff": str(df_k["date"].max().date()),
    "winsorize_method": "MAD 3σ",
    "neutralization": "log_amount_20d OLS",
    "barra_style": barra_style,
    "barra_explanation": barra_explanation,
    "source_type": "自研(日内反转论文启发)",
    "source_title": "日内反转效应 + CLV因子本土化",
    "source_url": "https://zhuanlan.zhihu.com/p/644380633",
    "correlations": corrs,
    "max_ic_correlation": max_abs_corr,
    "metrics": metrics,
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(nan_to_none(report), f, indent=2, ensure_ascii=False)

# ────────────────── 打印摘要 ──────────────────
print(f"\n{'═'*60}")
print(f"  LOW CLV v1 因子回测结果")
print(f"  (方向: 做多日内低点→做空日内高点)")
print(f"{'═'*60}")
print(f"  区间:        {report['period']}")
print(f"  股票/日期:   {report['n_stocks']} / {report['n_dates']}")
print(f"  IC均值:      {ic_mean*100:+.3f}% (t={ic_t:.2f})")
sig5 = metrics.get("ic_significant_5pct")
sig1 = metrics.get("ic_significant_1pct")
print(f"  IC显著:      {'✓5%' if sig5 else '✗5%'}  {'✓1%' if sig1 else '✗1%'}")
print(f"  Rank IC:     {metrics.get('rank_ic_mean', 0)*100:+.3f}% (t={metrics.get('rank_ic_t_stat',0):.2f})")
print(f"  IR:          {metrics.get('ir',0):.4f}")
print(f"  多空Sharpe:  {ls_sh:.4f}")
print(f"  多空MDD:     {metrics.get('long_short_mdd',0):.2%}")
print(f"  单调性:      {mono:.4f}")
print(f"  换手率:      {metrics.get('turnover_mean',0):.2%}")
print(f"  G5 Sharpe:   {g5_s:.4f}" if g5_s is not None else f"  G5 Sharpe:   N/A")
print(f"  G5年化:      {g5_ann:.2%}" if g5_ann is not None else f"  G5年化:      N/A")
print(f"  最大IC相关:  {max_abs_corr:.3f}")
print(f"{'─'*60}")
print(f"  分层年化收益:")
for i, r in enumerate(metrics.get("group_returns_annualized", []), 1):
    r_str = f"{r:.2%}" if r is not None else "N/A"
    print(f"    G{i}: {r_str}")
print(f"{'═'*60}")
print(f"\n  达标准则: IC>0.02, t>2.0, G5 Sharpe>0.8, 单调性>0.8")
pass_ic    = ic_mean > 0.02
pass_t     = ic_t > 2.0
pass_sh    = (g5_s or 0) > 0.8
pass_mono  = mono > 0.8
n_pass  = sum([pass_ic, pass_t, pass_sh, pass_mono])
msg = "全部达标 ✓✓✓✓" if n_pass == 4 else ("3项达标 ✓✓✓" if n_pass == 3 else ("2项达标 ✓✓" if n_pass == 2 else "✓" if n_pass == 1 else "未达标 ✗"))
print(f"  {'✓' if pass_ic else '✗'} IC>0.02  {'✓' if pass_t else '✗'} t>2  {'✓' if pass_sh else '✗'} G5 Sharpe>0.8  {'✓' if pass_mono else '✗'} Mono>0.8  → {msg}")
