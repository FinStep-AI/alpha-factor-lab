#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: return_entropy_v1 — 收益熵因子
==========================================

方向: 信息论 / 微观结构

构造:
  1. 将过去20日的日收益率离散化为5个bin (quintile-based)
  2. 计算每只股票收益率分布的Shannon信息熵: H = -Σ p(i) * log(p(i))
  3. 成交额OLS中性化
  4. 5%缩尾
  5. 双向探索

逻辑:
  收益熵衡量价格运动的随机性/可预测性:
  
  高熵 (收益分布均匀/随机):
  → 市场对该股票定价效率高, 无明显规律可利用
  → 信息充分反映在价格中, 无超额收益(弱有效市场)
  
  低熵 (收益集中在特定区间/有规律):
  → 存在趋势(持续涨或跌) 或 均值回复(围绕某值波动)
  → 信息传播不均匀, 价格发现不完整, 存在可利用的模式
  → 可能是知情交易者集中交易导致的价格模式
  
  在A股中证1000小盘股上:
  → 低熵股票(有明确模式) 可能有动量延续或反转效应
  → 高熵股票(完全随机) 可能无alpha
  → 方向待数据确认

理论:
  - Bentes & Menezes (2012) "Entropy: A new measure of stock market volatility?" JPhysA
  - Dionisio et al. (2006) "Entropy-based independence test" Nonlinear Dynamics
  - 信息论视角: Shannon entropy衡量不确定性, 低熵=有序=有信息=可利用
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ────────────────── 参数 ──────────────────
WINDOW = 20
N_BINS = 5  # 离散化bin数
FORWARD_DAYS = 5
REBALANCE_FREQ = 5
N_GROUPS = 5
COST = 0.003
WINSORIZE_PCT = 0.05
DATA_CUTOFF = "2026-03-13"
FACTOR_ID = "return_entropy_v1"

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "csi1000_kline_raw.csv"
SCRIPTS_DIR = BASE_DIR / "skills" / "alpha-factor-lab" / "scripts"
OUTPUT_DIR = BASE_DIR / "output" / FACTOR_ID
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

# ────────────────── 数据加载 ──────────────────
print(f"[1] 加载数据 (cutoff: {DATA_CUTOFF})...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] <= DATA_CUTOFF]
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

close_piv = df.pivot_table(index="date", columns="stock_code", values="close")
open_piv = df.pivot_table(index="date", columns="stock_code", values="open")
amount_piv = df.pivot_table(index="date", columns="stock_code", values="amount")

ret_piv = close_piv.pct_change()
log_amt = np.log(amount_piv.rolling(20).mean().clip(lower=1))

dates = close_piv.index.tolist()
stocks = close_piv.columns.tolist()
n_dates = len(dates)
n_stocks = len(stocks)
ret_vals = ret_piv.values
print(f"   {n_dates} 日, {n_stocks} 股")

# ────────────────── 因子构造 ──────────────────
print(f"[2] 构造收益熵因子 (window={WINDOW}, bins={N_BINS})...")

def compute_entropy(returns, n_bins=5):
    """计算一组收益率的Shannon信息熵
    
    使用截面等频分bin(基于当天所有股票的收益分布),
    而非绝对值分bin, 以消除市场整体波动影响。
    
    然后在时间维度上, 使用固定边界(基于窗口内数据的分位数)
    """
    valid = returns[~np.isnan(returns)]
    n = len(valid)
    if n < 10:
        return np.nan
    
    # 使用分位数确定bin边界(适应性分bin)
    edges = np.percentile(valid, np.linspace(0, 100, n_bins + 1))
    edges[0] -= 1e-10  # 确保最小值被包含
    edges[-1] += 1e-10
    
    # 确保边界严格递增
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-10
    
    counts = np.histogram(valid, bins=edges)[0]
    probs = counts / n
    probs = probs[probs > 0]  # 过滤零概率
    
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

# 方法1: 个股时序分bin (每只股票过去20日自己的收益分布)
entropy_mat = np.full((n_dates, n_stocks), np.nan)

for i in range(WINDOW, n_dates):
    for j in range(n_stocks):
        window_rets = ret_vals[i - WINDOW:i, j]
        entropy_mat[i, j] = compute_entropy(window_rets, N_BINS)

factor_raw = pd.DataFrame(entropy_mat, index=dates, columns=stocks)

print(f"   非空率: {factor_raw.notna().mean().mean():.2%}")
print(f"   均值: {factor_raw.stack().mean():.4f}, std: {factor_raw.stack().std():.4f}")
print(f"   min: {factor_raw.stack().min():.4f}, max: {factor_raw.stack().max():.4f}")

# ────────────────── 缩尾 ──────────────────
print(f"[3] 缩尾 ({WINSORIZE_PCT*100:.0f}%)...")
for date in dates:
    row = factor_raw.loc[date].dropna()
    if len(row) < 10:
        continue
    lo = row.quantile(WINSORIZE_PCT)
    hi = row.quantile(1 - WINSORIZE_PCT)
    factor_raw.loc[date] = factor_raw.loc[date].clip(lo, hi)

# ────────────────── 中性化 ──────────────────
print(f"[4] 成交额中性化 (OLS)...")
factor_neutral = factor_raw.copy()
for date in dates:
    f = factor_raw.loc[date].dropna()
    m = log_amt.loc[date].reindex(f.index).dropna()
    common = f.index.intersection(m.index)
    if len(common) < 30:
        continue
    f_c = f[common].values
    m_c = m[common].values
    X = np.column_stack([np.ones(len(m_c)), m_c])
    try:
        beta = np.linalg.lstsq(X, f_c, rcond=None)[0]
        factor_neutral.loc[date, common] = f_c - X @ beta
    except:
        pass

# ────────────────── 回测 ──────────────────
print(f"[5] 回测引擎加载...")

sys.path.insert(0, str(SCRIPTS_DIR))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data
)

common_dates = sorted(factor_neutral.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
common_stocks = sorted(factor_neutral.columns.intersection(ret_piv.columns))
fa = factor_neutral.loc[common_dates, common_stocks]
ra = ret_piv.loc[common_dates, common_stocks]

# ────────────────── 方向探索 ──────────────────
print(f"[6] 方向探索...")

# 正向: 做多高熵 (高随机性/高效定价 → ?)
ic_pos = compute_ic_dynamic(fa, ra, FORWARD_DAYS, "pearson")
gr_pos, _, _ = compute_group_returns(fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_pos = compute_metrics(gr_pos, ic_pos, ic_pos, [], N_GROUPS)

# 反向: 做多低熵 (有模式/低效定价 → ?)
ic_neg = compute_ic_dynamic(-fa, ra, FORWARD_DAYS, "pearson")
gr_neg, _, _ = compute_group_returns(-fa, ra, N_GROUPS, REBALANCE_FREQ, COST)
m_neg = compute_metrics(gr_neg, ic_neg, ic_neg, [], N_GROUPS)

pos_sh = m_pos.get("long_short_sharpe", 0) or 0
neg_sh = m_neg.get("long_short_sharpe", 0) or 0
pos_ic = m_pos.get("ic_mean", 0) or 0
neg_ic = m_neg.get("ic_mean", 0) or 0
pos_t = m_pos.get("ic_t_stat", 0) or 0
neg_t = m_neg.get("ic_t_stat", 0) or 0
pos_mono = m_pos.get("monotonicity", 0) or 0
neg_mono = m_neg.get("monotonicity", 0) or 0

print(f"   正向 (高熵=高收益): IC={pos_ic:.4f}, t={pos_t:.2f}, Sharpe={pos_sh:.4f}, Mono={pos_mono:.2f}")
print(f"   反向 (低熵=高收益): IC={neg_ic:.4f}, t={neg_t:.2f}, Sharpe={neg_sh:.4f}, Mono={neg_mono:.2f}")

# 选择更好的方向
pos_score = abs(pos_ic) * max(pos_sh, 0) * max(pos_mono, 0.1)
neg_score = abs(neg_ic) * max(neg_sh, 0) * max(neg_mono, 0.1)

if neg_score > pos_score:
    print(f"   → 使用反向 (做多低熵)")
    direction = -1
    fa_final = -fa
    direction_desc = "反向（低熵=有模式=高预期收益）"
else:
    print(f"   → 使用正向 (做多高熵)")
    direction = 1
    fa_final = fa
    direction_desc = "正向（高熵=高随机性=高预期收益）"

# ────────────────── 最终回测 ──────────────────
print(f"[7] 最终回测 (方向={direction})...")

ic_series = compute_ic_dynamic(fa_final, ra, FORWARD_DAYS, "pearson")
rank_ic_series = compute_ic_dynamic(fa_final, ra, FORWARD_DAYS, "spearman")

group_returns, turnovers, holdings_info = compute_group_returns(
    fa_final, ra, N_GROUPS, REBALANCE_FREQ, COST
)
metrics = compute_metrics(
    group_returns, ic_series, rank_ic_series, turnovers, N_GROUPS,
    holdings_info=holdings_info
)

# ────────────────── 前瞻期敏感性 ──────────────────
print(f"[8] 前瞻期敏感性...")
for fwd in [10, 20]:
    rebal = fwd
    ic_t = compute_ic_dynamic(fa_final, ra, fwd, "pearson")
    gr_t, _, _ = compute_group_returns(fa_final, ra, N_GROUPS, rebal, COST)
    m_t = compute_metrics(gr_t, ic_t, ic_t, [], N_GROUPS)
    print(f"   Forward={fwd}d: IC={m_t.get('ic_mean',0):.4f}, t={m_t.get('ic_t_stat',0):.2f}, "
          f"Sharpe={m_t.get('long_short_sharpe',0):.4f}, Mono={m_t.get('monotonicity',0):.2f}")

# ────────────────── 窗口敏感性 ──────────────────
print(f"[9] 窗口敏感性...")
for test_window in [10, 40, 60]:
    ent_test = np.full((n_dates, n_stocks), np.nan)
    for i in range(test_window, n_dates):
        for j in range(n_stocks):
            wr = ret_vals[i - test_window:i, j]
            ent_test[i, j] = compute_entropy(wr, N_BINS)
    
    fraw_test = pd.DataFrame(ent_test, index=dates, columns=stocks)
    
    # 缩尾
    for d in dates:
        row = fraw_test.loc[d].dropna()
        if len(row) < 10:
            continue
        lo_t = row.quantile(0.05)
        hi_t = row.quantile(0.95)
        fraw_test.loc[d] = fraw_test.loc[d].clip(lo_t, hi_t)
    
    # 中性化
    fn_test = fraw_test.copy()
    for d in dates:
        ft = fraw_test.loc[d].dropna()
        mt = log_amt.loc[d].reindex(ft.index).dropna()
        cm = ft.index.intersection(mt.index)
        if len(cm) < 30:
            continue
        fc = ft[cm].values
        mc = mt[cm].values
        Xt = np.column_stack([np.ones(len(mc)), mc])
        try:
            beta = np.linalg.lstsq(Xt, fc, rcond=None)[0]
            fn_test.loc[d, cm] = fc - Xt @ beta
        except:
            pass
    
    cd = sorted(fn_test.dropna(how="all").index.intersection(ra.index))
    cs = sorted(fn_test.columns.intersection(ra.columns))
    if len(cd) < 50:
        continue
    
    fat = (direction * fn_test).loc[cd, cs]
    ic_t2 = compute_ic_dynamic(fat, ra.loc[cd, cs], FORWARD_DAYS, "pearson")
    gr_t2, _, _ = compute_group_returns(fat, ra.loc[cd, cs], N_GROUPS, REBALANCE_FREQ, COST)
    m_t2 = compute_metrics(gr_t2, ic_t2, ic_t2, [], N_GROUPS)
    print(f"   Window={test_window}d: IC={m_t2.get('ic_mean',0):.4f}, t={m_t2.get('ic_t_stat',0):.2f}, "
          f"Sharpe={m_t2.get('long_short_sharpe',0):.4f}, Mono={m_t2.get('monotonicity',0):.2f}")

# ────────────────── bin数量敏感性 ──────────────────
print(f"[10] Bin数量敏感性...")
for test_bins in [3, 4, 7, 10]:
    ent_bins = np.full((n_dates, n_stocks), np.nan)
    for i in range(WINDOW, n_dates):
        for j in range(n_stocks):
            wr = ret_vals[i - WINDOW:i, j]
            ent_bins[i, j] = compute_entropy(wr, test_bins)
    
    fraw_bins = pd.DataFrame(ent_bins, index=dates, columns=stocks)
    for d in dates:
        row = fraw_bins.loc[d].dropna()
        if len(row) < 10:
            continue
        lo_t = row.quantile(0.05)
        hi_t = row.quantile(0.95)
        fraw_bins.loc[d] = fraw_bins.loc[d].clip(lo_t, hi_t)
    
    fn_bins = fraw_bins.copy()
    for d in dates:
        ft = fraw_bins.loc[d].dropna()
        mt = log_amt.loc[d].reindex(ft.index).dropna()
        cm = ft.index.intersection(mt.index)
        if len(cm) < 30:
            continue
        fc = ft[cm].values
        mc = mt[cm].values
        Xb = np.column_stack([np.ones(len(mc)), mc])
        try:
            beta = np.linalg.lstsq(Xb, fc, rcond=None)[0]
            fn_bins.loc[d, cm] = fc - Xb @ beta
        except:
            pass
    
    cd = sorted(fn_bins.dropna(how="all").index.intersection(ra.index))
    cs = sorted(fn_bins.columns.intersection(ra.columns))
    if len(cd) < 50:
        continue
    
    fab = (direction * fn_bins).loc[cd, cs]
    ic_b = compute_ic_dynamic(fab, ra.loc[cd, cs], FORWARD_DAYS, "pearson")
    gr_b, _, _ = compute_group_returns(fab, ra.loc[cd, cs], N_GROUPS, REBALANCE_FREQ, COST)
    m_b = compute_metrics(gr_b, ic_b, ic_b, [], N_GROUPS)
    print(f"   Bins={test_bins}: IC={m_b.get('ic_mean',0):.4f}, t={m_b.get('ic_t_stat',0):.2f}, "
          f"Sharpe={m_b.get('long_short_sharpe',0):.4f}, Mono={m_b.get('monotonicity',0):.2f}")

# ────────────────── 相关性 ──────────────────
print(f"[11] 与现有因子相关性...")

high_piv = df.pivot_table(index="date", columns="stock_code", values="high")
low_piv = df.pivot_table(index="date", columns="stock_code", values="low")
turnover_piv = df.pivot_table(index="date", columns="stock_code", values="turnover")
amplitude_piv = df.pivot_table(index="date", columns="stock_code", values="amplitude")

# 已有因子重构
amihud_raw = (ret_piv.abs() / (amount_piv / 1e8).clip(lower=1e-8))
amihud_factor = np.log(amihud_raw.rolling(20, min_periods=10).mean().clip(lower=1e-12))

upper_sr = (high_piv - np.maximum(close_piv, open_piv)) / (high_piv - low_piv).clip(lower=1e-8)
lower_sr = (np.minimum(close_piv, open_piv) - low_piv) / (high_piv - low_piv).clip(lower=1e-8)
shadow = (upper_sr - lower_sr).rolling(20, min_periods=10).mean()

oret = open_piv / close_piv.shift(1) - 1
iret = close_piv / open_piv - 1
overnight_mom = oret.rolling(20, min_periods=10).sum() - iret.rolling(20, min_periods=10).sum()

cvar_mat = np.full((n_dates, n_stocks), np.nan)
for i in range(10, n_dates):
    w = ret_vals[i-10:i, :]
    s = np.sort(w, axis=0)
    bot2 = np.nanmean(s[:2, :], axis=0)
    vc = np.sum(~np.isnan(w), axis=0)
    bot2[vc < 5] = np.nan
    cvar_mat[i, :] = -bot2
cvar_df = pd.DataFrame(cvar_mat, index=dates, columns=stocks)

neg_freq = (ret_piv <= -0.03).astype(float).rolling(10, min_periods=5).mean()
turnover_level = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8))
tae_raw = np.log(turnover_piv.rolling(20, min_periods=10).mean().clip(lower=1e-8) /
                  (amplitude_piv.rolling(20, min_periods=10).mean().clip(lower=0.01)))

amp_raw = ((high_piv - low_piv) / close_piv.shift(1).clip(lower=0.01))
amp_level = np.log(amp_raw.rolling(60, min_periods=30).mean().clip(lower=1e-8))

ma5 = close_piv.rolling(5).mean() / close_piv
ma10 = close_piv.rolling(10).mean() / close_piv
ma20 = close_piv.rolling(20).mean() / close_piv
ma40 = close_piv.rolling(40).mean() / close_piv
ma60 = close_piv.rolling(60).mean() / close_piv
ma120 = close_piv.rolling(120).mean() / close_piv
ma_disp = pd.DataFrame(np.nanstd(np.stack([ma5.values, ma10.values, ma20.values,
                                             ma40.values, ma60.values, ma120.values]), axis=0),
                         index=dates, columns=stocks)

vol_cv = (amount_piv.rolling(20, min_periods=10).std() /
          amount_piv.rolling(20, min_periods=10).mean().clip(lower=1))
vol_cv_neg = -vol_cv

to_ma5 = turnover_piv.rolling(5, min_periods=3).mean()
to_ma20 = turnover_piv.rolling(20, min_periods=10).mean()
turnover_decel = -np.log((to_ma5 / to_ma20.clip(lower=1e-8)).clip(lower=1e-8))

correlations = {}
for name, other in [
    ('amihud_illiq_v2', amihud_factor),
    ('shadow_pressure_v1', shadow),
    ('overnight_momentum_v1', overnight_mom),
    ('tail_risk_cvar_v1', cvar_df),
    ('neg_day_freq_v1', neg_freq),
    ('turnover_level_v1', turnover_level),
    ('tae_v1', tae_raw),
    ('ma_disp_v1', ma_disp),
    ('amp_level_v2', amp_level),
    ('vol_cv_neg_v1', vol_cv_neg),
    ('turnover_decel_v1', turnover_decel),
]:
    try:
        f1 = fa_final.stack()
        f2 = (direction * other).reindex_like(factor_neutral).loc[common_dates, common_stocks].stack()
        valid = pd.DataFrame({'a': f1, 'b': f2}).dropna()
        if len(valid) > 100:
            correlations[name] = round(valid['a'].corr(valid['b']), 4)
        else:
            correlations[name] = None
    except:
        correlations[name] = None

print("   相关性:")
for k, v in correlations.items():
    print(f"   {k}: {v}")

# ────────────────── 评估 ──────────────────
print(f"\n[12] 评估结果...")

ic_mean = metrics.get("ic_mean", 0) or 0
ic_t = metrics.get("ic_t_stat", 0) or 0
sharpe = metrics.get("long_short_sharpe", 0) or 0
mono = metrics.get("monotonicity", 0) or 0
mdd = metrics.get("long_short_mdd", 0) or 0

print(f"   IC均值: {ic_mean:.4f}")
print(f"   IC t值: {ic_t:.2f}")
print(f"   多空Sharpe: {sharpe:.4f}")
print(f"   单调性: {mono:.2f}")
print(f"   多空MDD: {mdd:.4f}")
print(f"   方向: {direction_desc}")

# 入库标准: |IC| > 0.015, t > 2, Sharpe > 0.5
passed = abs(ic_mean) > 0.015 and abs(ic_t) > 2 and sharpe > 0.5

if passed:
    print(f"\n✅ 因子达标！保存结果...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_backtest_data(
        group_returns, ic_series, rank_ic_series, turnovers,
        metrics, FACTOR_ID, str(OUTPUT_DIR), holdings_info=holdings_info
    )
    print(f"   已保存到 {OUTPUT_DIR}/")
else:
    print(f"\n❌ 因子未达标")
    print(f"   |IC|>0.015: {'✅' if abs(ic_mean)>0.015 else '❌'} ({abs(ic_mean):.4f})")
    print(f"   t>2: {'✅' if abs(ic_t)>2 else '❌'} ({abs(ic_t):.2f})")
    print(f"   Sharpe>0.5: {'✅' if sharpe>0.5 else '❌'} ({sharpe:.4f})")

# 输出完整结果供后续处理
result = {
    "factor_id": FACTOR_ID,
    "passed": passed,
    "direction": direction,
    "direction_desc": direction_desc,
    "metrics": metrics,
    "correlations": correlations,
}
print(f"\n[RESULT_JSON]{json.dumps(result, default=str)}")
