#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: turnover_vol_resid_fwd20_v1 — 换手-成交额回归残差 v1 (20日前瞻)
========================================================================

方向: 流动性异常 / 资金效率

构造:
  1. 20 日滚动窗口内，每日按 log(换手率) ~ log(成交额) 做 OLS 截面回归
  2. 取每个截面日的回归残差(r = turnover - fitted_by_amount)
  3. 残差截面 Z-score → 每日常截面 5% 缩尾 → 成交额 OLS 中性化
  4. 最终截面 Z-score 作为当日期因子

正向定义:
  正残差 = 相同成交额下换手率更高 → 流动性需求异常旺盛 → 散户主导偏好、
  信息轮动中注意无效传播 → 后续下跌 / 较保守配置
  → 反向(低残差/负残差 = 机构持仓效率高 + 涨回调更好)
  本脚本保留因子本正方向；需要反向就注册时用 "-1×(factor值)" 策略。

与现有因子区别:
  - amihud_illiq / idlevel : 价-量综合流动性;
    本因子拆开换手与成交额比例残差 — 信息效率类型;
  - turnover_level / turnover_decel: 纯换手时序;
    本因子加入成交额作为捕捉资金利用效率维度;
  - vs: informed_flow_v1 只衡量 amount×|ret| 方向。
    换手 vs 成交额 OL S残差独立维度方法;

理论:
  - Chordia, Roll & Subrahmanyam (2001): 换手率中不可由量比解释成分含信息;
  - Llorente et al. (2002): 成交量-收益动态关系;
  - 换手率正向"噪声"时成交额走的慢,这提供有有时也能成为"偏见"动量因子。

回测参数:
  WINDOW=20, FORWARD_DAYS=20, REBALANCE=20, N=5, COST=0.003
  Data cutoff: 2026-05-01
"""

import json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ────────────────── 任务参数 ──────────────────
WINDOW         = 20
FORWARD_DAYS   = 20
REBALANCE_FREQ = 20          # 与 amihud_illiq_v2、turnover_level_v1 一致,中频
N_GROUPS       = 5
COST           = 0.003
WINSORIZE_PCT  = 0.05
DATA_CUTOFF    = "2026-05-01"
FACTOR_ID      = "turnover_vol_resid_fwd20_v1"

BASE_DIR     = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH    = BASE_DIR / "data" / "csi1000_kline_raw.csv"
SCRIPTS_DIR  = BASE_DIR / "skills" / "alpha-factor-lab" / "scripts"
OUTPUT_DIR   = BASE_DIR / "output" / FACTOR_ID
REPORT_PATH  = OUTPUT_DIR / "backtest_report.json"

sys.path.insert(0, str(SCRIPTS_DIR))
from factor_backtest import (
    compute_group_returns, compute_ic_dynamic,
    compute_metrics, save_backtest_data
)

t0 = time.time()

# ────────────────── 1. 数据加载 ──────────────────
print(f"[1] 加载数据 (cutoff={DATA_CUTOFF})...")
df = pd.read_csv(DATA_PATH, encoding="utf-8")
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] <= pd.Timestamp(DATA_CUTOFF)].copy()
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

tp = df.pivot_table(index="date", columns="stock_code", values="turnover")
ap = df.pivot_table(index="date", columns="stock_code", values="close")

dates  = tp.index.tolist()
stocks = tp.columns.tolist()
ret = ap.pct_change()
print(f"   {len(dates)} 日, {len(stocks)} 股  ({time.time()-t0:.1f}s)")

# ────────────────── 2. 因子构造 ──────────────────
print(f"[2] 换手-成交额 20日滚动 OLS 残差 (window={WINDOW})...")

tp_f   = tp.values.astype(np.float64)
ap_f   = ap.values.astype(np.float64)

# 日成交额代理 ≈ close × turnover（无量纲，与换手率同阶）
daily_amt = ap_f * tp_f
log_turn  = np.log(tp_f.clip(min=1e-8))
log_amnt  = np.log(daily_amt.clip(min=1e-8))

n_dates, n_stocks = tp_f.shape
resid_mat = np.full((n_dates, n_stocks), np.nan)
log_amt_20mean = np.log(
    pd.DataFrame(daily_amt, index=dates, columns=stocks)
    .rolling(WINDOW, min_periods=int(WINDOW*0.5)).mean().clip(lower=1e-8).values
)

for i in range(WINDOW, n_dates):
    x_win = log_amnt[i - WINDOW:i, :]
    y_win = log_turn[i - WINDOW:i, :]
    for j in range(n_stocks):
        col_x = x_win[:, j]
        col_y = y_win[:, j]
        valid = ~(np.isnan(col_x) | np.isnan(col_y))
        if valid.sum() < max(10, int(WINDOW * 0.5)):
            continue
        xv = col_x[valid]; yv = col_y[valid]
        x_dm = xv - xv.mean(); y_dm = yv - yv.mean()
        denom = np.dot(x_dm, x_dm)
        if denom < 1e-12:
            continue
        beta  = np.dot(x_dm, y_dm) / denom
        alpha = yv.mean() - beta * xv.mean()
        curr_x = log_amnt[i, j]
        if np.isnan(curr_x):
            continue
        resid_mat[i, j] = log_turn[i, j] - (alpha + beta * curr_x)

factor_raw = pd.DataFrame(resid_mat, index=dates, columns=stocks)
non_null = factor_raw.notna().mean().mean()
print(f"   非空率: {non_null:.2%}")
print(f"   均值: {factor_raw.stack().mean():.4f}, std: {factor_raw.stack().std():.4f}  ({time.time()-t0:.1f}s)")

# ────────────────── 3. 截面 Z-score + 5% 缩尾 ──────────────────
print(f"[3] 截面 Z-score & 缩尾 ({WINSORIZE_PCT*100:.0f}%)...")
def cs_clean(mat: pd.DataFrame, wins_pct: float = WINSORIZE_PCT) -> pd.DataFrame:
    arr = mat.values.copy()
    for i in range(arr.shape[0]):
        row = arr[i]
        valid = row[~np.isnan(row)]
        if len(valid) < 20:
            continue
        mu = valid.mean(); sd = valid.std()
        if sd > 1e-8:
            row[~np.isnan(row)] = (valid - mu) / sd
        arr[i] = row
    mat = pd.DataFrame(arr, index=mat.index, columns=mat.columns)
    # 缩尾
    for d in dates:
        r = mat.loc[d].dropna()
        if len(r) < 10: continue
        lo = r.quantile(wins_pct); hi = r.quantile(1 - wins_pct)
        mat.loc[d] = mat.loc[d].clip(lo, hi)
    # 再标准化
    arr2 = mat.values.copy()
    for i in range(arr2.shape[0]):
        row = arr2[i]; valid = row[~np.isnan(row)]
        if len(valid) < 10: continue
        mu = valid.mean(); sd = valid.std()
        if sd > 1e-8: row[~np.isnan(row)] = (valid - mu) / sd
    return pd.DataFrame(arr2, index=mat.index, columns=mat.columns)

factor_std = cs_clean(factor_raw)
print(f"   ({time.time()-t0:.1f}s)")

# ────────────────── 4. 成交额 OLS 中性化 ──────────────────
print(f"[4] 成交额 OLS 中性化...")
log_amt_20 = np.log(daily_amt.clip(min=1e-8))
log_amt_ma = pd.DataFrame(
    np.nanmean(np.log(daily_amt.clip(min=1e-8)), axis=1, keepdims=True),
    index=dates, columns=stocks
)  # 占位, 当日截面不用均值, 用当日成交额代理作为截面中性化自变量

fa_arr = factor_std.values.copy()
am_arr = log_amt_20  # same shape
factor_neutral = factor_std.copy()

for d_idx, d in enumerate(dates):
    f = factor_std.loc[d].dropna()
    # 用日成交额代理金额做中性化
    am_row = pd.Series(log_amt_20[d_idx], index=stocks).reindex(f.index).dropna()
    common = f.index.intersection(am_row.index)
    if len(common) < 30:
        continue
    y = f[common].values; x = am_row[common].values
    x_dm = x - x.mean(); y_dm = y - y.mean()
    denom = np.dot(x_dm, x_dm)
    if denom < 1e-12: continue
    beta  = np.dot(x_dm, y_dm) / denom
    alpha = y.mean() - beta * x.mean()
    resid = y_dm - beta * x_dm
    factor_neutral.loc[d, common] = resid
print(f"   ({time.time()-t0:.1f}s)")

# ────────────────── 5. 最终截面 Z-score ──────────────────
print(f"[5] 最终截面 Z-score...")
arr_fin = factor_neutral.values.copy()
for i in range(arr_fin.shape[0]):
    row = arr_fin[i]; valid = row[~np.isnan(row)]
    if len(valid) < 10: continue
    mu = valid.mean(); sd = valid.std()
    if sd > 1e-8: row[~np.isnan(row)] = (valid - mu) / sd
factor_final = pd.DataFrame(arr_fin, index=dates, columns=stocks).clip(-3, 3)
print(f"   非空率: {factor_final.notna().mean().mean():.2%}  ({time.time()-t0:.1f}s)")

# ────────────────── 6. 前瞻收益 ──────────────────
print(f"[6] 前瞻 {FORWARD_DAYS} 日收益...")
log_ret_mat = np.log1p(ret.values.clip(lower=-0.999))
cum_log     = np.cumsum(np.nan_to_num(log_ret_mat, nan=0.0), axis=0)
shift_cum   = np.roll(cum_log, -FORWARD_DAYS, axis=0)
shift_cum[-FORWARD_DAYS:] = np.nan
fwd_cum_log = shift_cum - cum_log
fwd_ret     = np.expm1(fwd_cum_log)
fwd_ret     = pd.DataFrame(fwd_ret, index=dates, columns=stocks)

# ────────────────── 7. 回测 ──────────────────
print(f"[7] 回测...")
fwd_periods = int(len(dates) / FORWARD_DAYS)
ic_pearson  = compute_ic_dynamic(factor_final, ret, FORWARD_DAYS, method="pearson")
ic_spearman = compute_ic_dynamic(factor_final, ret, FORWARD_DAYS, method="spearman")

gr, tv, hi = compute_group_returns(
    factor_final, ret, N_GROUPS, REBALANCE_FREQ, COST
)
mets = compute_metrics(gr, ic_pearson, ic_spearman, tv, N_GROUPS,
                       holdings_info=hi)

# ────────────────── 8. 方向确认 + 输出 ──────────────────
print(f"[8] 评估结果...")
ic_val  = float(mets["ic_mean"])
t_val   = float(mets["ic_t_stat"])
sh_val  = float(mets["long_short_sharpe"])
mono    = float(mets["monotonicity"])

gr_ann  = mets.get("group_returns_annualized", [])
print(f"   IC={ic_val:+.4f}  t={t_val:+.2f}  sh={sh_val:+.3f}  mono={mono:.2f}")
print(f"   group_ann: {[f'{g:.1%}' for g in gr_ann]}")

# IC 方向 = 正 => 高因子得分 → 高后续收益 => 本脚本因子直接正
# 若需要反向设: factor_final = -factor_final 再重跑回测
sign = 1 if ic_val > 0 else -1

fa_out = factor_final if sign > 0 else -factor_final

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
elapsed = time.time() - t0

# ── report.json ──
report = {
    "factor_id":   FACTOR_ID,
    "factor_name": "换手率-成交额回归残差 v1 fwd20 (换手率-量价效率因子)",
    "name_en":     "Turnover-Amount OLS Residual v1 Fwd20",
    "category":    "流动性 / 资金效率",
    "direction":   "正残差 ← 换手偏高(已确认)",
    "sign":        sign,
    "static_factor": False,
    "description": (
        "20日滚动 OLS 残差：每日 log(换手率) ~ log(成交额) 截面回归，取当前截面残差。"
        "正残差 = 同等成交下换手更高 → 资金消耗大于价格推动 → 流动性异常消耗方向。"
        f"lookback={WINDOW}日，前瞻={FORWARD_DAYS}日，rebalance={REBALANCE_FREQ}日"
    ),
    "construction": (
        "tp=换手率 pivot；daily_amt=tp×close（成交额代理）；"
        "20日滚动 OLS y=log(tp)~x=log(amt)；残差 = 当前 y - (α+β x)；"
        "截面 z-score → 5% 缩尾 → 成交额 OLS 中性化 → 最终截面 z-score"
    ),
    "params": dict(WINDOW=WINDOW, FORWARD_DAYS=FORWARD_DAYS,
                   REBALANCE_FREQ=REBALANCE_FREQ, N_GROUPS=N_GROUPS,
                   COST=COST, WINSORIZE_PCT=WINSORIZE_PCT),
    "period":       f"{dates[0].date()} ~ {dates[-1].date()}",
    "n_dates":      len(dates),
    "n_stocks":     len(stocks),
    "n_groups":     N_GROUPS,
    "rebalance_freq": REBALANCE_FREQ,
    "forward_days": FORWARD_DAYS,
    "cost_per_trade": COST,
    "metrics":      mets,
    "elapsed_sec":  round(elapsed, 2),
}
(OUTPUT_DIR / "backtest_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, default=str),
    encoding="utf-8"
)

# ── ic_series.json / cumulative_returns.json via save_backtest_data ──
ic_signed = ic_pearson if sign > 0 else -ic_pearson
rank_signed = ic_spearman if sign > 0 else -ic_spearman
save_backtest_data(gr, ic_signed, rank_signed, str(OUTPUT_DIR))
print(f"[DONE] {FACTOR_ID}  {elapsed:.1f}s")
print(json.dumps({k: mets.get(k) for k in
    ["ic_mean","ic_t_stat","ic_ir","long_short_sharpe",
     "monotonicity","ic_p_value","ic_significant_5pct","ic_significant_1pct","total_ic_ann"]},
    ensure_ascii=False, indent=2))
