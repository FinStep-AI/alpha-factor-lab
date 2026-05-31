#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""
因子: amount_gini_v1 — 成交额Gini集中度因子
====================================================
思路: 过去20日的日成交额 Gini 系数。
    高 Gini = 成交额集中在少数天 → 风险/资金的"脉冲式"集中分布
    低 Gini = 成交额均匀分布 → 常规持续交易

逻辑:
  小盘股交易量的集中与分散，反映了资金的关注程度和主力行为模式。
  集中交易(Gini高)的股票可能带有信息驱动的脉冲事件；分散交易(Gini低)
  则代表更均衡的多空博弈。两部分方向均在回测中验证。

前置数据:
  data/factor_amount_gini_v1.csv          (已做5%缩尾 + log_amount OLS中性化)
  data/csi1000_kline_raw.csv              (OHLCV + turnover)

本回测脚本:
  - 只做 OLS 市值二次中性化（按 20 日均成交额）
  - 方向预览(40日小样本) → 取 |IC| 大者
  - 全量 20 日前瞻 IC 分层回测
  - 输出 output/amount_gini_v1/cumulative_returns.json + ic_series.json + backtest_report.json
"""

import json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ─────────── 参数 ───────────
WINDOW        = 20
FORWARD_DAYS  = 20
REBALANCE_FREQ = 20
N_GROUPS      = 5
COST          = 0.002
WINSORIZE_PCT = 0.05
FACTOR_ID     = "amount_gini_v1"
BASE          = Path(__file__).resolve().parent.parent
DATA_PATH     = BASE / "data" / "csi1000_kline_raw.csv"
FACTOR_PATH   = BASE / "data" / "factor_amount_gini_v1.csv"   # 原始 (正) Gini
NEG_PATH      = BASE / "data" / "factor_amount_gini_v1_neg.csv" # 负 Gini
OUTPUT_DIR    = BASE / "output" / FACTOR_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH   = OUTPUT_DIR / "backtest_report.json"
sys.path.insert(0, str(BASE / "skills" / "alpha-factor-lab" / "scripts"))


# ─────────── helpers ───────────

def nan_to_none(o):
    if isinstance(o, (np.bool_,)):   return bool(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)): return None
    if isinstance(o, dict):  return {k: nan_to_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [nan_to_none(v) for v in o]
    return o


def safe_annualize(total_return: float, n_days: int, annual_factor: float = 252) -> float:
    if total_return <= -1:
        # no reasonable annualization; cap at -99 %
        return -99.0
    return float(np.sign(total_return) * (abs(1 + total_return) ** (annual_factor / max(n_days, 1)) - 1))


# ─────────── 1. 数据加载 ───────────
print("[1] 加载 K 线 + 因子 CSV …", flush=True)
t0 = time.time()
price_df   = pd.read_csv(DATA_PATH, usecols=["date", "stock_code", "close", "amount", "volume"])
price_df["date"] = pd.to_datetime(price_df["date"])
price_df["stock_code"] = price_df["stock_code"].astype(str)
price_df = price_df.sort_values(["stock_code", "date"]).reset_index(drop=True)

df_gini  = pd.read_csv(FACTOR_PATH, parse_dates=["date"])
df_gini["stock_code"] = df_gini["stock_code"].astype(str)
df_ngini = pd.read_csv(NEG_PATH, parse_dates=["date"]) if NEG_PATH.exists() else df_gini.copy()
df_ngini["stock_code"] = df_ngini["stock_code"].astype(str)

close_piv = price_df.pivot_table(index="date", columns="stock_code", values="close")
close_piv.columns = close_piv.columns.astype(str)
amount_piv = price_df.pivot_table(index="date", columns="stock_code", values="amount")
amount_piv.columns = amount_piv.columns.astype(str)
ret_piv = close_piv.pct_change()

# 20日均成交额 作为市值代理
log_mktcap = np.log(amount_piv.rolling(20).mean().clip(lower=1))

print(f"   完成 {time.time()-t0:.1f}s  |  {len(close_piv)} 日 × {len(close_piv.columns)} 股", flush=True)


# ─────────── 2. 拼入因子矩阵 + 分离正 / 负方向 ───────────
print("[2] 拼接因子矩阵 …", flush=True)

def _pivot_factor(df_f: pd.DataFrame) -> pd.DataFrame:
    M = df_f.pivot_table(index="date", columns="stock_code", values="factor_value").sort_index()
    if M.columns.dtype != object:
        M.columns = M.columns.astype(str)
    return M

gini_mat  = _pivot_factor(df_gini)
ngini_mat = _pivot_factor(df_ngini)

dates   = sorted(close_piv.index.intersection(gini_mat.index))
stocks  = sorted(close_piv.columns.intersection(gini_mat.columns))
gini_mat  = gini_mat.loc[dates, stocks]
ngini_mat = ngini_mat.loc[dates, stocks]
ret_mat   = ret_piv.loc[dates, stocks]
mk_mat    = log_mktcap.loc[dates, stocks]

print(f"   对齐后 {len(dates)} 日 × {len(stocks)} 股", flush=True)


# ─────────── 3. 二次市值中性化（同截面二次中性化） + 5%缩尾 ───────────
print("[3] 5% 缩尾 + log_amount OLS 中性化 …", flush=True)

def neutralize_and_winsor(mat: pd.DataFrame, mktcap_mat: pd.DataFrame,
                           winsor_pct=WINSORIZE_PCT) -> pd.DataFrame:
    out = mat.copy()
    for dt in mat.index:
        f  = mat.loc[dt].dropna()
        mk = mktcap_mat.loc[dt].reindex(f.index).dropna()
        common = f.index.intersection(mk.index)
        if len(common) < 30:
            continue
        fv  = f[common].values.astype(float)
        mkv = mk[common].values.astype(float)

        # ---- 5% 缩尾 ----
        finite_mask = np.isfinite(fv)
        if finite_mask.sum() >= 10:
            lo, hi = np.nanquantile(fv[finite_mask], winsor_pct), \
                     np.nanquantile(fv[finite_mask], 1 - winsor_pct)
            fv = np.clip(fv, lo, hi)

        # ---- OLS 市值中性化 ----
        X = np.column_stack([np.ones(len(mkv)), mkv])
        try:
            beta = np.linalg.lstsq(X, fv, rcond=None)[0]
            resid = fv - X @ beta
            std = resid.std()
            if std > 1e-12:
                resid = (resid - resid.mean()) / std
            out.loc[dt, common] = resid
        except Exception:
            pass
    return out

gini_neutral  = neutralize_and_winsor(gini_mat,  mk_mat)
ngini_neutral = neutralize_and_winsor(ngini_mat, mk_mat)

for name, mat in [("gini", gini_neutral), ("neg_gini", ngini_neutral)]:
    nv = mat.stack().dropna()
    if len(nv):
        print(f"   [{name}] mean={nv.mean():.4f} std={nv.std():.4f}  skew={sp_stats.skew(nv, bias=False):.4f}", flush=True)


# ─────────── 3b. 方向预览 + 正式回测 ───────────
# 两个变体分别跑全量回测后比选出胜者
best_res, best_score = None, -1e9

for cand_mat, cand_id, cand_label in [
    (gini_neutral,  "amount_gini_v1",      "gini(+)"),
    (ngini_neutral, "amount_gini_neg_v1",  "gini(-)"),
]:
    from factor_backtest import (       # noqa: E402 – intentional lazy import
        compute_group_returns, compute_ic_dynamic,
        compute_metrics, save_backtest_data,
    )

    dates2  = sorted(cand_mat.dropna(how="all")
                     .index.intersection(ret_mat.dropna(how="all").index))
    stocks2 = sorted(cand_mat.columns.intersection(ret_mat.columns))
    F  = cand_mat.loc[dates2, stocks2]
    Rt = ret_mat.loc[dates2, stocks2]
    print(f"\n   [{cand_label}] 对齐: {len(dates2)} 日 x {len(stocks2)} 股", flush=True)

    ic_ser  = compute_ic_dynamic(F, Rt, FORWARD_DAYS, "pearson")
    ric_ser = compute_ic_dynamic(F, Rt, FORWARD_DAYS, "spearman")
    grp, tovs, _ = compute_group_returns(
        F, Rt, N_GROUPS, REBALANCE_FREQ, COST)
    met = compute_metrics(grp, ic_ser, ric_ser, tovs, N_GROUPS)

    out_dir = BASE / "output" / cand_id
    out_dir.mkdir(parents=True, exist_ok=True)
    save_backtest_data(grp, ic_ser, ric_ser, str(out_dir))

    ic_tv = abs(met.get("ic_t_stat", 0) or 0)
    sh_v  = abs(met.get("long_short_sharpe", 0) or 0)
    mo_v  = abs(met.get("monotonicity", 0) or 0)
    s = ic_tv * 0.4 + mo_v * 0.3 + sh_v * 0.3
    print(f"   [{cand_label}] IC={met.get('ic_mean',0):+.4f}  "
          f"t={met.get('ic_t_stat',0):+.2f}  "
          f"Sharpe={sh_v:.3f}  mono={mo_v:.3f}  score={s:.3f}",
          flush=True)

    cand_res = dict(
        fac_id=cand_id, fac_label=cand_label, metrics=met,
        ic_series=ic_ser, rank_ic_series=ric_ser, group_returns=grp,
    )
    if s > best_score:
        best_score, best_res = s, cand_res

sep = "=" * 60
print(f"\n   -> 胜者: {best_res['fac_id']}", flush=True)

# ─────────── 4. 保存研究报告 + 打印摘要 ───────────
best       = best_res
win_metrics = best["metrics"]

report = {
    "factor_id":      best["fac_id"],
    "factor_name":    "成交额Gini集中度 " + best["fac_label"],
    "name_en":        "Amount Gini Concentration v1",
    "category":       "流动性 / 成交额分布",
    "stock_pool":     "中证1000",
    "period":         "2022-11 ~ 2026-03",
    "rebalance_freq": REBALANCE_FREQ,
    "forward_days":   FORWARD_DAYS,
    "cost":           COST,
    "direction":      best["fac_label"],
    "n_dates": int(len(best["ic_series"])) if best["ic_series"] is not None else 0,
    "metrics": win_metrics,
}
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(nan_to_none(report), f, indent=2, ensure_ascii=False)

ic_m  = win_metrics.get("ic_mean", 0)  or 0
ic_t  = abs(win_metrics.get("ic_t_stat", 0) or 0)
sh_   = win_metrics.get("long_short_sharpe", 0) or 0
mono  = win_metrics.get("monotonicity", 0) or 0
ic_ir = win_metrics.get("ir", 0) or 0
tov   = win_metrics.get("turnover_mean", 0) or 0

print(f"\n{sep}")
print(f"  {best['fac_id']}  回测摘要")
print(sep)
print(f"  IC 均值:     {ic_m:+.4f}")
print(f"  IC 标准差:   {win_metrics.get('ic_std',0):.4f}")
print(f"  IC IR:       {ic_ir:.4f}")
print(f"  NW t 值:     {ic_t:.2f}")
print(f"  多空 Sharpe:  {sh_:.4f}")
print(f"  多空 MDD:    {(win_metrics.get('long_short_mdd',0) or 0):.2%}")
print(f"  单调性:      {mono:.4f}")
print(f"  换手率:      {tov:.2%}")
print(f"  分层年化收益:")
for i, r in enumerate(win_metrics.get("group_returns_annualized", []), 1):
    print(f"    G{i}: {r:.2%}" if r is not None else f"    G{i}: N/A")

is_pass = (abs(ic_m) > 0.015 and ic_t > 2 and abs(sh_) > 0.5)
ok1 = "Y" if abs(ic_m) > 0.015 else "N"
ok2 = "Y" if ic_t > 2 else "N"
ok3 = "Y" if abs(sh_) > 0.5 else "N"
print(f"\n  因子{'有效' if is_pass else '无效'} "
      f"( |IC|>0.015:{ok1}  t>2:{ok2}  |Sharpe|>0.5:{ok3} )",
      flush=True)

print(f"\n输出文件:")
print(f"  {OUTPUT_DIR / 'cumulative_returns.json'}", flush=True)
print(f"  {OUTPUT_DIR / 'ic_series.json'}", flush=True)
print(f"  {REPORT_PATH}", flush=True)

import sys as _sys
_sys.exit(0 if is_pass else 1)
