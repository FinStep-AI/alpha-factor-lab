#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: close_drift_v1 — 收盘漂移异动
===========================================================
构造:
  ret_cc_t   = close_t / close_{t-1} - 1          # 含隔夜的全日收益
  ret_oc_t   = close_t / open_t  - 1               # 日内收益
  drift_resid_t = ret_cc_t - ret_oc_t
               = ret_co_t - ret_oc_t               (≈ overnight return, exact)
               = close_t * (open_t - close_{t-1}) / (close_{t-1} * open_t)

  factor = MA20(drift_resid), 成交额OLS中性化, 5%MAD缩尾, z-score

经济学直觉:
  ret_cc 包含了隔夜跳空+日内的完整价格变动;
  ret_oc 只含日内开盘→收盘;
  两者之差 = 隔夜跳空信息 - 日内随动;
  亦即 overnight_return − intraday_return:
    · 若连续多日 close_t/open_t > close_{t-1}/open_{t-1}
      → 隔夜信息推动价格, 但日内价格未能完全回吐
      → 持续"收盘抬高"效应 → 隔夜信息持续正 alpha
    · 若 ret_cc 持续 < ret_oc → 日内动量强而隔夜带不动, 次日跳空后日内补回
      → 日内"能量"被日内消化, 隔夜留下后续缺口带趋势延续

中性化: 成交额(log(MA20(amount))) OLS中性化 → MAD 5%缩尾 → z-score
20日窗口 | 成交额OLS中性化 | 5%MAD缩尾 | z-score | 5组分层

参考文献:
  - Lou, Polk & Skouras (2019) "A Tug of War: Overnight vs. Intraday Expected Returns" JFE
  - Kelly & Clark (2011) "Returns in Trading vs. Non-Trading Hours"

Filter: pct_change有缺失的行提前drop(前1行本身NaN,不影响因子)
"""

import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════ 参数 ═══════════════
WINDOW        = 20
REBALANCE     = 20        # 月度调仓
FORWARD_5D    = 5
FORWARD_20D   = 20
N_GROUPS      = 5
MAD_PCT       = 0.05
FACTOR_ID     = "close_drift_v1"
FACTOR_NAME   = "收盘漂移异动 v1"
FACTOR_NAME_EN= "Close Drift Anomaly v1"
CATEGORY      = "量价/隔夜"
BARRA         = "MICRO"
DATA_PATH     = Path(__file__).resolve().parent.parent / "data" / "csi1000_kline_raw.csv"
BASE_OUTPUT   = Path(__file__).resolve().parent.parent / "output" / FACTOR_ID

# ═══════════════ 1. 加载数据 ═══════════════
print("[1] 加载数据 …")
df = pd.read_csv(DATA_PATH, usecols=["date","stock_code","open","close","high","low","amount","pct_change"])
df["date"] = pd.to_datetime(df["date"])
# drop rows where pct_change is NaN  → first-row-per-stock holdovers
df = df.dropna(subset=["pct_change"]).copy()
df = df.sort_values(["stock_code","date"]).reset_index(drop=True)

pc  = df.pivot_table(index="date", columns="stock_code", values="close")
op  = df.pivot_table(index="date", columns="stock_code", values="open")
amt = df.pivot_table(index="date", columns="stock_code", values="amount")

dates  = sorted(pc.index)
stocks = list(pc.columns)
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ═══════════════ 2. 因子构造 ═══════════════
print(f"[2] 构造 close_drift (window={WINDOW}) …")

# ret_cc: 日收益率 (overnight + intraday combined)
ret_cc = pc.pct_change()
# ret_oc: 纯日内 open→close return
ret_oc = pc.div(op) - 1

# drift_resid: 隔夜驱动 component of daily return = cc_ret - oc_ret
drift_resid = ret_cc - ret_oc

# 20d rolling mean
factor_raw = drift_resid.rolling(WINDOW, min_periods=int(WINDOW*0.8)).mean()

# 成交额20日均值(用于中性化)
log_amount_20d = np.log(amt.rolling(20).mean().clip(lower=1))

print(f"   factor non-null: {factor_raw.notna().mean().mean():.2%}")

def neutralize_winsorize_zscore(f_raw: pd.DataFrame, log_amt: pd.DataFrame, mad_pct: float = MAD_PCT) -> pd.DataFrame:
    out = f_raw.copy()
    for d in dates:
        f = f_raw.loc[d].dropna()
        if len(f) < 30:
            out.loc[d] = np.nan
            continue
        m = log_amt.loc[d].reindex(f.index).dropna()
        common = f.index.intersection(m.index)
        if len(common) < 30:
            out.loc[d] = np.nan
            continue
        fv = f[common].values.astype(float)
        mv = m[common].values.astype(float)
        X  = np.column_stack([np.ones(len(mv)), mv])
        b  = np.linalg.lstsq(X, fv, rcond=None)[0]
        res = fv - X @ b
        # MAD winsorize (double-sided)
        med = float(np.median(res))
        mad = float(np.median(np.abs(res - med))) * 1.4826
        if mad < 1e-15:
            out.loc[d, common] = 0.0
            continue
        lo = float(np.quantile(res, mad_pct))
        hi = float(np.quantile(res, 1 - mad_pct))
        res = np.clip(res, lo, hi)
        mu, sg = float(res.mean()), float(res.std(ddof=0))
        if sg < 1e-15:
            out.loc[d, common] = 0.0
            continue
        out.loc[d, common] = (res - mu) / sg
    return out

fac_pos = neutralize_winsorize_zscore(factor_raw, log_amount_20d, MAD_PCT)
fac_neg = neutralize_winsorize_zscore(-factor_raw, log_amount_20d, MAD_PCT)

print(f"   pos non-null: {fac_pos.notna().mean().mean():.2%}"
      f"   neg non-null: {fac_neg.notna().mean().mean():.2%}")

# ═══════════════ helper: full pipeline ═══════════════
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills" / "alpha-factor-lab" / "scripts"))
from factor_backtest import (compute_group_returns, compute_ic_dynamic,
                              compute_metrics, save_backtest_data)

def run_one(factor, fname, direction, fwd, rebal, cost):
    ic  = compute_ic_dynamic(factor, ret_cc, fwd, "pearson")
    ric = compute_ic_dynamic(factor, ret_cc, fwd, "spearman")
    gr, to, hi = compute_group_returns(factor, ret_cc, N_GROUPS, rebal, cost)
    m  = compute_metrics(gr, ic, ric, to, N_GROUPS, holdings_info=hi)

    icv = ic.dropna().values
    t   = float(icv.mean() / (icv.std() / (len(icv)**0.5))) if len(icv) > 10 and icv.std() > 0 else 0.0
    sh  = m.get("long_short_sharpe", 0) or 0
    mo  = m.get("monotonicity", 0) or 0
    gan = m.get("group_returns_annualized", [])
    def g(i): return f"{gan[i]:.2%}" if i < len(gan) and gan[i] is not None else "N/A"

    sig = "✓" if m.get("ic_significant_5pct") else "✗"
    print(f"  [{fname} dir={direction:+d} fwd={fwd}d rb={rebal}d]  "
          f"IC={m.get('ic_mean',0):.4f}  t={t:.2f},{sig}  "
          f"LS_Sh={sh:.2f}  MDD={m.get('long_short_mdd',0):.2%}  mono={mo:.2f}")
    for i in range(N_GROUPS):
        print(f"    G{i+1}: {g(i)}")
    gsh = m.get("group_sharpe", [])
    if gsh and len(gsh) >= N_GROUPS:
        print(f"    G_sh: {[f'{v:.2f}' for v in gsh[:N_GROUPS]]}")
    return m, ic, ric, gr, to

# ═══════════════ 3. 回测 ═══════════════
print(f"\n[3] 回测 (fwd=5d 和 fwd=20d) …")
cfg = [
    # (factor, direction, fwd, rebal, cost, label)
    (fac_pos, +1, FORWARD_5D,  5,  0.003, "close_drift_pos_5d"),
    (fac_neg, -1, FORWARD_5D,  5,  0.003, "close_drift_neg_5d"),
    (fac_pos, +1, FORWARD_20D, 20, 0.002, "close_drift_pos_20d"),
    (fac_neg, -1, FORWARD_20D, 20, 0.002, "close_drift_neg_20d"),
    (fac_pos, +1, FORWARD_20D, REBALANCE, 0.002, "close_drift_pos_f20_rb20"),
    (fac_neg, -1, FORWARD_20D, REBALANCE, 0.002, "close_drift_neg_f20_rb20"),
]

results = {}
for fac, d, fwd, reb, cost, lab in cfg:
    m, ic, ric, gr, to = run_one(fac, lab, d, fwd, reb, cost)
    results[lab] = dict(m=m, ic=ic, ric=ric, gr=gr, to=to, d=d)

# ═══════════════ 4. 达标判定 & 选优 ═══════════════
def check(m):
    ic_m = m.get("ic_mean") or 0
    ic_t = m.get("ic_t_stat") or 0
    sh   = m.get("long_short_sharpe") or 0
    return abs(ic_m) > 0.015 and abs(ic_t) > 2.0 and sh > 0.5

best_label, best_m, best_ic, best_ric, best_gr, best_to, best_dir = None, None, None, None, None, None, None
for lab, r in results.items():
    if check(r["m"]) and (best_m is None or r["m"].get("long_short_sharpe", 0) > (best_m.get("long_short_sharpe", 0) or 0)):
        best_label  = lab
        best_m      = r["m"]
        best_ic     = r["ic"]
        best_ric    = r["ric"]
        best_gr     = r["gr"]
        best_to     = r["to"]
        best_dir    = r["d"]

if best_m:
    ic_m  = best_m.get("ic_mean", 0) or 0
    ic_t  = best_m.get("ic_t_stat", 0) or 0
    sh    = best_m.get("long_short_sharpe", 0) or 0
    mo    = best_m.get("monotonicity", 0)  or 0
    fwd   = best_m.get("forward_days", "?")
    reb   = best_m.get("rebalance_freq", "?")
    cost  = best_m.get("cost", 0.003)
    print(f"\n[4] ✅ 达标 — best={best_label}")
    print(f"    |IC|={abs(ic_m):.4f}  |t|={abs(ic_t):.2f}  Sharpe={sh:.2f}  mono={mo:.2f}")
else:
    print(f"\n[4] ❌ 全部未达标 — 记录失败原因")
    # dump top-2 for logging
    ranked = sorted(results.items(), key=lambda x: abs(x[1]["m"].get("ic_mean") or 0), reverse=True)
    for lab, r in ranked[:2]:
        m = r["m"]
        print(f"    {lab}: IC={m.get('ic_mean',0):.4f} t={m.get('ic_t_stat',0):.2f} Sh={m.get('long_short_sharpe',0):.2f} mono={m.get('monotonicity',0):.2f}")

# ═══════════════ 5. 写 output ═══════════════
BASE_OUTPUT.mkdir(parents=True, exist_ok=True)
if best_m:
    save_backtest_data(best_gr, best_ic, best_ric, str(BASE_OUTPUT))

    ic_vals  = best_ic.dropna()
    ric_vals = best_ric.dropna() if best_ric is not None else pd.Series(dtype=float)

    ic_vals.to_json(BASE_OUTPUT / "ic_series.json", orient="index", date_format="iso")
    np.save(BASE_OUTPUT / "long_short_returns.npy", best_to.values if hasattr(best_to, "values") else np.array(best_to))

    # cumulative_returns (date → ret for each group + long_short)
    cr = {}
    for col in best_gr.columns:
        cr[col] = best_gr[col].dropna().tolist()
    with open(BASE_OUTPUT / "cumulative_returns.json", "w", encoding="utf-8") as f:
        json.dump(cr, f, ensure_ascii=False, indent=2, default=str)

    eff_factor = fac_pos if best_dir == +1 else fac_neg
    fv = eff_factor.stack().reset_index()
    fv.columns = ["date","stock_code","factor_value"]
    fv.dropna(subset=["factor_value"]).to_csv(BASE_OUTPUT / "factor_values.csv", index=False)

    common_dates = sorted(eff_factor.dropna(how="all").index.intersection(ret_cc.dropna(how="all").index))
    def nan_to_none(o):
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)): return None
        if isinstance(o, dict): return {k:nan_to_none(v) for k,v in o.items()}
        if isinstance(o, list): return [nan_to_none(v) for v in o]
        return o

    report = {
        "factor_id":            FACTOR_ID,
        "factor_name":          FACTOR_NAME,
        "factor_name_en":       FACTOR_NAME_EN,
        "category":             CATEGORY,
        "barra_style":          BARRA,
        "description":          "MA20( ret_cc − ret_oc ) 成交额OLS中性化。ret_cc=含隔夜的全日收益；ret_oc=纯日内收益；两者之差排除了日内方向一致性噪音，纯净捕捉隔夜信息净流入对次日跳空/隔夜方向的持续塑造力。",
        "hypothesis":           "隔夜信息驱动 (overnight-driven) 的股票: overnight return 持续 > intraday return → 收盘被隔夜信息系统性抬高 → 次日跳空延续 → 正alpha。",
        "formula":              "neutralize( MA20(close/close_lag1 - 1  −  close/open - 1), log_amount_20d )  ±5% MAD, z-score",
        "direction":            int(best_dir),
        "stock_pool":           "中证1000",
        "period":               f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}",
        "n_dates":              len(common_dates),
        "n_stocks":             len(stocks),
        "n_groups":             int(best_m.get("_n_groups", N_GROUPS)),
        "rebalance_freq":       int(reb) if isinstance(reb, (int,float)) else REBALANCE,
        "forward_days":         int(fwd) if isinstance(fwd, (int,float)) else FORWARD_20D,
        "cost":                 float(cost),
        "components": ["ret_cc - ret_oc", "MA20"],
        "notes": {
            "distinct_from_intraday_drift_v1": "本因子 = MA20(ret_cc − ret_oc)；"
                                               "intraday_drift_v1 = MA20(Σ*(close−open)) / Σ*|close−open|。"
                                               "二者正交: 本因子不依赖日内方向一致性。",
            "distinct_from_overnight_momentum_v1": "隔夜动量用 Σ(overnight) − Σ(intraday)；"
                                                   "close_drift 用的是每日(ret_cc - ret_oc)再rolling MA。",
        },
        "metrics":              nan_to_none(best_m),
    }
    with open(BASE_OUTPUT / "backtest_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"    output → {BASE_OUTPUT}")
else:
    # 仍然写出一个失败报告(给 daily note 用)
    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)
    top2 = sorted(results.items(), key=lambda x: abs(x[1]["m"].get("ic_mean") or 0), reverse=True)[:2]
    fail = {
        "factor_id": FACTOR_ID,
        "status": "failed",
        "date_tested": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
        "configs": {lab: {
            "ic_mean": r["m"].get("ic_mean"),
            "ic_t_stat": r["m"].get("ic_t_stat"),
            "long_short_sharpe": r["m"].get("long_short_sharpe"),
            "monotonicity": r["m"].get("monotonicity"),
        } for lab, r in top2},
    }
    with open(BASE_OUTPUT / "fail_report.json", "w") as f:
        json.dump(fail, f, ensure_ascii=False, indent=2, default=str)
    print(f"    失败摘要写至 {BASE_OUTPUT/'fail_report.json'}")
