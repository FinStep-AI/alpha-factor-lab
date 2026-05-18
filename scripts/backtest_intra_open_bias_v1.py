#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: intra_open_bias_v1 — 开盘跳空-振幅比 20日均值
===========================================================
构造:
  gap_t       = open_t - close_{t-1}
  intra_bias_t = gap_t / (high_t - low_t)      # 隔夜跳空占日内振幅比例
  factor       = MA20(intra_bias_t)            # 20日滚动均值

  正值 = 近期开盘跳空(高开)幅度相对日内振幅大
  负值 = 近期开盘跳空(低开)幅度相对日内振幅大

方向: 待实证定 — 先在两种方向都测一下; 根据 IC 符号最终标记
20日窗口 | 成交额OLS中性化 | 5%MAD缩尾 | z-score | 5组分层

经济学直觉:
  - 持续高开(正值) → 夜间利空不足/抢筹, 日内如果买方还能维持高位但振幅相对小 → 信号偏脆弱?
  - 持续低开(负值) → 夜间暴露了抛压, 但日内振幅能把缺口收回去(低开高走), 说明承接力好 → 反转alpha
  - 数据决定方向 ; 两种方向都先跑出来

参考文献:
  - Gao, Han, Li & Zhou (2018) / Bogousslavsky (2016)
  - 日内开盘效率类因子在A股中证1000的小盘股上尚未充分探索
"""

import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════ 参数 ═══════════════
WINDOW        = 20
REBALANCE     = 20
FORWARD       = 20
N_GROUPS      = 5
COST          = 0.002          # 双边成本 20bp（低频）
FACTOR_ID     = "intra_open_bias_v1"
DATA_PATH     = Path(__file__).resolve().parent.parent / "data" / "csi1000_kline_raw.csv"
OUTPUT_DIR    = Path(__file__).resolve().parent.parent / "output" / FACTOR_ID
REPORT_PATH   = OUTPUT_DIR / "backtest_report.json"

# ═══════════════ 1. 加载 ═══════════════
print("[1] 加载数据 …")
df = pd.read_csv(DATA_PATH, usecols=["date","stock_code","open","close","high","low","amount"])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code","date"]).reset_index(drop=True)

pc   = df.pivot_table(index="date", columns="stock_code", values="close")
hi   = df.pivot_table(index="date", columns="stock_code", values="high")
lo   = df.pivot_table(index="date", columns="stock_code", values="low")
op   = df.pivot_table(index="date", columns="stock_code", values="open")
amt  = df.pivot_table(index="date", columns="stock_code", values="amount")
ret  = pc.pct_change()
log_amount_20d = np.log(amt.rolling(20).mean().clip(lower=1))

dates  = sorted(ret.index)
stocks = list(ret.columns)
print(f"   {len(dates)} 日, {len(stocks)} 股")

# ═══════════════ 2. 因子构造 ═══════════════
print(f"[2] 构造 intra_open_bias (window={WINDOW}) …")

rng  = np.clip(hi.values - lo.values, 1e-8, None)
prev = pc.shift(1).values

# intra_bias = (open - prev_close) / (high - low)
# clip 到 ±25% ≈ 涨跌停范围
bias_raw = np.clip((op.values - prev) / rng, -0.25, 0.25)
intra_bias = pd.DataFrame(bias_raw, index=dates, columns=stocks)

# 两个候选版本
factor_pos = intra_bias.rolling(WINDOW, min_periods=int(WINDOW*0.8)).mean()   # 正向：高开多
factor_neg = (-intra_bias).rolling(WINDOW, min_periods=int(WINDOW*0.8)).mean() # 反向：低开多

# ingestion OLS neutralization + MAD winsorize + z-score for each direction
def neutralize_winsorize_zscore(f_raw: pd.DataFrame, log_amount: pd.DataFrame) -> pd.DataFrame:
    out = f_raw.copy()
    for d in dates:
        f = f_raw.loc[d].dropna()
        if len(f) < 30:
            out.loc[d] = np.nan
            continue
        m = log_amount.loc[d].reindex(f.index).dropna()
        common = f.index.intersection(m.index)
        if len(common) < 30:
            out.loc[d] = np.nan
            continue
        fv = f[common].values.astype(float)
        mv = m[common].values.astype(float)
        X  = np.column_stack([np.ones(len(mv)), mv])
        b  = np.linalg.lstsq(X, fv, rcond=None)[0]
        res = fv - X @ b
        med = float(np.median(res))
        mad = float(np.median(np.abs(res - med))) * 1.4826
        if mad < 1e-12:
            out.loc[d, common] = 0.0
            continue
        clipped = np.clip(res, med - 5*mad, med + 5*mad)
        mu, sg   = float(clipped.mean()), float(clipped.std(ddof=0))
        if sg < 1e-12:
            out.loc[d, common] = 0.0
            continue
        out.loc[d, common] = (clipped - mu) / sg
    return out

print("   neutralising positive direction …")
fac_pos = neutralize_winsorize_zscore(factor_pos, log_amount_20d)
print("   neutralising negative direction …")
fac_neg = neutralize_winsorize_zscore(factor_neg, log_amount_20d)

print(f"   pos非空: {fac_pos.notna().mean().mean():.2%}   neg非空: {fac_neg.notna().mean().mean():.2%}")

# ═══════════════ helper: full pipeline ═══════════════
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills" / "alpha-factor-lab" / "scripts"))
from factor_backtest import (compute_group_returns, compute_ic_dynamic,
                              compute_metrics, save_backtest_data)

def run_one(factor, fname, direction):
    ic  = compute_ic_dynamic(factor, ret, FORWARD, "pearson")
    ric = compute_ic_dynamic(factor, ret, FORWARD, "spearman")
    gr, tr, holdings = compute_group_returns(factor, ret, N_GROUPS, REBALANCE, COST)
    m   = compute_metrics(gr, ic, ric, tr, N_GROUPS, holdings_info=holdings)

    icv = ic.dropna().values
    icv = icv[~np.isnan(icv)]
    t   = float(icv.mean()/(icv.std()/(len(icv)**0.5))) if len(icv)>10 and icv.std()>0 else 0.0
    print(f"\n[{fname} d={direction}]  IC_mean={m['ic_mean']:.4f}  t={t:.2f}  "
          f"LS_sharpe={m['long_short_sharpe']:.2f}  MDD={m['long_short_mdd']:.2%}  mono={m['monotonicity']:.2f}")
    gann = m.get("group_returns_annualized", [])
    if len(gann)==N_GROUPS:
        print(f"  G1={gann[0]:.2%}  G2={gann[1]:.2%}  G3={gann[2]:.2%}  G4={gann[3]:.2%}  G5={gann[4]:.2%}")
    return m, ic, ric, gr, tr

print(f"\n[3] 回测: fwd={FORWARD}d reb={REBALANCE}d cost={COST*100:.1f}% …")

m_pos,ic_pos,ric_pos,gr_pos,tr_pos = run_one(fac_pos, "intra_open_bias_v1", +1)
m_neg,ic_neg,ric_neg,gr_neg,tr_neg = run_one(fac_neg, "intra_open_bias_v1", -1)

# 选优：先看 t 然后看 Sharpe, 最终用 IC 符号实际方向
best_m, best_ic, best_ric, best_gr, best_tr, best_dir = (
    (m_pos,ic_pos,ric_pos,gr_pos,tr_pos,+1)
    if abs(m_pos["ic_mean"]) >= abs(m_neg["ic_mean"]) else
    (m_neg,ic_neg,ric_neg,gr_neg,tr_neg,-1)
)

# ═══════════════ 3. 达标判定 ═══════════════
ic_mean = best_m.get("ic_mean")
ic_t    = best_m.get("ic_t_stat")
sharpe  = best_m.get("long_short_sharpe")
mono    = best_m.get("monotonicity")

passes = (
    (ic_mean is not None and abs(ic_mean) > 0.015) and
    (ic_t    is not None and abs(ic_t)    > 2.0)  and
    (sharpe  is not None and sharpe        > 0.5)
)
print(f"\n[4] 判定: |IC|={abs(ic_mean) if ic_mean else 'N/A':.4f}  |t|={abs(ic_t) if ic_t else 0:.2f}  Sharpe={sharpe:.2f}  mono={mono:.2f}")
print(f"    {'✅ 达标 — 写入 output + 准备入库' if passes else '❌ 未达标 — 记录失败原因'}")

# ═══════════════ 4. 写入 output（达标才写完整数据） ═══════════════
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_backtest_data(best_gr, best_ic, best_ric, str(OUTPUT_DIR))

effective_factor = fac_pos if best_dir == +1 else fac_neg
effective_factor_vals = effective_factor.stack().reset_index()
effective_factor_vals.columns = ["date","stock_code","factor_value"]
effective_factor_vals.to_csv(OUTPUT_DIR / "factor_values.csv", index=False)

def nan_to_none(obj):
    if isinstance(obj,(np.bool_,)):    return bool(obj)
    if isinstance(obj,(np.integer,)):  return int(obj)
    if isinstance(obj,(np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj,float) and (np.isnan(obj) or np.isinf(obj)): return None
    if isinstance(obj,dict):  return {k:nan_to_none(v) for k,v in obj.items()}
    if isinstance(obj,list):  return [nan_to_none(v) for v in obj]
    return obj

common_dates = sorted(effective_factor.dropna(how="all").index.intersection(ret.dropna(how="all").index))
report = {
    "factor_id":         FACTOR_ID,
    "factor_name":       "开盘跳空-振幅比 v1",
    "factor_name_en":    "Intra Open-Bias Ratio v1",
    "category":          "开盘/隔夜",
    "description":       "过去20日隔夜跳空(open-prev_close)占日内振幅(high-low)比例的均值，成交额OLS中性化+MAD缩尾+z-score。捕捉开盘跳空幅度相对日内振幅的信息密度。",
    "hypothesis":        "隔夜跳空幅度相对日内振幅较大的股票，开盘定价偏离更大、信息冲击更集中，后续收益有截面预测力。",
    "formula":           f"neutralize(MA20((open-prev_close)/(high-low)), log_amount_20d), direction={best_dir:+d}",
    "direction":         int(best_dir),
    "stock_pool":        "中证1000",
    "period":            f"{common_dates[0].strftime('%Y-%m-%d')} ~ {common_dates[-1].strftime('%Y-%m-%d')}",
    "n_dates":           len(common_dates),
    "n_stocks":          len(stocks),
    "n_groups":          N_GROUPS,
    "rebalance_freq":    REBALANCE,
    "forward_days":      FORWARD,
    "cost":              COST,
    "best_direction":    int(best_dir),
    "metrics":           nan_to_none(best_m),
}
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)
print(f"\n[5] 报告已写 {REPORT_PATH}")

if passes:
    print("\n✅ 因子达标 — 后续需: 写 compute 脚本, 写入 factor_csv output, 写入 factors.json, git commit+push")
    # print 配方建议
    print(f"   推荐配方: fwd={FORWARD}d reb={REBALANCE}d cost={COST*100:.1f}% direction={int(best_dir):+d}")
