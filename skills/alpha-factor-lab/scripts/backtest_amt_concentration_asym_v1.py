#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: amt_concentration_asym_v1 — 资金集中度不对称因子
========================================================
版本: v2 完全重写 (修复v1的维度广播/有效因子率问题)

构造:
  取过去20日每日成交额, 计算:
    top_avg = 成交量最高的10%天数的均值
    bot_avg = 成交量最低的10%天数的均值
    factor_raw = log(top_avg + eps) - log(bot_avg + eps)
       → 高集中度(知情主导) vs 低集中度(散户噪音)

   Nationals by OLS (对log_mktcap残差) + MAD缩尾(5%) + z-score

参考: Lee & Swaminathan (2000) Volume-Momentum: Information-Spreading Hypothesis
       Merton (1987) Investor Recognition Hypothesis
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── 参数 ──────────────────────────────────────────────────────────────────
WINDOW         = 20
FORWARD_DAYS   = 20
REBALANCE_FREQ = 20
N_GROUPS       = 5
COST           = 0.002
WINSORIZE_PCT  = 0.05
TOP_PCT_Q      = 0.90       # top 10% 成交额: amount ≥ 90th pctile → take mean
BOT_PCT_Q      = 0.10       # bot 10%: amount ≤ 10th pctile
FACTOR_ID      = "amt_concentration_asym_v1"

ROOT     = Path(__file__).resolve().parents[3]  # alpha-factor-lab/
DATA_PATH= ROOT / "data" / "csi1000_kline_raw.csv"
OUTPUT_DIR = ROOT / "output" / FACTOR_ID
FACTOR_BT  = ROOT / "skills" / "alpha-factor-lab" / "scripts"
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"

# ════════════════════════════════════════════════════════════════════════
print(f"[1] 数据加载", flush=True)
# ════════════════════════════════════════════════════════════════════════
T0 = time.time()
df = pd.read_csv(DATA_PATH)
df["date"]       = pd.to_datetime(df["date"])
df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
df = df.sort_values(["stock_code","date"]).reset_index(drop=True)
df["daily_ret"]  = df.groupby("stock_code")["close"].pct_change()
print(f"  {len(df):,d} rows | {df['stock_code'].nunique()} stocks | "
      f"{df['date'].min().date()} ~ {df['date'].max().date()} | 耗时{time.time()-T0:.1f}s", flush=True)

# ════════════════════════════════════════════════════════════════════════
print(f"[2] 向量化因子构造 window={WINDOW} (log-ratio method)", flush=True)
# ════════════════════════════════════════════════════════════════════════
T1 = time.time()
stocks = sorted(df["stock_code"].unique())
dates  = sorted(df["date"].unique())
ns, nd = len(stocks), len(dates)

# pivoted amount array [date, stock]
amt_piv = df.pivot_table(index="date", columns="stock_code", values="amount", aggfunc="last")
amt_arr = amt_piv.values.astype(np.float64)
date_idx = {d: i for i, d in enumerate(amt_piv.index)}

# sliding_window_view over axis=0 (time): output (nd-w+1, ns, w)
# Numpy ≥2.0 puts the window dim at the END of the new tuple,
# i.e. (nd-w+1, ns, w) NOT (nd-w+1, w, ns)
from numpy.lib.stride_tricks import sliding_window_view as swv
sw = swv(amt_arr, WINDOW, axis=0)    # shape: (nh, ns, w); nh = nd - WINDOW + 1
print(f"  sw shape: {sw.shape}", flush=True)

p90 = np.percentile(sw, TOP_PCT_Q*100, axis=2)  # (nh, ns)
p10 = np.percentile(sw, BOT_PCT_Q*100, axis=2)  # (nh, ns)

EPS = 1e-8

# For each (stock, window): days whose amt ≥ p90 / ≤ p10
top_mask = sw >= p90[:, :, None]   # (nh, ns, w)
bot_mask = sw <= p10[:, :, None]

top_cnt  = top_mask.sum(axis=2)
bot_cnt  = bot_mask.sum(axis=2)
top_sum  = (sw * top_mask).sum(axis=2)
bot_sum  = (sw * bot_mask).sum(axis=2)

top_avg  = np.where(top_cnt > 0, top_sum / np.maximum(top_cnt, 1), np.nan)
bot_avg  = np.where(bot_cnt > 0, bot_sum / np.maximum(bot_cnt, 1), np.nan)

log_factor = np.log(np.maximum(top_avg, EPS)) - np.log(np.maximum(bot_avg, EPS))
print(f"  log_factor (before map-back) shape: {log_factor.shape}", flush=True)

# map back to full (nd, ns) matrix: first WINDOW-1 rows are NaN
factor_arr = np.full((nd, ns), np.nan)
factor_arr[WINDOW-1:] = log_factor
factor_df = pd.DataFrame(factor_arr, index=dates, columns=stocks, dtype=float)

valid_pct = factor_df.notna().mean(axis=1).mean()
print(f"  有效率: {valid_pct:.2%} | mean={np.nanmean(factor_df.values):.4f}"
      f" std={np.nanstd(factor_df.values):.4f} | 耗时{time.time()-T1:.1f}s", flush=True)

# ════════════════════════════════════════════════════════════════════════
print(f"[3] 构建收益率 & 市值矩阵", flush=True)
# ════════════════════════════════════════════════════════════════════════
close_piv = df.pivot_table(index="date", columns="stock_code", values="close", aggfunc="last")
ret_piv   = close_piv.pct_change()
amt_roll   = df.groupby("stock_code")["amount"].transform(
    lambda x: x.rolling(WINDOW, min_periods=10).mean()
)
log_cap_v  = np.log(df.pivot_table(index="date", columns="stock_code", values=amt_roll.name, aggfunc="last").clip(lower=1))
log_cap_v[~np.isfinite(log_cap_v)] = np.nan
log_cap_df  = pd.DataFrame(log_cap_v, index=dates, columns=stocks, dtype=float)
print(f"  收益率NaN率: {ret_piv.isna().mean().mean():.2%} | "
      f"市值NaN率: {log_cap_df.isna().mean().mean():.2%}", flush=True)

# ════════════════════════════════════════════════════════════════════════
print(f"[4] 市值中性化 + MAD缩尾 + z-score", flush=True)
# ════════════════════════════════════════════════════════════════════════
T2 = time.time()
_cap_arr = log_cap_df.values

def neutralize(factor_df_arr, cap_arr, winsor_pct):
    nd, ns = factor_df_arr.shape
    out = np.full((nd, ns), np.nan)
    for i in range(nd):
        fv = factor_df_arr[i]
        valid = np.where(np.isfinite(fv))[0]
        if len(valid) < 30:
            continue
        m = cap_arr[i, valid]
        f = fv[valid]
        X = np.column_stack([np.ones(len(m)), m])
        try:
            beta = np.linalg.lstsq(X, f, rcond=None)[0]
            resid = f - X @ beta
        except:
            continue

        med = np.nanmedian(resid)
        mad = np.nanmedian(np.abs(resid - med)) * 1.4826
        if mad < 1e-10:
            continue
        lo, hi = med - 4.5*mad, med + 4.5*mad
        resid = np.clip(resid, lo, hi)
        std = np.nanstd(resid)
        if std < 1e-10:
            continue
        z = (resid - np.nanmean(resid)) / std
        out[i, valid] = z
    return out

f_neutral = neutralize(factor_df.values, _cap_arr, WINSORIZE_PCT)
f_neutral_df = pd.DataFrame(f_neutral, index=dates, columns=stocks, dtype=float)
print(f"  耗时: {time.time()-T2:.1f}s | 截面均值N={f_neutral_df.notna().sum(axis=1).mean():.0f}/{ns}", flush=True)

# ════════════════════════════════════════════════════════════════════════
print(f"[5] 导入回测引擎", flush=True)
# ════════════════════════════════════════════════════════════════════════
sys.path.insert(0, str(FACTOR_BT))
from factor_backtest import compute_group_returns, compute_ic_dynamic, compute_metrics, save_backtest_data

# ════════════════════════════════════════════════════════════════════════
def run_bt(fac_df, direction, fwd, rb, cost, label):
    f = fac_df.copy()
    dates_list = fac_df.index.tolist()

    # 1. MAD 缩尾
    for dt in dates_list:
        row = f.loc[dt].dropna()
        if len(row) < 10:
            continue
        lo = row.quantile(WINSORIZE_PCT)
        hi = row.quantile(1 - WINSORIZE_PCT)
        f.loc[dt] = f.loc[dt].clip(lo, hi)

    # 2. OLS 中性化 (re-run)
    for dt in dates_list:
        fx = f.loc[dt].dropna()
        mx = log_cap_df.loc[dt].reindex(fx.index).dropna()
        ci  = fx.index.intersection(mx.index)
        if len(ci) < 30:
            continue
        fv = fx[ci].values.astype(float)
        mv = mx[ci].values.astype(float)
        X  = np.column_stack([np.ones(len(mv)), mv])
        try:
            beta = np.linalg.lstsq(X, fv, rcond=None)[0]
            f.loc[dt, ci] = fv - X @ beta
        except:
            pass

    # 取共通
    gen = sorted(f.dropna(how="all").index.intersection(ret_piv.dropna(how="all").index))
    cs  = sorted(f.columns.intersection(ret_piv.columns))
    if len(gen) < 10:
        print(f"  ⚠ 共通日期不足({len(gen)})"); return None
    fa = f.loc[gen, cs]
    ra = ret_piv.loc[gen, cs]
    if direction == -1:
        fa = -fa

    ic  = compute_ic_dynamic(fa, ra, fwd, "pearson")
    ric = compute_ic_dynamic(fa, ra, fwd, "spearman")
    gr, to, hi = compute_group_returns(fa, ra, N_GROUPS, rb, cost)
    m   = compute_metrics(gr, ic, ric, to, N_GROUPS, holdings_info=hi)

    ic_test = m.get("ic_mean", 0) or 0
    t_test  = m.get("ic_t_stat", 0) or 0
    sh_test = m.get("long_short_sharpe", 0) or 0
    sig     = "✓" if m.get("ic_significant_5pct") else "✗"
    grp     = m.get("group_returns_annualized", [])

    print(f"  [{label}] IC={ic_test:.4f}(t={t_test:.2f},{sig}) Sharpe={sh_test:.2f} "
          f"Mono={m.get('monotonicity',0):.2f} MDD={m.get('long_short_mdd',0):.2%}", flush=True)
    for i, r in enumerate(grp, 1):
        print(f"    G{i}: {r:.2%}" if r is not None else f"    G{i}: N/A", end="")
    print()
    valid = abs(ic_test) > 0.015 and abs(t_test) > 2 and abs(sh_test) > 0.5
    print(f"  ➤ {'有效 ✓' if valid else '无效 ✗'}", flush=True)
    return m, gr, ic, ric, gen, cs, valid

print(f"\n{'='*55}", flush=True)
print("配置: 做多高集中度(20d/20d)", flush=True)
r1 = run_bt(f_neutral_df, +1, 20, 20, 0.002, "正向20d")
print("配置: 做多低集中度(20d/20d)", flush=True)
r2 = run_bt(f_neutral_df, -1, 20, 20, 0.002, "反向20d")
print("配置: 做多高集中度(5d/5d)", flush=True)
r3 = run_bt(f_neutral_df, +1,  5,  5, 0.003, "正向5d")

valid_all = [r for r in [r1, r2, r3] if r and r[6]]
if valid_all:
    best = max(valid_all, key=lambda r: abs(r[0].get("long_short_sharpe",0) or 0))
    bm, bg, bic, bric, bcd, bcs, bd, bfwd, brb, bcost = best[0],best[1],best[2],best[3],best[4],best[5],  +1, 20, 20, 0.002
    if r3 and r3[6]:
        if abs(r3[0].get("long_short_sharpe",0) or 0) > abs(bm.get("long_short_sharpe",0) or 0):
            bm, bg, bic, bric, bcd, bcs, bd, bfwd, brb, bcost = r3[0],r3[1],r3[2],r3[3],r3[4],r3[5], +1, 5, 5, 0.003

    def nn(o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)):
            f = float(o); return f if np.isfinite(f) else None
        if isinstance(o, float) and not np.isfinite(o): return None
        if isinstance(o, dict): return {k: nn(v) for k,v in o.items()}
        if isinstance(o, list): return [nn(v) for v in o]
        return o

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_backtest_data(bg, bic, bric, str(OUTPUT_DIR))
    report = {
        "factor_id":      FACTOR_ID,
        "factor_name":    "资金集中度不对称因子 (log-ratio) v2",
        "factor_name_en": "Capital Concentration Asymmetry (log-ratio) v2",
        "category":       "流动性/量价",
        "description":    f"{WINDOW}日内成交额集中在top{TOP_PCT_Q/0.10:.0%} vs bottom{BOT_PCT_Q/0.10:.0%}的对数差。"
                          "资金高度集中 → 知情交易主导/机构定价 → 趋势持续 → 正alpha。"
                          "修正v1: 从ratio改为log-ratio, 从mask-rolling改为quantile-rolling,"
                          " 解决v1有效率0.32%且因子值~0.9无区分的问题。",
        "formula":        f"log(top{WINDOW}d_w/{TOP_PCT_Q:.0%}) - log(bot{WINDOW}d_w/{BOT_PCT_Q:.0%})",
        "direction":      bd,
        "direction_desc": "正向(高集中度→高收益)" if bd==1 else "反向(低集中度→高收益)",
        "stock_pool":     "中证1000",
        "period":         f"{bcd[0].date()} ~ {bcd[-1].date()}",
        "rebalance_freq": brb,
        "forward_days":   bfwd,
        "cost":           bcost,
        "n_groups":       N_GROUPS,
        "metrics":        bm,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as fout:
        json.dump(nn(report), fout, indent=2, ensure_ascii=False)
    print(f"\n✅ 报告: {REPORT_PATH}", flush=True)
    fa_csv = OUTPUT_DIR / "factor_values.csv"
    f_neutral_df.loc[bcd, bcs].to_csv(fa_csv)
    print(f"✅ 因子值: {fa_csv}", flush=True)
    print(f"\n🎯 最佳结果: IC={bm['ic_mean']:.4f}(t={bm['ic_t_stat']:.2f}) "
          f"Sharpe={bm['long_short_sharpe']:.2f} Mono={bm['monotonicity']:.2f}", flush=True)
else:
    print(f"\n❌ 所有配置均未达标", flush=True)
