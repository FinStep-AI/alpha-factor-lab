#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: daynight_split_v1 — 昼夜分离反转因子
============================================
构造逻辑:
  华安证券《昼夜分离，隔夜跳空与日内反转选股因子》(2020)
  BigQuant 文章 https://bigquant.com/wiki/doc/0V7PUsN9g8

  将收益拆成隔夜 + 日内两段，分别取绝对变幅/振幅归一化后等权合成，
  构造"跳空强度 / 日内走向偏离"度量的绝对值复合信号，整体反转使用。

  日度构建量:
    overnight_gap_norm  = |open / prev_close - 1|
                          ─────────────────────────────────
                          (high-low) / prev_close + ε    (当日振幅归一化)

    intraday_ret_norm   = |close / open - 1|   (日内绝对变幅)

    信号混合: raw = overnight_gap_norm * 1.0  +  intraday_ret_norm * 0.35
    高 raw  → 隔夜大跳空 + 日内继续大幅单边运行 → 市场定价滞后/情绪极端
    反向使用 (DIRECTION = -1) → 捕捉均值回复 alpha

  平滑: 40 日滚动均值
  中性化: 对数成交额 20 日均值 OLS 横截面回归取残差 → MAD 3σ 缩尾 → z-score

回测配置:
  forward_days  = 5（5日前瞻，与 close_low/neg_day_freq 一致）
  rebalance     = 5
  cost          = 0.003
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── 参数 ─────────────────────────────────────────────────────────────────
WINDOW           = 40          # 平滑窗口
FORWARD_DAYS     = 5           # 前瞻
REBALANCE_FREQ   = 5           # 调仓
N_GROUPS         = 5
COST             = 0.003
WINSORIZE_MAD    = 3.0         # MAD 缩尾 nσ
INTRADAY_WEIGHT = 0.35        # 日内权重
FACTOR_ID        = "daynight_split_v1"
FACTOR_NAME_CN   = "昼夜分离反转因子 v1"
FACTOR_NAME_EN   = "Day-Night Split Reversal v1"
DIRECTION        = -1          # 反向：高 split → 低预期收益
EPS              = 1e-6

BASE   = Path(__file__).resolve().parent.parent
DATA_KLINE   = BASE / "data" / "csi1000_kline_raw.csv"
DATA_RET     = BASE / "data" / "csi1000_returns.csv"
OUTPUT_DIR   = BASE / "output" / FACTOR_ID
FACTOR_CSV   = BASE / "data" / f"factor_{FACTOR_ID}.csv"
REPORT_PATH  = OUTPUT_DIR / "backtest_report.json"


# ─── 主流程 ────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print(f"\n{'='*55}", flush=True)
    print(f"  {FACTOR_NAME_CN}  ({FACTOR_ID})", flush=True)
    print(f"  华安昼夜分离 + BigQuant 反转因子本土化", flush=True)
    print(f"{'='*55}", flush=True)

    # 1. 读取数据
    print(f"\n[1] 读取 K线数据 ...", flush=True)
    df = pd.read_csv(DATA_KLINE,
                     usecols=["date","stock_code","open","close","high","low","amount"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code","date"]).reset_index(drop=True)
    print(f"    rows={len(df):,}  stocks={df['stock_code'].nunique()}"
          f"  {time.time()-t0:.1f}s", flush=True)

    # 2. 构造日度昼夜分裂信号
    print(f"\n[2] 计算日度 overnight_gap / intraday_ret ...", flush=True)
    t1 = time.time()

    prev_close = df.groupby("stock_code")["close"].shift(1).values

    open_arr   = df["open"].values
    close_arr  = df["close"].values
    high_arr   = df["high"].values
    low_arr    = df["low"].values

    overnight_gap = np.abs(open_arr / prev_close - 1.0)
    intra_rng     = (high_arr - low_arr) / prev_close + EPS
    # gap norm = 隔夜跳空占当日振幅比
    gap_norm  = overnight_gap / intra_rng

    # intraday ret norm
    intra_ret = np.abs(close_arr / open_arr - 1.0)

    df["overnight_gap_norm"] = gap_norm
    df["intraday_ret_norm"]  = intra_ret

    # 信号融合: 隔夜 + 加权日内，取绝对值均可
    df["daynight_raw_daily"] = (
        gap_norm * 1.0 + intra_ret * INTRADAY_WEIGHT
    )
    _g = gap_norm[np.isfinite(gap_norm)]
    _i = intra_ret[np.isfinite(intra_ret)]
    print(f"    overnight_gap_norm  mean={_g.mean():.4f}  "
          f"intraday_norm mean={_i.mean():.4f}  "
          f"{time.time()-t1:.1f}s", flush=True)

    # 3. 滚动平滑
    print(f"\n[3] {WINDOW}日滚动均值平滑 ...", flush=True)
    t2 = time.time()

    df["factor_raw"] = df.groupby("stock_code")["daynight_raw_daily"].transform(
        lambda s: s.rolling(WINDOW, min_periods=max(WINDOW // 3, 10)).mean()
    )

    def _finite(s):
        m = np.isfinite(s)
        return s[m] if m.any() else pd.Series(dtype=float)
    finite_vals = _finite(df["factor_raw"])
    if len(finite_vals):
        print(f"    raw mean={finite_vals.mean():.4f}  std={finite_vals.std():.4f}"
              f"  coverage={df['factor_raw'].notna().mean():.2%}",
              flush=True)
    else:
        raise RuntimeError("factor_raw 全 NaN")
    print(f"    {time.time()-t2:.1f}s", flush=True)

    # 4. 构建 pivot
    print(f"\n[4] 构建 pivot 矩阵 ...", flush=True)
    t3 = time.time()

    dates  = sorted(df["date"].unique())
    stocks = sorted(df["stock_code"].unique())
    dmap   = {d: i for i, d in enumerate(dates)}
    smap   = {s: i for i, s in enumerate(stocks)}

    F = np.full((len(dates), len(stocks)), np.nan)
    for _, row in df.dropna(subset=["factor_raw"]).iterrows():
        F[dmap[row["date"]], smap[row["stock_code"]]] = row["factor_raw"]
    factor_df = pd.DataFrame(F, index=dates, columns=stocks)

    # 成交额 20 日均值对数（市值代理）
    df["log_amt20"] = np.log(
        df.groupby("stock_code")["amount"]
          .transform(lambda x: x.rolling(20, min_periods=5).mean())
          .clip(lower=1.0)
    )
    log_amt_piv = df.pivot_table(index="date", columns="stock_code",
                                 values="log_amt20", dropna=False)

    print(f"    dates={len(dates)}  stocks={len(stocks)}  {time.time()-t3:.1f}s",
          flush=True)

    # 5. 截面 OLS 中性化 + MAD 缩尾
    print(f"\n[5] 截面OLS中性化 (log_amt20) + MAD{WINSORIZE_MAD}σ ...", flush=True)
    t4 = time.time()

    factor_neu = factor_df.copy()
    for dt in dates:
        f = factor_df.loc[dt].dropna()
        m = log_amt_piv.loc[dt].reindex(f.index).dropna()
        common = f.index.intersection(m.index)
        if len(common) < 30:
            continue
        f_c = f[common].values.astype(float)
        m_c = m[common].values.astype(float)
        X   = np.column_stack([np.ones(len(m_c)), m_c])
        try:
            b  = np.linalg.lstsq(X, f_c, rcond=None)[0]
            res = f_c - X @ b
            factor_neu.loc[dt, common] = res
        except Exception:
            pass

    # MAD 缩尾
    for dt in dates:
        row = factor_neu.loc[dt].dropna()
        if len(row) < 10:
            continue
        med = row.median()
        mad = (row - med).abs().median() * 1.4826 + 1e-8
        lo  = med - WINSORIZE_MAD * mad
        hi  = med + WINSORIZE_MAD * mad
        factor_neu.loc[dt] = factor_neu.loc[dt].clip(lo, hi)

    print(f"    mean={factor_neu.stack().mean():.4f}  std={factor_neu.stack().std():.4f}"
          f"  {time.time()-t4:.1f}s", flush=True)

    # 6. 读取收益矩阵
    print(f"\n[6] 读取收益矩阵 (long→pivot) ...", flush=True)
    ret_long = pd.read_csv(DATA_RET)  # [date, stock_code, return]
    ret_long["date"] = pd.to_datetime(ret_long["date"])
    ret_piv = ret_long.pivot_table(index="date", columns="stock_code",
                                   values="return", dropna=False)
    print(f"    ret_piv: {ret_piv.shape}", flush=True)

    # 7. 回测
    print(f"\n[7] 回测引擎 fwd={FORWARD_DAYS}d, rb={REBALANCE_FREQ}d, cost={COST:.3f}",
          flush=True)
    t5 = time.time()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                          / "skills" / "alpha-factor-lab" / "scripts"))
    from factor_backtest import (
        compute_group_returns, compute_ic_dynamic,
        compute_metrics, save_backtest_data,
    )

    common_dates  = sorted(
        factor_neu.dropna(how="all").index
        .intersection(ret_piv.dropna(how="all").index)
    )
    common_stocks = sorted(factor_neu.columns.intersection(ret_piv.columns))
    F_al = factor_neu.loc[common_dates, common_stocks]
    R_al = ret_piv.loc[common_dates, common_stocks]
    print(f"    矩阵: {len(common_dates)}日 × {len(common_stocks)}股  {time.time()-t5:.1f}s",
          flush=True)

    t6 = time.time()
    ic_series  = compute_ic_dynamic(F_al, R_al, FORWARD_DAYS, "pearson")
    ric_series = compute_ic_dynamic(F_al, R_al, FORWARD_DAYS, "spearman")
    grp, to, hi = compute_group_returns(F_al, R_al, N_GROUPS, REBALANCE_FREQ, COST)
    metrics  = compute_metrics(grp, ic_series, ric_series, to, N_GROUPS,
                               holdings_info=hi)
    print(f"    回测完成 {time.time()-t6:.1f}s", flush=True)

    # 8. 打印摘要
    ic_mean = metrics.get("ic_mean") or 0.0
    ic_t    = metrics.get("ic_t_stat") or 0.0
    ls_sh   = metrics.get("long_short_sharpe") or 0.0
    ls_ann  = metrics.get("long_short_ann_return") or 0.0
    ls_mdd  = metrics.get("long_short_mdd") or 0.0
    mono    = metrics.get("monotonicity") or 0.0

    print(f"\n{'='*55}")
    print(f"  {FACTOR_NAME_CN}")
    print(f"{'='*55}")
    print(f"  周期: {common_dates[0]} ~ {common_dates[-1]}  n={len(common_dates)}")
    print(f"  IC_mean={ic_mean:.4f}  t={ic_t:.2f}  IR={metrics.get('ir',0) or 0:.3f}"
          f"  正IC比={metrics.get('ic_positive_ratio',0) or 0:.2%}")
    r_ic  = metrics.get("rank_ic_mean") or 0.0
    r_ics = metrics.get("rank_ic_std") or 0.0
    print(f"  Rank_IC={r_ic:.4f}  std={r_ics:.4f}")
    print(f"  LS Sharpe={ls_sh:.3f}  LS 年化={ls_ann:.2%}  LS MDD={ls_mdd:.2%}")
    print(f"  换手率={metrics.get('turnover_mean',0) or 0:.3f}  单调性={mono:.3f}")
    grp_ann = metrics.get("group_returns_annualized", [])
    g_sh    = metrics.get("group_sharpe", [])
    for i in range(N_GROUPS):
        g = grp_ann[i] if i < len(grp_ann) else None
        s = g_sh[i]    if i < len(g_sh)    else None
        print(f"    G{i+1}: {g:>10.2%}" if g is not None
              else f"    G{i+1}: N/A")
    print(f"{'='*55}")

    # 9. 输出文件
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_backtest_data(grp, ic_series, ric_series, str(OUTPUT_DIR))

    # factor CSV
    if len(common_stocks) == len(stocks):
        out_F = factor_neu.loc[common_dates, common_stocks]
    else:
        out_F = factor_neu.loc[common_dates, :]
    out_F.to_csv(FACTOR_CSV)
    print(f"\n  因子值  → {FACTOR_CSV}")

    def _o(x):
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            v = float(x)
            return None if (np.isnan(v) or np.isinf(v)) else v
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return None
        if isinstance(x, dict):
            return {k: _o(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_o(v) for v in x]
        return x

    report = {
        "factor_id":      FACTOR_ID,
        "factor_name":    FACTOR_NAME_CN,
        "factor_name_en": FACTOR_NAME_EN,
        "direction":      DIRECTION,
        "window":         WINDOW,
        "intraday_weight": INTRADAY_WEIGHT,
        "period": (f"{common_dates[0].strftime('%Y-%m-%d')}"
                   f" ~ {common_dates[-1].strftime('%Y-%m-%d')}"),
        "n_dates":        len(common_dates),
        "n_stocks":       len(common_stocks),
        "forward_days":   FORWARD_DAYS,
        "rebalance_freq": REBALANCE_FREQ,
        "cost":           COST,
        "neutralization": "OLS(log_amount_20d)",
        "winsorize":      f"MAD{WINSORIZE_MAD}σ",
        "metrics":        metrics,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(_o(report), f, indent=2, ensure_ascii=False)
    print(f"  报告   → {REPORT_PATH}")
    print(f"  总耗时: {time.time()-t0:.1f}s\n")


if __name__ == "__main__":
    main()
