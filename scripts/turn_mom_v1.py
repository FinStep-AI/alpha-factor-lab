#!/usr/bin/env python3
"""
turn_mom_v1 — 换手冲击效率动量因子

核心思路
-------
对每只股票按日常计算冲击效率（amplitude / turnover），然后计算其 20 日谱段动量
  (当日 + 前 19 日平均) - (向前 20～39 日均值和向後 40～59 日均值的交叉均值)
『当前价值 - 历史基线』的时序动量改进版，使用更长的历史基线
规避当日噪声，保留可持续的阻力差异

使用 log_amount 截面中性（OLS 残差）
输出方向: 提升 = 优质高换手

优势对比 turn_efficiency_v1
  * 一样 e_r2t: 20 日均值，仅换用较长的 60d 基准均值（前40 期均值的最近交叉均值）
  * 不再用历史 median 作锚（偏稳健但容易抹平信号）
  * 把中间 20 段 和左段 40~59、右段 20~39 均值比较
"""

import argparse, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════
# Neutralization & standardization helpers
# ══════════════════════════════════════════

def winsorize_mad(s: pd.Series, n_mad: float = 5.0) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if mad < 1e-9:
        return s
    return s.clip(med - n_mad * mad, med + n_mad * mad)


def neutralize_cs(factor: pd.Series, mktcap: pd.Series) -> pd.Series:
    mask = factor.notna() & mktcap.notna() & (mktcap > 0)
    out = pd.Series(np.nan, index=factor.index)
    if mask.sum() < 30:
        return out
    y = factor[mask].values
    X = np.log(mktcap[mask].values.reshape(-1, 1))
    X = np.hstack([np.ones((len(X), 1)), X])
    try:
        betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        out[mask] = y - X @ betas
    except Exception:
        pass
    return out


def cross_section_zscore(s_raw: pd.Series) -> pd.Series:
    s = winsorize_mad(s_raw, n_mad=5.0)
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if mad < 1e-9:
        return pd.Series(np.nan, index=s.index)
    return (s - med) / mad


# ══════════════════════════════════════════
# Main
# ══════════════════════════════════════════

def compute(args):
    kline = pd.read_csv(args.kline, parse_dates=["date"])
    kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)
    kline["stock_code"] = kline["stock_code"].astype(str).str.zfill(6)

    # ① daily_r2t = amplitude / turnover
    zero_mask = kline["turnover"] <= 0
    kline["turnover_safe"] = kline["turnover"].where(~zero_mask, np.nan)
    kline["turnover_safe"] = kline.groupby("stock_code")["turnover_safe"].transform(
        lambda x: x.fillna(x[x > 0].min() if (x > 0).any() else 0.01)
    )
    kline["daily_r2t"] = kline["amplitude"].clip(lower=0) / kline["turnover_safe"]

    # ② rolling windows on daily_r2t (per stock)
    wins = dict(
        w20 = ("daily_r2t", 20, "mean"),
        w40 = ("daily_r2t", 40, "mean"),
        w60 = ("daily_r2t", 60, "mean"),
    )
    for alias, (col, win, func) in wins.items():
        kline[alias] = (
            kline.groupby("stock_code")[col]
            .transform(lambda s: getattr(s.rolling(win, min_periods=max(win // 2, 5)), func)())
        )

    # ③ effective baseline = mean of (w40 … w60) — represents longer-term steady baseline
    #  fold-in (forward and backward) gives gradual startup protection
    kline["baseline_r2t"] = (kline["w40"].fillna(kline["w60"]) + kline["w60"].fillna(kline["w40"])) / 2

    # ④ e_r2t = MA20 - baseline(MA40+MA60)
    kline["e_r2t"] = kline["w20"] - kline["baseline_r2t"]

    # ⑤ log_amount as mktcap proxy
    kline["log_amount"] = np.log(kline["amount"].clip(lower=1))

    # ⑥ cross-section neutralization per-date
    rows = []
    work = kline.dropna(subset=["e_r2t", "log_amount"])
    for dt, grp in work.groupby("date"):
        if len(grp) < 50:
            continue
        neu = neutralize_cs(
            grp["e_r2t"].reset_index(drop=True),
            grp["log_amount"].reset_index(drop=True),
        )
        neu_s = pd.Series(neu, index=grp.index)
        z = cross_section_zscore(neu_s)
        tmp = grp[["date", "stock_code"]].copy()
        tmp["factor"] = z.values
        rows.append(tmp.dropna(subset=["factor"]))

    if not rows:
        print("ERROR: no factor values", file=sys.stderr)
        sys.exit(1)

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["date", "stock_code"]).reset_index(drop=True)
    out.to_csv(args.output, index=False)
    print(f"turn_mom_v1: {len(out)} rows | {out['date'].nunique()} dates | {out['date'].min()} ~ {out['date'].max()}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kline",  default="data/csi1000_kline_raw.csv")
    ap.add_argument("--output", default="data/factor_turn_mom_v1.csv")
    args = ap.parse_args()
    compute(args)


if __name__ == "__main__":
    main()
