#!/usr/bin/env python3
"""
turn_efficiency_v1 — 换手率-价格冲击效率因子

idea
----
用日内振幅和换手率刻画『每单位流动性产生的价格冲击』：
  daily_r2t = amplitude / turnover

基础逻辑和高换手但低振幅的股票不同，它衡量的是：
  * 同量换手下 → 振幅低的股票其实冲击很低（和 range_efficiency_v1 正相关）；
  * 同振幅下 → 换手率低的股票对每一个百分点换手率"承载"的价格步进更大。

Alpha 来源：
  e_r2t_t = mean(daily_r2t, 20d) - median(daily_r2t, 60d)
  以个股自身历史中位数做锚定保留跨股票差异
  更高 e_r2t = 当前单位换手率触发的价格步进更高→价格发现更快→后续收益更优

输出列为 date / stock_code / factor_raw（行情中性化后的分数越高越好）

neutralize 用截面 OLS 残差（不引入行业变量）
"""

import argparse, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------- 行业中性化（OLS 残差，未做行业虚拟变量） ----------

def winsorize_mad(s: pd.Series, n_mad: float = 5.0) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if mad < 1e-9:
        return s
    lo, hi = med - n_mad * mad, med + n_mad * mad
    return s.clip(lo, hi)


def neutralize_cs(factor: pd.Series, mktcap: pd.Series) -> pd.Series:
    """截面 OLS 中性化：factor ~ log(mktcap)，返回残差。"""
    mask = factor.notna() & mktcap.notna() & (mktcap > 0)
    out = pd.Series(np.nan, index=factor.index)
    if mask.sum() < 30:
        return out
    y = factor[mask].values
    log_cap = np.log(mktcap[mask].values)
    log_cap_mat = np.column_stack([np.ones(len(log_cap)), log_cap])
    try:
        beta, _, _, _ = np.linalg.lstsq(log_cap_mat, y, rcond=None)
        resid = y - log_cap_mat @ beta
    except Exception:
        return out
    out[mask] = resid
    return out


def robust_zscore(s: pd.Series) -> pd.Series:
    s2 = winsorize_mad(s, 5.0)
    med = s2.median()
    mad = (s2 - med).abs().median() * 1.4826
    if mad < 1e-9:
        return pd.Series(np.nan, index=s.index)
    return (s2 - med) / mad


# ---------- 主流程 ----------

def compute(args):
    kline = pd.read_csv(args.kline, parse_dates=["date"])
    kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)
    kline["stock_code"] = kline["stock_code"].astype(str).str.zfill(6)

    # ① daily_r2t = amplitude / turnover
    #    换手率为 0 的情况插一个极小值，避免除零
    kline["turnover_safe"] = kline["turnover"].replace(0, np.nan).fillna(
        kline.groupby("stock_code")["turnover"].transform(
            lambda x: x[x > 0].min() if (x > 0).any() else 0.01
        )
    )
    am = kline["amplitude"].clip(lower=0)
    kline["daily_r2t"] = am / kline["turnover_safe"]

    # ② 个股历史中位数 anchor（60 日滚动）
    kline["hist_med_r2t"] = (
        kline.groupby("stock_code")["daily_r2t"]
        .transform(lambda x: x.rolling(60, min_periods=20).median())
    )

    # ③ 20 日均值
    kline["ma20_r2t"] = (
        kline.groupby("stock_code")["daily_r2t"]
        .transform(lambda x: x.rolling(20, min_periods=10).mean())
    )

    # ④ e_r2t = MA20 - hist_median   （保留差异，而不是百分比变化）
    kline["e_r2t"] = kline["ma20_r2t"] - kline["hist_med_r2t"]

    # ⑤ 以对数成交额代理市值（kline 无 mktcap 列）
    kline["log_amount"] = np.log(kline["amount"].clip(lower=1))

    # ⑥ 截面 OLS 中性化 + MAD z-score
    results = []
    for dt, grp in kline.dropna(subset=["e_r2t", "log_amount"]).groupby("date"):
        if len(grp) < 50:
            continue
        raw = grp["e_r2t"].copy()
        neu = neutralize_cs(raw.reset_index(drop=True),
                            grp["log_amount"].reset_index(drop=True))
        z = robust_zscore(pd.Series(neu, index=grp.index))
        tmp = grp[["date", "stock_code"]].copy()
        tmp["factor"] = z.values
        results.append(tmp.dropna(subset=["factor"]))

    if not results:
        print("ERROR: no factor values produced", file=sys.stderr)
        sys.exit(1)

    out = pd.concat(results, ignore_index=True)
    out = out.sort_values(["date", "stock_code"]).reset_index(drop=True)
    out.to_csv(args.output, index=False)
    print(f"factor rows: {len(out)}  dates: {out['date'].min()} ~ {out['date'].max()}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kline",   default="data/csi1000_kline_raw.csv")
    ap.add_argument("--output",  default="data/factor_turn_efficiency_v1.csv")
    args = ap.parse_args()
    compute(args)


if __name__ == "__main__":
    main()
