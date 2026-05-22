#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: 量能领先收益滞后相关 (Volume→Return Lead-Lag Cross-Correlation)
ID: vol_turn_lead_corr_20d_v1

逻辑:
  参考中金 (2022)《量化多因子系列（7）：价量因子手册》量能领先因子(corr_ret_turn_post_1M)
  与 中信建投 (2025)《逐鹿Alpha专题报告（二十九）——隔夜-日内异象因子》领先-滞后框架。

  核心思想：A股量能领先价格（非同步），T-1日换手率异常高/低可以预测 T 日收益方向。
  不是当日的量价共动，而是显式隔一日的 lead-lag：

     lag_input[i]  = z-score(turnover_{i-1}  vs 自身过去20日均值)    # 昨日换手率异常度
     label[i]     = close_i / close_{i-1} - 1                         # 今日隔夜收益

     在连续 20 个交易日窗口里，对 (lag_input, label) 做截面 Pearson 相关，
     取该滚动值的当日截面值，再成交额OLS中性化。

  高因子值 = 过去20天里"换手率高→次日赚钱"这种量能领先关系一直很强 → 信息流持续领先价格

历史背景:
  - 中金手册: corr_ret_turn_post_1M 全市场 IC_IR=0.52，CSI500也有不小选股力
  - 中证1000内, 换手率因子 IC_IR=-0.81, 反转因子 IC_IR=-0.78 (2022中金价量手册)
  - A股"隔夜负收益" + "量能先于价格" 是被多篇论文验证的微观结构异象

Barra风格: Momentum / MICRO
中性化: log(amount_20d) OLS + MAD Winsorize + z-score
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import lstsq

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def neutralize_ols(value: pd.Series, control: pd.Series) -> pd.Series:
    """成交额OLS中性化 + MAD Winsorize + z-score，返回同索引Series."""
    df = pd.DataFrame({"v": value, "c": control}).dropna()
    if len(df) < 30:
        return pd.Series(np.nan, index=value.index)
    x = df["c"].values.astype(float)
    y = df["v"].values.astype(float)
    X = np.column_stack([np.ones(len(x)), x])
    try:
        b, _, _, _ = lstsq(X, y, rcond=None)
        r = y - X @ b
    except Exception:
        return pd.Series(np.nan, index=value.index)
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    if mad < 1e-10:
        return pd.Series(0.0, index=value.index)
    r = np.clip(r, med - 5.2 * mad, med + 5.2 * mad)
    std = r.std()
    if std < 1e-10:
        return pd.Series(0.0, index=value.index)
    out = pd.Series(np.nan, index=value.index, dtype=float)
    out.loc[df.index] = (r - med) / std
    return out


# ── 主计算 ────────────────────────────────────────────────────────────────────

def calc_vol_turn_lead_corr(
    kline_path: str,
    output_path: str,
    lookback: int = 20,
    min_periods: int = 12,
    amount_win: int = 20,
) -> pd.DataFrame:
    """计算 vol_turn_lead_corr_20d_v1 因子."""

    print(f"[vol_turn_lead_corr] Loading {kline_path}")
    df = pd.read_csv(kline_path, usecols=["date", "stock_code", "close", "turnover", "amount"])
    df["date"] = pd.to_datetime(df["date"].astype(str))
    for col in ["close", "turnover", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["close"] > 0) & (df["turnover"] >= 0)].copy()
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    # ① 隔夜日收益率: close_t / close_{t-1} - 1
    df["ret_cc"] = df.groupby("stock_code")["close"].pct_change()

    # ② lag-1 换手率（T日对应的昨日换手率 T-1 turnover）
    df["turnover_lag1"] = df.groupby("stock_code")["turnover"].shift(1)

    # ③ 昨日换手率的 20 日均值与标准差（截面外实测值，用在同一日的股票间比较）
    df["turn_ma20"] = df.groupby("stock_code")["turnover_lag1"]\
        .transform(lambda s: s.rolling(lookback, min_periods=min_periods).mean())
    df["turn_std20"] = df.groupby("stock_code")["turnover_lag1"]\
        .transform(lambda s: s.rolling(lookback, min_periods=min_periods).std())

    # ④ lag-z-turnover: (turn{T-1} - MA20{T-1}) / std20{T-1}
    df["lag_turn_z"] = (df["turnover_lag1"] - df["turn_ma20"]) / df["turn_std20"].replace(0, np.nan)
    df["lag_turn_z"] = df["lag_turn_z"].clip(-10, 10)

    # ⑤ 成交额中性化控制变量
    df["log_amount_20d"] = np.log(
        df.groupby("stock_code")["amount"]
        .transform(lambda s: s.rolling(amount_win, min_periods=10).mean().clip(lower=1))
    )

    # ⑥ 逐日截面：先在每个股票时序上滚动计算 20 日滚动相关系数，
    #    再在截面上对 (lag_turn_z, ret_cc) 方向做OLS中性化
    print(f"[vol_turn_lead_corr] Computing rolling corr(lag_turn_z, ret_cc) window={lookback} ...")

    def _rolling_corr_20d(group: pd.DataFrame) -> pd.Series:
        """单只股票：对 20d 内 (lag_turn_z, ret_cc) 滑动窗口算 Pearson r."""
        z = group["lag_turn_z"].values.astype(float)
        r = group["ret_cc"].values.astype(float)
        n = len(group)
        out = np.full(n, np.nan)
        half_w = lookback // 2
        for i in range(half_w, n):
            a = z[max(0, i - half_w): i + 1]
            b = r[max(0, i - half_w): i + 1]
            # ensure vector length == `lookback`
            if len(a) < min_periods:
                continue
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < min_periods:
                continue
            aa, bb = a[mask], b[mask]
            if aa.std() < 1e-10 or bb.std() < 1e-10:
                continue
            out[i] = np.corrcoef(aa, bb)[0, 1]
        return pd.Series(out, index=group.index)

    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    df["raw_corr"] = (
        df.groupby("stock_code", group_keys=False)
        .apply(_rolling_corr_20d)
        .values
    )
    df["raw_corr"] = df["raw_corr"].clip(-1, 1)

    # ⑦ 截面上成交额中性化
    print("[vol_turn_lead_corr] Cross-sectional OLS neutralization ...")
    records = []
    for dt, grp in df.groupby("date", sort=True):
        y = grp["raw_corr"].values
        x = grp["log_amount_20d"].values
        m = np.isfinite(y) & np.isfinite(x)
        if m.sum() < 30:
            continue
        z = neutralize_ols(pd.Series(y, index=grp.index), pd.Series(x, index=grp.index))
        z = z.dropna()
        if len(z) == 0:
            continue
        sub = grp.loc[z.index, ["stock_code", "date"]].copy()
        sub["factor_value"] = z.values
        records.append(sub)

    result = pd.concat(records, ignore_index=True).drop_duplicates(["stock_code", "date"])
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"[vol_turn_lead_corr] Saved {len(result)} rows  dates={result['date'].min()}~{result['date'].max()}  -> {out_path}")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="量能领先收益滞后相关因子 vol_turn_lead_corr_20d_v1")
    ap.add_argument("--kline", default="data/csi1000_kline_raw.csv", help="K线CSV")
    ap.add_argument("--output", default="data/factor_vol_turn_lead_corr_20d.csv", help="因子输出")
    ap.add_argument("--lookback", type=int, default=20, help="滚动相关窗口")
    ap.add_argument("--amount-win", type=int, default=20, help="成交额中性化窗口")
    args = ap.parse_args()
    r = calc_vol_turn_lead_corr(args.kline, args.output,
                                lookback=args.lookback, amount_win=args.amount_win)
    print(r["factor_value"].describe().to_string())
