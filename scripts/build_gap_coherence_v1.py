#!/usr/bin/env python3
"""
因子: 缺口同向一致性 (Gap Coherence) v1
factor_id: gap_coherence_v1

逻辑（全新方向，与 gap_momentum/gap_fill 均不重复）:
  1. gap_ret   = open / prev_close - 1          （隔夜跳空幅度）
  2. ret_body  = close / open - 1               （日内实体涨跌）
  3. gap_range = (high - low) / prev_close       （日内波动空间锚定前收）
  4. 当日 coherent_sign = sign(gap_ret) * sign(ret_body)
     >0: 日内实体方向与隔夜缺口同向（缺口当天"被守住/延续"）
     =1 且 gap_ret 超过当日波动空间一半(pierce_gap=1): 缺口两段方向完全被日内实体替代穿过，是最强的一致性形式
  5. daily_coherence = coherent_sign * min(1, |gap_ret| / gap_range)
     （方向一致 × 缺口幅度占日内波动空间的占比；cap=1防止缺口覆盖全波幅的极端噪声）
  6. factor_raw = MA20(daily_coherence)，成交额OLS中性化 + 3σ MAD缩尾 + z-score

假设:
  - A股中证1000小盘股的跳空缺口（尤其是低开的逆向反转）会维持多久？
  - 隔夜跳空并不一定被日内完全回补，如果日内实体方向仍然与缺口方向一致，说明该股价格承载力是
    有方向指向性的。这种情况下，该股在随后会延续穿缺口方向的概率更大。
  - 换句话说：缺口不是"被回补"，而是"被守住"——守住缺口方向=有知情交易者方向定住=后续趋势延续。

与 gap_momentum 的区别: gap_momentum 看缺口幅度和连续一致性；因子看的是 "持仓切实固在缺口方向上的定量"——本质方向与缺口一致性，和缺口的补位是截然不同的对称分叉。
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

WINDOW = 20
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KLINE = os.path.join(BASE, "data", "csi1000_kline_raw.csv")
OUT   = os.path.join(BASE, "data", "factor_gap_coherence_v1.csv")

def neutralize_cs(group):
    y = group["factor_raw"].values.astype(float).copy()
    x = group["log_amount_20d"].values.astype(float).copy()
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 40:
        return pd.Series(np.nan, index=group.index)
    yv, xv = y[ok], x[ok]
    # MAD winsorize
    med_y, mad_y = np.median(yv), np.median(np.abs(yv - np.median(yv))) * 1.4826
    if mad_y > 0:
        yv = np.clip(yv, med_y - 3 * mad_y, med_y + 3 * mad_y)
    X = np.column_stack([np.ones(len(xv)), xv])
    b = np.linalg.lstsq(X, yv, rcond=None)[0]
    resid = yv - X @ b
    mu, sd = resid.mean(), resid.std()
    if sd < 1e-9:
        return pd.Series(np.nan, index=group.index)
    z = (resid - mu) / sd
    out = np.full(len(y), np.nan)
    out[ok] = z
    return pd.Series(out, index=group.index)

def main():
    print(f"[1] load: {KLINE}")
    df = pd.read_csv(KLINE, dtype={"stock_code": str})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    df["stock_code"] = df["stock_code"].str.zfill(6)

    g = df.groupby("stock_code", sort=False)

    # prev close for computing gaps
    df["prev_close"] = g["close"].shift(1)
    df["gap_ret"]    = df["open"] / df["prev_close"] - 1
    df["ret_body"]   = df["close"] / df["open"] - 1
    df["gap_range"]  = (df["high"] - df["low"]) / df["prev_close"]

    # do not use when prev close missing or gap/high-low zero
    valid = df["prev_close"].notna() & (df["prev_close"] > 0) & \
            df["gap_range"].notna() & (df["gap_range"] > 0)

    coh = np.where(valid,
                   np.sign(df["gap_ret"]) * np.sign(df["ret_body"]) *
                   np.minimum(1.0, np.abs(df["gap_ret"]) / df["gap_range"]),
                   np.nan)

    df["daily_coh"] = coh

    # 20-day rolling mean of daily coherence
    df["factor_raw"] = g["daily_coh"].transform(
        lambda s: s.rolling(WINDOW, min_periods=14).mean()
    )
    df["mean_amt_20d"] = g["amount"].transform(
        lambda s: s.rolling(WINDOW, min_periods=16).mean()
    )
    df["log_amount_20d"] = np.log(df["mean_amt_20d"].clip(lower=1))

    fdf = df[["date", "stock_code", "factor_raw", "log_amount_20d"]].dropna().copy()
    print(f"  raw rows: {len(fdf)}")

    print("[2] cross-sectional neutralize …")
    fdf["factor"] = (
        fdf.groupby("date", group_keys=False)
           .apply(neutralize_cs)
    ).values

    out = (fdf[["date", "stock_code", "factor"]]
           .dropna(subset=["factor"])
           .sort_values(["date", "stock_code"]))
    out.to_csv(OUT, index=False)
    print(f"[3] saved {OUT}  rows={len(out)}  "
          f"{out['date'].min().date()}~{out['date'].max().date()}  "
          f"stocks/date ≈ {out.groupby('date')['stock_code'].count().mean():.0f}")

    # quick stats
    mu = out["factor"].mean()
    sd = out["factor"].std()
    print(f"    factor mean={mu:+.4f}  std={sd:.4f}")

if __name__ == "__main__":
    main()
