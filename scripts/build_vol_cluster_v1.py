#!/usr/bin/env python3
"""
vol_cluster_v1 — 波动率持续性因子
================================
公式: autocorr_1(|ret|², 20d)  ← 20日滚动收益率平方的一阶自相关
逻辑: 高自相关 = 大波动之后大波动持续（波动率持久/聚集）→ 波动率风险溢价补偿
      低自相关 = 大事件是偶发的 → 持续性风险低 → 无溢价
中性化: 成交额OLS中性化 + MAD缩尾 + z-score

Barra style: Volatility
"""
import numpy as np
import pandas as pd
import json, sys, os

WORK = "/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab"

def _ols_residual(y, X):
    """numpy OLS 残差（含 1e-6 正则化防奇异）"""
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < X.shape[1] + 2:
        return y
    y_c, X_c = y[mask], X[mask]
    try:
        XtX = X_c.T @ X_c; Xty = X_c.T @ y_c
        beta = np.linalg.solve(XtX + 1e-6 * np.eye(XtX.shape[0]), Xty)
        r = np.full_like(y, np.nan, dtype=float)
        r[mask] = y_c - X_c @ beta
        return r
    except np.linalg.LinAlgError:
        return y

def rolling_autocorr1(series: pd.Series, window: int) -> pd.Series:
    """groupby stock_code 内滚动 lag-1 自相关"""
    def _ac1(x):
        y = x.values.astype(float)
        out = np.full(len(y), np.nan)
        for i in range(window, len(y) + 1):
            seg = y[i-window:i]
            if np.sum(np.isfinite(seg)) < window:
                continue
            m = seg.mean()
            d = seg - m
            denom = np.sqrt((d[:-1]**2).sum() * (d[1:]**2).sum())
            if denom == 0:
                continue
            out[i-1] = (d[:-1] @ d[1:]) / denom
        return pd.Series(out, index=x.index)
    return series.groupby(level=0, group_keys=False).apply(_ac1)

def main():
    # 1. 读取原始 K 线 + 收益率
    kline = pd.read_csv(f"{WORK}/data/csi1000_kline_raw.csv",
                        parse_dates=["date"], low_memory=False)
    kline = kline.sort_values(["stock_code", "date"]).set_index(["stock_code", "date"])

    ret = pd.read_csv(f"{WORK}/data/csi1000_returns.csv",
                      parse_dates=["date"], low_memory=False)
    ret = ret.sort_values(["stock_code", "date"]).set_index(["stock_code", "date"])

    print(f"[info] kline rows={len(kline)}, ret rows={len(ret)}")

    # 2. 计算 |ret|²
    ret_s = ret["return"].astype(float)
    sq_ret = (ret_s.abs() ** 2)
    sq_ret.name = "sq_ret"

    # 3. 20日滚动 lag-1 自相关
    print("[info] computing rolling autocorr_1(|ret|^2, 20d)…")
    vol_cluster_raw = rolling_autocorr1(sq_ret, 20)
    vol_cluster_raw.name = "vol_cluster_raw"
    print(f"[info] vol_cluster_raw stats:\n{vol_cluster_raw.describe()}")

    # 4. 合并到日期×stock 的 DataFrame
    df = vol_cluster_raw.reset_index()
    df = df.merge(kline[["amount"]].reset_index(), on=["stock_code", "date"], how="left")

    # 成交额 20d 均值（对数）做中性化基准
    amt_ma20 = (df.groupby("stock_code")["amount"]
                  .transform(lambda x: x.rolling(20, min_periods=10).mean()))
    df["log_amount_20d"] = np.log(amt_ma20.clip(lower=1))

    # 5. 截面 OLS 中性化 + MAD 缩尾 + z-score  ← 逐行写，与 factor_calculator.py 一致
    neutralized = np.full(len(df), np.nan, dtype=float)
    for dt, g_idx in df.groupby("date").groups.items():
        gi = list(g_idx)
        y  = df.loc[gi, "vol_cluster_raw"].values.astype(float)
        xv = df.loc[gi, "log_amount_20d"].values.astype(float)
        ok = np.isfinite(y) & np.isfinite(xv)
        if ok.sum() < 5:
            continue
        X = np.column_stack([np.ones(ok.sum()), xv[ok]])
        res = _ols_residual(y[ok], X)
        neutralized[gi] = res          # 已在 _ols_residual 里写回全长位置 (残差外围 NaN)

    df["factor_neutral"] = neutralized

    # MAD 缩尾 5.2σ
    cols = []
    for dt, g_idx in df.groupby("date").groups.items():
        gi = list(g_idx)
        s = pd.Series(df.loc[gi, "factor_neutral"].values, index=gi)
        med = s.median()
        mad = (s - med).abs().median()
        if mad > 0:
            scaled_mad = 1.4826 * mad
            lo, hi = med - 5.2 * scaled_mad, med + 5.2 * scaled_mad
            df.loc[gi, "factor_neutral"] = s.clip(lo, hi).values
            cols.append(dt)

    # z-score
    z = np.full(len(df), np.nan, dtype=float)
    for dt, g_idx in df.groupby("date").groups.items():
        gi = list(g_idx)
        vals = df.loc[gi, "factor_neutral"].values.astype(float)
        ok = np.isfinite(vals)
        if ok.sum() < 3: continue
        v = vals[ok]; mu, sd = v.mean(), v.std()
        if sd > 0:
            tmp = np.full(len(vals), np.nan); tmp[ok] = (v - mu) / sd
            z[gi] = tmp
    df["vol_cluster_v1"] = z

    # 6. 输出
    out = df[["date", "stock_code", "vol_cluster_v1"]].dropna(subset=["vol_cluster_v1"])
    out_path = f"{WORK}/data/factor_vol_cluster_v1.csv"
    out.to_csv(out_path, index=False)
    print(f"\n[OK] saved {len(out):,} rows → {out_path}")
    print(f"[OK] date range: {out['date'].min()} ~ {out['date'].max()}")
    print(f"[OK] vol_cluster_v1 stats:\n{out['vol_cluster_v1'].describe()}")

    # 7. 写一个轻量 metadata
    meta = {
        "factor_id": "vol_cluster_v1",
        "name": "波动率持续性 v1",
        "name_en": "Volatility Clustering v1",
        "category": "波动率",
        "description": "20日滚动 |ret|² 的 lag-1 自相关，成交额OLS中性化+MAD缩尾+z-score。高=波动率持续集聚，低=大波动孤例性事件。Ang et al. (2006) 低波异象在高波溢价A股中反转为波动率聚集风险补偿思路。",
        "hypothesis": "波动率聚集越强→持续性风险越高→高波动率风险溢价补偿（A股中证1000为高波溢价市场）",
        "formula": "autocorr_1(|ret|^2, 20d)",
        "direction": 1,
        "barra_style": "Volatility",
        "source_type": "学术文献启发",
        "source_title": "波动率聚集/波动率风险溢价",
        "data_file": "data/factor_vol_cluster_v1.csv"
    }
    with open(f"{WORK}/data/factor_vol_cluster_v1_meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] meta → data/factor_vol_cluster_v1_meta.json")

if __name__ == "__main__":
    main()
