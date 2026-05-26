"""
factor_turnover_drive_v1.py
============================
因子名：turnover_drive_v1 — "换手率驱动强度（反向）"
方向：反转 — 做多低 ret↔turnover 相关（量价背离大 = 非换手驱动涨 = 信号更强）

构造逻辑（来自本轮挖矿快照探针）:
  1. 对每只股票，每日取 pct_change / 100 作为 ret，turnover 为换手率。
  2. 20d 滚动窗口计算 corr(ret, turnover)，记为 ret_to_corr。
     - 高 ret_to_corr → 换手率上升时价格也在涨/换手率下降时价格也在跌
       → 上涨由换手/流动性驱动，信息含量低。
     - 低 ret_to_corr → 换手与价格方向偏离大
       → 上涨/下跌并非换手推动，更像事件驱动/信息驱动，后续更易延续。
  3. 因子值 = -ret_to_corr（反转方向：低相关 → 高因子值）。
  4. 成交额中性化：OLS 残差 (x ~ log_amount_20d)
     → MAD(5.2σ)缩尾 → z-score。
"""

import numpy as np
import pandas as pd
import os

# ── 配置 ──────────────────────────────────────────────────────────────────
KLINE    = "data/csi1000_kline_raw.csv"
OUT      = "data/factor_turnover_drive_v1.csv"
WINDOW   = 20
MIN_PER  = 15


def _rolling_corr_1d(r: np.ndarray, t: np.ndarray, win: int, mp: int) -> np.ndarray:
    """向量化 Numpy 版本 rolling Pearson corr（双序列同长）"""
    n  = len(r)
    out = np.full(n, np.nan)
    if n < win:
        return out
    r_mean = np.empty(n)
    t_mean = np.empty(n)
    # running mean via cumsum
    rc = np.cumsum(np.insert(r, 0, 0.0))
    tc = np.cumsum(np.insert(t, 0, 0.0))
    r_mean[win-1:] = (rc[win:] - rc[:-win]) / win
    t_mean[win-1:] = (tc[win:] - tc[:-win]) / win
    cov = np.empty(n)
    rs2 = np.empty(n)
    ts2 = np.empty(n)
    for i in range(win-1, n):
        rw = r[i-win+1:i+1]; tw = t[i-win+1:i+1]
        dr = rw - r_mean[i]; dt = tw - t_mean[i]
        cov[i] = (dr*dt).sum() / mp
        rs2[i]  = (dr*dr).sum() / mp
        ts2[i]  = (dt*dt).sum() / mp
    denom = np.sqrt(rs2 * ts2)
    out[win-1:] = np.where(denom < 1e-12, 0.0, cov[win-1:] / denom[win-1:])
    return out


def main():
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)

    print(f"[turnover_drive_v1] loading {KLINE} …")
    df = pd.read_csv(KLINE, usecols=[
        "date","stock_code","volume","amount","pct_change","turnover"
    ])
    df["date"] = pd.to_datetime(df["date"], format="ISO8601")
    df = df.sort_values(["stock_code","date"]).reset_index(drop=True)

    # ── step 1: ret & vol_ma20 ────────────────────────────────────────────
    df["ret"] = df["pct_change"] / 100.0

    vol_ma = (
        df.groupby("stock_code")["volume"]
          .transform(lambda x: x.rolling(WINDOW, min_periods=MIN_PER).mean())
    )

    # step2: simple pandas rolling corr per stock
    corr_list = []
    for code, g in df.groupby("stock_code", sort=False):
        c = g["pct_change"].rolling(WINDOW, min_periods=MIN_PER).corr(g["turnover"])
        for i in range(len(g)):
            if not np.isnan(c.iloc[i]):
                corr_list.append((g["date"].iloc[i], code, -c.iloc[i], vol_ma.iloc[g.index[i]]))
    base = pd.DataFrame(corr_list, columns=["date","stock_code","factor_raw","vol_ma20"])
    base["date"] = pd.to_datetime(base["date"])

    # ── step 3: neutralizer = log(MA20 amount) ───────────────────────────
    amt_ma = (
        df[["date","stock_code","amount"]]
          .groupby("stock_code")
          .apply(lambda g: g.assign(_d=g["date"],
                                    _a=g["amount"].rolling(WINDOW,min_periods=MIN_PER).mean()))
          .reset_index(drop=True)
    )
    amt_ma["log_amount_20d"] = np.log(amt_ma["_a"].clip(lower=1))
    amt_ma = amt_ma[["date","stock_code","log_amount_20d"]].drop_duplicates(["date","stock_code"])

    base = base.merge(amt_ma, on=["date","stock_code"], how="inner")
    base = base.dropna(subset=["factor_raw","log_amount_20d"])
    print(f"[turnover_drive_v1] records after merge: {len(base)}")

    # ── step 4: cross-section neutralization + z-score ───────────────────
    frames = []
    for dt, sec in base.groupby("date"):
        x  = sec["factor_raw"].values.astype(np.float64)
        nv = sec["log_amount_20d"].values.astype(np.float64)
        if len(sec) < 50:
            continue
        med = np.median(x); mad = np.median(np.abs(x - med)) + 1e-9
        x = np.clip(x, med - 5.2*mad, med + 5.2*mad)
        X = np.column_stack([np.ones(len(nv)), nv])
        coef = np.linalg.lstsq(X, x, rcond=None)[0]
        resid = x - X @ coef
        mu, sd = resid.mean(), resid.std()
        if sd < 1e-9:
            continue
        z = (resid - mu) / sd
        sec2 = sec[["date","stock_code"]].copy()
        sec2["factor_value"] = z
        frames.append(sec2)

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"[turnover_drive_v1] saved {len(out)} rows → {OUT}")
    print(out["factor_value"].describe())


if __name__ == "__main__":
    main()
