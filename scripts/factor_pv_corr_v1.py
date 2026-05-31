"""
pv_corr_v1 — 量价相关性背离因子
---------------------------------
构造：20日滚动窗口内，个股日收益率与日成交量变化率的Pearson相关系数。
      全市场截面 rank + 成交额 OLS 中性化 + MAD 缩尾 + z-score。

逻辑：corr(ret, Δvol) < 0  量价背离（放量跌 / 缩量涨）→ 信息未充分反映 → 后续反转修复？
      corr(ret, Δvol) > 0  量价同向（放量涨）→ 动量延续

文献参考：
  - Chordia & Subrahmanyam (2004) Order Imbalance and Individual Stock Returns
  - 方正金工《量价互动的选股因子2》(2018)
  - BigQuant《量价结合因子》
"""

import numpy as np
import pandas as pd
import sys, os, json, warnings
warnings.filterwarnings("ignore")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_SCRIPT_DIR)
KLINE = os.path.join(_ROOT, "data", "csi1000_kline_raw.csv")
OUT   = os.path.join(_ROOT, "data", "factor_pv_corr_v1.csv")
WINDOW = 20          # rolling corr window
NEUT_WINDOW = 20     # amount neutralization window
MAD_K   = 5.2        # MAD multiplier
TOP_FRAC = 0.05      # top / bottom MAD fraction


def mad_winsorize(s: pd.Series, k: float = MAD_K) -> pd.Series:
    med = s.median()
    dev = (s - med).abs().median() * 1.4826
    if dev == 0 or np.isnan(dev):
        return s
    lo, hi = med - k * dev, med + k * dev
    return s.clip(lo, hi)


def load_kline(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["date", "stock_code", "pct_change", "turnover"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    # 日收益率 = pct_change / 100
    df["ret"] = df["pct_change"] / 100.0

    # 成交量变化率：log(vol_t / vol_{t-1})
    df["vol_raw"] = df["volume"].replace(0, np.nan)
    df["log_vol"] = np.log(df["vol_raw"])
    df["vol_chg"] = df.groupby("stock_code")["log_vol"].diff()

    # 成交额 20 日均值（用于中性化）
    df["log_amount"] = np.log(df["amount"].replace(0, np.nan))
    df["log_amount_20d"] = df.groupby("stock_code")["log_amount"].transform(
        lambda x: x.rolling(NEUT_WINDOW, min_periods=10).mean()
    )

    # 换手率去零（用于相关性计算时 dropna）
    df["turnover_c"] = df["turnover"].replace(0, np.nan)

    return df


def rolling_corr(group: pd.DataFrame) -> pd.Series:
    """对单个 stock 计算滚动 corr(ret, vol_chg)。"""
    g = group.sort_values("date").copy()
    r = g["ret"].rolling(WINDOW, min_periods=int(WINDOW * 0.7)).corr(g["vol_chg"])
    return pd.Series(r.values, index=g["date"])


def compute_factor(df: pd.DataFrame) -> pd.Series:
    print("  computing rolling corr(ret, vol_chg) …")
    corr_list = []
    for code, grp in df.groupby("stock_code", sort=False):
        c = rolling_corr(grp)
        c.index = pd.MultiIndex.from_product([[code], c.index], names=["stock_code", "date"])
        corr_list.append(c)
    raw = pd.concat(corr_list).rename("raw")
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    return raw


def cross_section_neutralize(daily: pd.DataFrame) -> pd.Series:
    """截面 OLS 中性化 + MAD 缩尾 + z-score，返回 factor Series。"""
    vals = []
    for dt, sub in daily.groupby("date", sort=True):
        x = sub["log_amount_20d"].values
        y = sub["raw"].values
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 30:
            continue
        x_m, y_m = x[m], y[m]
        X = np.column_stack([np.ones_like(x_m), x_m])
        try:
            beta = np.linalg.lstsq(X, y_m, rcond=None)[0]
            resid = np.full(len(x), np.nan)
            resid[m] = y_m - X @ beta
        except Exception:
            continue

        out = pd.Series(resid, index=sub.index)
        out = mad_winsorize(out)
        mu, std = out.mean(), out.std(ddof=0)
        if std == 0 or np.isnan(std):
            continue
        out = (out - mu) / std
        vals.append(out)

    if not vals:
        return pd.Series(dtype=float)
    factor = pd.concat(vals)
    factor.name = "factor"
    return factor


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    print("[1/3] loading kline …")
    df = load_kline(KLINE)

    print("[2/3] computing raw factor …")
    raw = compute_factor(df)

    print("[3/3] cross-section neutralize …")
    # 先把 raw merge 回 df（只保留有 raw 的行）
    df2 = df.set_index(["stock_code", "date"])
    raw.index.names = ["stock_code", "date"]
    df2 = df2.join(raw, how="inner")

    factor_list = []
    for dt, sub in df2.groupby("date", sort=True):
        f = cross_section_neutralize(sub[["log_amount_20d", "raw"]])
        if f.empty:
            continue
        for idx, val in f.items():
            factor_list.append({"stock_code": idx[0], "date": idx[1], "factor": val})

    out_df = pd.DataFrame(factor_list)
    out_df = out_df.sort_values(["date", "stock_code"]).reset_index(drop=True)
    out_df.to_csv(OUT, index=False)
    print(f"  saved {OUT}  rows={len(out_df)}  dates={out_df['date'].nunique()}")


if __name__ == "__main__":
    main()
