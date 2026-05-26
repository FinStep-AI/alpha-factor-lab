"""
factor_pv_divergence_v1.py
==========================
因子：量价背离强度
反向使用：低量价相关系数 → 缩量上涨 / 放量滞涨 → 后续反转 / 趋势延续信号

构造：
  1. volume_chg = volume.pct_change()
  2. rolling_corr_20d = corr(ret, vol_chg, 20d window, min_periods=15)
  3. raw = -rolling_corr_20d   ← 低相关(背离)取正
  4. neutralize(20日均值) against log(MA20(amount))
     → OLS residual → MAD winsorize → z-score
"""
import pandas as pd
import numpy as np
import sys, os

def neutralize(series: pd.Series, neutralizer: pd.Series,
               winsorize_mad: float = 5.2) -> pd.Series:
    """市值/成交额中性化 + MAD缩尾 + z-score，按截面日期逐日做。"""
    df = pd.DataFrame({'x': series, 'n': neutralizer}).dropna()
    out = pd.Series(np.nan, index=series.index)
    for dt, grp in df.groupby(level=0 if df.index.nlevels > 1 else df.index):
        pass

    # 实际处理逻辑：在截面 GroupBy 里操作
    frames = []
    for dt, grp in df.groupby(df.index.get_level_values(0) if df.index.nlevels > 1
                              else df.index):
        if len(grp) < 30:
            continue
        x = grp['x'].values.astype(float)
        n = grp['n'].values.astype(float)
        # MAD winsorize
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-9
        xc = np.clip(x, med - winsorize_mad * mad, med + winsorize_mad * mad)
        # OLS neutralize (n -> x)
        n_ = np.column_stack([np.ones(len(n)), n])
        try:
            beta = np.linalg.lstsq(n_, xc, rcond=None)[0]
            resid = xc - n_ @ beta
        except Exception:
            resid = xc
        # z-score
        mu, sd = resid.mean(), resid.std()
        if sd < 1e-9:
            continue
        z = (resid - mu) / sd
        idx = grp.index
        frames.append(pd.Series(z, index=idx))

    if not frames:
        return series * np.nan
    result = pd.concat(frames)
    result.name = series.name
    return result


def compute_factor(
    kline_path="data/csi1000_kline_raw.csv",
    returns_path="data/csi1000_returns.csv",
    output_path="data/factor_pv_divergence_v1.csv",
    window: int = 20,
):
    print(f"[pv_divergence_v1] loading kline from {kline_path} …")
    df = pd.read_csv(kline_path, usecols=["date","stock_code","open","close",
                                            "high","low","volume","amount","pct_change","turnover"])
    df["date"] = pd.to_datetime(df["date"], format="ISO8601")
    df = df.sort_values(["stock_code", "date"]).drop_duplicates(["stock_code","date"])

    # --- step 1: ret & volume change ---
    df["ret"]      = df["pct_change"] / 100.0
    df["vol_chg"]  = df["volume"].pct_change()

    # --- step 2: rolling 20d corr(ret, vol_chg)  ---
    def rolling_corr(g):
        return g["ret"].rolling(window, min_periods=15).corr(g["vol_chg"])

    df["pv_corr"] = df.groupby("stock_code", group_keys=False).apply(rolling_corr)

    # --- step 3: raw = -pv_corr (低相关→正因子值) ---
    df["factor_raw"] = -df["pv_corr"]

    # --- step 4: neutralizer = log(MA20 amount)  ---
    df["log_amount_20d"] = (
        df.groupby("stock_code")["amount"]
          .transform(lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1))
    )

    # --- step 5: 截面中性化 ---
    df = df.dropna(subset=["factor_raw","log_amount_20d"]).set_index(["date","stock_code"])
    print(f"[pv_divergence_v1] records before neutralization: {len(df)}")

    def neutral_section(frame):
        x  = frame["factor_raw"].values.astype(float)
        nv = frame["log_amount_20d"].values.astype(float)
        # MAD winsorize
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-9
        xc = np.clip(x, med - 5.2 * mad, med + 5.2 * mad)
        # OLS
        X = np.column_stack([np.ones(len(nv)), nv])
        try:
            coef = np.linalg.lstsq(X, xc, rcond=None)[0]
            resid = xc - X @ coef
        except Exception:
            resid = xc
        # z-score
        mu, sd = resid.mean(), resid.std()
        if sd < 1e-9:
            return pd.Series(np.nan, index=frame.index)
        z = (resid - mu) / sd
        return pd.Series(z, index=frame.index, name="factor_value")

    df["factor_value"] = df.groupby(level=0, group_keys=False).apply(neutral_section)

    out = df[["factor_value"]].reset_index().dropna(subset=["factor_value"])
    out = out.rename(columns={"factor_value": "factor_value"})
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[pv_divergence_v1] saved {len(out)} rows → {output_path}")
    print(out["factor_value"].describe())
    return out


if __name__ == "__main__":
    out = compute_factor()
