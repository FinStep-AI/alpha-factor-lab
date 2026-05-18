"""
amplitude_concentration_v1.py
====================================
因子：日内振幅集中度 (Amplitude Concentration)

逻辑：
  20日滚动窗口 → 每日计算 (high-low)/prev_close（振幅）
  同时按振幅将交易日分为低/中/高三档
  → 计算各档的 成交额占比
  → 赫芬达尔指数 = Σ(各档占比²)

解释：
  值域 [1/3 ~ 1.0]
  HHI=1.0    → 成交额集中在某一档（极端聚焦）
  HHI=0.33   → 三档均匀分布（无集中）

信心驱动：高振幅日内成交额占比高 → 显性市场事件/信息驱动 →
知情交易参与度高 → 价格发现有序 → 后续正alpha（正向）

处理：成交额OLS中性化 + MAD缩尾5σ + z-score
"""

import pandas as pd
import numpy as np

KLINE_PATH = "data/csi1000_kline_raw.csv"
OUTPUT_PATH = "data/factor_amp_conc_v1.csv"
LOOKBACK = 20


def compute_amplitude(stock_df: pd.DataFrame) -> pd.Series:
    """振幅 = (high - low) / prev_close（日频）"""
    amp = (stock_df["high"] - stock_df["low"]) / stock_df["close"].shift(1)
    return amp.fillna(0)


def compute_concentration(amp: np.ndarray, amt: np.ndarray) -> float:
    """20日窗口内振幅集中度（HHI）."""
    total = amt.sum()
    if total <= 0:
        return np.nan

    q33, q67 = np.quantile(amp, [0.33, 0.67])
    low_amt = amt[amp <= q33].sum()
    mid_amt = amt[(amp > q33) & (amp <= q67)].sum()
    high_amt = amt[amp > q67].sum()

    w3 = [low_amt / total, mid_amt / total, high_amt / total]
    return sum(w ** 2 for w in w3)  # 0.333 ~ 1.0


def main():
    # ---- 读数据 ----
    kline = pd.read_csv(KLINE_PATH, parse_dates=["date"])
    # 截断到一致的安全截止日（多数因子用2026-03-13）
    cutoff = pd.Timestamp("2026-03-13")
    kline = kline[kline["date"] <= cutoff].copy()
    kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)

    print(f"加载 {len(kline)} 行, {kline.stock_code.nunique()} 只, 截止 {kline.date.max().date()}")

    # ---- 日内振幅 ----
    kline["amplitude_daily"] = kline.groupby("stock_code").apply(
        lambda df: compute_amplitude(df)
    ).reset_index(level=0, drop=True)

    # ---- 滚动集中度 ----
    raw_factor = []
    for code, grp in kline.groupby("stock_code", sort=False):
        grp = grp.sort_values("date")
        amp = grp["amplitude_daily"].values
        amt = grp["amount"].values
        idx = grp.index

        fvals = np.full(len(grp), np.nan)
        for i in range(LOOKBACK, len(grp)):
            fvals[i] = compute_concentration(amp[i - LOOKBACK : i], amt[i - LOOKBACK : i])

        raw_factor.append(pd.Series(fvals, index=idx))

    raw_factor = pd.concat(raw_factor).sort_index()
    assert len(raw_factor) == len(kline)

    kline["raw_factor"] = raw_factor.values

    # ---- 截面中性化：成交额OLS残差 ----
    from numpy.linalg import lstsq

    def neutralizer(group):
        mask = group["raw_factor"].notna() & group["amount_20d"].notna() & (group["amount_20d"] > 0)
        if mask.sum() < 30:
            group["neutralized"] = np.nan
            return group

        x = np.log(group.loc[mask, "amount_20d"].values)
        y = group.loc[mask, "raw_factor"].values
        x = np.column_stack([np.ones(len(x)), x])
        beta, _, _, _ = lstsq(x, y, rcond=None)
        resid = np.full(len(group), np.nan)
        resid[mask] = y - x @ beta
        group["neutralized"] = resid

        # ---- MAD缩尾 + z-score ----
        med = np.nanmedian(resid)
        mad = np.nanmedian(np.abs(resid - med))
        if mad < 1e-12:
            group["factor_value"] = np.where(np.isnan(resid), np.nan, 0.0)
            return group
        upper = med + 5.5 * 1.4826 * mad
        lower = med - 5.5 * 1.4826 * mad
        clipped = np.clip(resid, lower, upper)
        mu, sigma = np.nanmean(clipped), np.nanstd(clipped)
        if sigma < 1e-12:
            group["factor_value"] = 0.0
        else:
            group["factor_value"] = (clipped - mu) / sigma

        return group

    # 20日平均成交额
    kline["amount_20d"] = kline.groupby("stock_code")["amount"].transform(
        lambda x: x.rolling(LOOKBACK, min_periods=10).mean()
    )

    kline = kline.groupby("date", group_keys=False).apply(neutralizer)

    # ---- 输出 ----
    out = kline[["date", "stock_code", "factor_value"]].dropna(subset=["factor_value"])
    out = out.rename(columns={"factor_value": "factor_value"})
    out[["date", "stock_code", "factor_value"]].to_csv(OUTPUT_PATH, index=False)

    print(
        f"输出 {OUTPUT_PATH}: {len(out)} 行, "
        f"{out.stock_code.nunique()} 只, "
        f"日期 {out.date.min().date()} ~ {out.date.max().date()}"
    )
    print(f"\n统计:")
    print(out['factor_value'].describe())


if __name__ == "__main__":
    main()
