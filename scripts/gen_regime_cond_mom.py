#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子：regime_conditional_mom_v1
ID: regime_conditional_mom_v1

构造
----
最近5日累计收益 × 过去20日振幅水平 的交互项。

  regime_signal = MA20(amplitude)            # 当前波动状态
  ret_20d         = cumret(20d前 → 5d前)     # 中期趋势
  raw = ret_20d × regime_signal

  对数成交额OLS中性化 + MAD缩尾 + z-score

逻辑
----
在中证1000中低振幅=稳趋势=动量延续；高振幅=震荡=动量衰减/反转。
我们用 return * regime_signal 作为代理坡度：
  • 稳市场里近期净涨 → 高raw → 趋势依从
  • 稳市场里近期净跌 → 低raw → 避开
  • 高波震荡区间，raw的选股效果会自然收敛（被 regime 「抵消」）

Barra: Momentum（波动状态调节方向）
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

KLINE = "data/csi1000_kline_raw.csv"
OUT   = "data/factor_regime_cond_mom_v1.csv"

# ── 1. 读数据 ────────────────────────────────────────────────────
print("[1] 加载 …")
df = pd.read_csv(KLINE,
                 usecols=["date","stock_code","open","close","high","low","amount","amplitude"])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code","date"]).reset_index(drop=True)
df["prev_close"] = df.groupby("stock_code")["close"].shift(1)

# 振幅：补缺失
miss = df["amplitude"].isna() | (df["amplitude"] == 0)
df.loc[miss, "amplitude"] = ((df.loc[miss,"high"] - df.loc[miss,"low"]) /
                               df.loc[miss,"prev_close"]).clip(0, 1)

# ── 2. 滚动特征 ──────────────────────────────────────────────────
print("[2] 计算滚动特征 …")
df = df.sort_values(["stock_code","date"])

# ret_1d
df["ret_1d"] = df.groupby("stock_code")["close"].pct_change()

# MA20 振幅（波动状态）
df["amp_20d"]   = df.groupby("stock_code")["amplitude"].transform(
    lambda x: x.rolling(20, min_periods=15).mean())
# cumret(总43d: 20d lookback + 20d cumret + 5d fwd需要前推3d避免数据泄露)
# 用 date_index 对齐 cumret = prod(1+ret) over lag[1..20]
df["cumret_20d"] = df.groupby("stock_code")["ret_1d"].transform(
    lambda x: x.shift(3).rolling(20, min_periods=18).apply(lambda w: np.prod(1+w)-1, raw=True))

# log(1+MA20_amplitude) 避免数值过大
df["log_amp"]  = np.log1p(df["amp_20d"])
df["log_amt20"] = np.log(
    df.groupby("stock_code")["amount"]
      .transform(lambda x: x.rolling(20, min_periods=15).mean()) + 1)

# ── 3. 交互因子 & 单变量OLS中性化 ────────────────────────────────
print("[3] 交互 & 中性化 …")
df["factor_raw"] = df["cumret_20d"] * df["log_amp"]

def neutralize(g):
    y = g["factor_raw"].values.astype(float)
    x = g["log_amt20"].values.astype(float)
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 30:
        return pd.Series(np.nan, index=g.index)
    X = np.column_stack([np.ones(m.sum()), x[m]])
    try:
        b = np.linalg.lstsq(X, y[m], rcond=None)[0]
        r = np.empty(len(y), float); r[:] = np.nan
        r[m] = y[m] - X @ b
    except Exception:
        r = y.copy()
    # MAD
    med = np.nanmedian(r)
    mad = np.nanmedian(np.abs(r - med)) + 1e-8
    k   = 5.2 * 1.4826
    r   = np.clip(r, med - k * mad, med + k * mad)
    # z-score per group
    mu = np.nanmean(r);  sg = np.nanstd(r) + 1e-8
    return pd.Series((r - mu) / sg, index=g.index)

df["factor_value"] = df.groupby("date", group_keys=False).apply(neutralize)
out = df[["date","stock_code","factor_value"]].dropna(subset=["factor_value"])
out["date"] = out["date"].dt.strftime("%Y-%m-%d")
out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
out.to_csv(OUT, index=False)

# ── 4. 晴雨 ──────────────────────────────────────────────────────
mask = np.isfinite(df["factor_raw"].values)
print(f"[完成] {OUT}")
print(f"      raw mean={np.nanmean(df['factor_raw']):.6f}  "
      f"std={np.nanstd(df['factor_raw']):.6f}")
print(f"      行 {len(out)}  开始 {out['date'].min()}  结束 {out['date'].max()}")
