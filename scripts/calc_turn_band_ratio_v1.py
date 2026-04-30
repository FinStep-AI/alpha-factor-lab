#!/usr/bin/env python3
"""
因子：turn_band_ratio_v1 — Turnover-Band Efficiency Ratio（换手率-波段效率比）
目标 Barra: Quality / MICRO  
来源：自研 — 振幅/换手 反向差值  
参考: Stoll & Whaley (1989) 换手率, Amihud (2002) 流动性

逻辑：
  - 用过去 20 日的 (close-open)/close 作为日内波动代理
  - 因子 = -[MA20(振幅) / MA20(换手率)]
  - 负号：振幅小、换手率高 = 价格稳定但关注度高 = 市场质量佳
  - NA/0 换手 → 剔除
  - 成交额 log neutralization（OLS回归取残差）
  - MAD winsorize 3σ → log1p修正右偏 → z-score
"""

import numpy as np
import pandas as pd

print("[TBR v1] Computing Turnover-Band Efficiency Ratio...")

# ── Load data ────────────────────────────────────────────
kline = pd.read_csv("data/csi1000_kline_raw.csv")
kline["date"] = pd.to_datetime(kline["date"])
kline = kline[kline["date"] <= "2026-04-24"].copy()

# ── Derive close-to-open amplitude ───────────────────────
# kline.amplitude = (high-low)/prev_close * 100 → 以小数表示
# 但我们还可以算：(close-open)/close 的 20d mean std
# 直接用 amplitude 列进行换手率约简版，价格波动*成交额

gb = kline.groupby("stock_code")

# MA20 window
kline["amp_ma20"]  = gb["amplitude"].transform(lambda x: x.rolling(20, min_periods=10).mean())
kline["turn_ma20"] = gb["turnover"].transform(lambda x: x.rolling(20, min_periods=10).mean())
kline["amt_ma20"]  = gb["amount"].transform(
    lambda x: x.rolling(20, min_periods=10).mean()
)

# log_amount_20d
kline["log_amount_20d"] = kline["amt_ma20"].apply(lambda v: np.log(v + 1) if pd.notna(v) else v)

# Raw factor: negative...
# -MA20(中幅) / MA20(中幅)
# 并确保分母非零
def _raw_factor(row):
    if pd.notna(row["amp_ma20"]) and pd.notna(row["turn_ma20"]) and row["turn_ma20"] > 0:
        return -(row["amp_ma20"] / row["turn_ma20"])
    return np.nan

kline["factor_raw"] = kline.apply(_raw_factor, axis=1)

# ── Cross-sectional MAD winsorize + Neutralize + Z-score ──
valid_mask = kline["factor_raw"].notna() & kline["log_amount_20d"].notna()
sample = kline[valid_mask].copy()
print(f"[TBR v1] Valid rows: {len(sample):,}")

def mad_winsorize(s, n_mad=3.0):
    med = s.median()
    mad = (s - med).abs().median()
    scaled = 1.4826 * mad if mad > 0 else s.std()
    if scaled == 0 or np.isnan(scaled):
        return s
    return s.clip(med - n_mad * scaled, med + n_mad * scaled)

output_records = []

for dt, grp in sample.groupby("date"):
    g = grp.copy()
    n = len(g)
    if n < 30:
        continue

    # MAD winsorize
    g["f_w"] = mad_winsorize(g["factor_raw"], 3.0)

    # log1p-transform（负因子→正视化处理后log再负回去）
    # 因为 amp/turnover≈1~6 → 已经为正；取负后negative→用 sign*log1p
    g["f_l"] = np.sign(g["f_w"]) * np.log1p(g["f_w"].abs())

    # OLS neutralize against log_amount
    X = np.column_stack([np.ones(n), g["log_amount_20d"].values.astype(float)])
    y = g["f_l"].values.astype(float)
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        g["f_n"] = y - X @ beta
    except np.linalg.LinAlgError:
        g["f_n"] = y

    # z-score
    mu, sigma = g["f_n"].mean(), g["f_n"].std()
    g["factor_value"] = np.where(sigma > 0, (g["f_n"] - mu) / sigma, 0.0)

    for _, row in g[["date", "stock_code", "factor_value"]].iterrows():
        output_records.append({
            "date": row["date"],
            "stock_code": int(row["stock_code"]),
            "factor_value": round(float(row["factor_value"]), 6)
        })

result = pd.DataFrame(output_records).sort_values(["date", "stock_code"]).reset_index(drop=True)
print(f"[TBR v1] Final: {len(result):,} rows, {result.stock_code.nunique()} stocks")
print(f"[TBR v1] Factor stats:  mean={result.factor_value.mean():.4f}, std={result.factor_value.std():.4f}")
print(f"[TBR v1] Non-NaN pct: {result.factor_value.notna().mean()*100:.1f}%")

result.to_csv("data/factor_turn_band_ratio_v1.csv", index=False)
print("[TBR v1] Saved → data/factor_turn_band_ratio_v1.csv")
