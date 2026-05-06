#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子计算脚本: book_value_ratio_v1 — 账面市值比 (B/P Proxy)
Barra Style: Value

构造逻辑 (参考: Frazzini & French "Betting Against Beta", Fama-French HML):
  raw = bps / close_price
  1. 取对数 (消除scale): log(bps/close) = log_bps - log_close
  2. 过滤负BPS (财务异常)
  3. 前视避免: 季报披露后45天可用
  4. 收盘价当月末T-1此报告日查
方向: 正向 — 高B/P (账面价值高于市价) → 价值溢价 → 后续高收益

关键步骤:
  1. 季报日期 + lag(45d) → 信息可获交易日
  2. 用merge_asof对每只股票在(当前交易日×最新可用bps之间)forward-fill
  3. 截面: neutralize(log_bp, log_amount_20d) → MAD winsorize → z-score
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings, sys

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
KLINE_PATH = BASE_DIR / "data" / "csi1000_kline_raw.csv"
FUND_PATH  = BASE_DIR / "data" / "csi1000_fundamental_cache.csv"
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(exist_ok=True)
INFO_LAG_DAYS = 45  # 季报披露后45天可用

print("[1] 加载 K 线数据...")
kline = pd.read_csv(KLINE_PATH)
kline["date"] = pd.to_datetime(kline["date"])
kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)

# 构建log_amount_20d 中队代理
kline["log_amount"] = np.log(np.maximum(kline["amount"], 1))
kline["log_amount_20d"] = (
    kline.groupby("stock_code")["log_amount"]
    .transform(lambda x: x.rolling(20, min_periods=5).mean())
)

print("[2] 加载基本面数据...")
fund = pd.read_csv(FUND_PATH)
fund["report_date"] = pd.to_datetime(fund["report_date"])
fund = fund.sort_values(["stock_code", "report_date"]).reset_index(drop=True)

# 信息可交易日 = 报告日 + lag
fund["info_date"] = fund["report_date"] + pd.Timedelta(days=INFO_LAG_DAYS)

# 过滤负BPS（算入异常front-end，N/A处理）
fund = fund[fund["bps"] > 0].copy()
print(f"  保留正BPS: {len(fund):,} 条(原始{len(pd.read_csv(FUND_PATH)):,}条)")

# ─────────────────────────────────────────
# 3. Merge BPS到交易日 (merge_asof)
# ─────────────────────────────────────────
print("[3] 映射季报BPS到每个交易日...")

# 按股票分组后执行merge_asof
panel = kline[["date", "stock_code", "close", "log_amount_20d", "amount"]].copy()
panel = panel.sort_values(["stock_code", "date"]).drop_duplicates(["date", "stock_code"])

results = []
for sc, grp in panel.groupby("stock_code"):
    f = fund[fund["stock_code"] == sc][["info_date", "bps"]].sort_values("info_date")
    if len(f) == 0:
        continue
    merged = pd.merge_asof(
        grp.sort_values("date"),
        f,
        left_on="date", right_on="info_date",
        direction="backward"
    )
    results.append(merged)

panel = pd.concat(results, ignore_index=True)
print(f"  映射完成: {panel.shape}")
print(f"  BPS notna: {panel['bps'].notna().sum():,}")

# ─────────────────────────────────────────
# 4. 计算 B/P 因子原始值
# ─────────────────────────────────────────
print("[4] 计算 B\/P = log(bps\/close)...\n")  # prepared data for step 4
  # (raw_bp = log_bps - log_close computed below)
panel = panel.dropna(subset=["bps", "close", "log_amount_20d"])
panel = panel[panel["bps"] > 0]
panel = panel[panel["close"] > 0]

# log(bps/close) = log(bps) - log(close)
panel["log_bps"]  = np.log(panel["bps"].clip(lower=0.01))
panel["log_close"] = np.log(panel["close"].clip(lower=0.01))
panel["raw_bp"] = panel["log_bps"] - panel["log_close"]

print(f"  Raw BP 统计:\n{panel['raw_bp'].describe().round(4)}")

# ─────────────────────────────────────────
# 5. 截面中性化: neutralize(log_bp, log_amount_20d)
# ─────────────────────────────────────────
print("[5] 截面中性化 + Winsorize + z-score...")

def mad_winsorize(s: pd.Series, n_mad: float = 3.0) -> pd.Series:
    med  = s.median()
    mad  = (s - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return s
    scaled = 1.4826 * mad
    return s.clip(med - n_mad * scaled, med + n_mad * scaled)

def ols_neutralize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < X.shape[1] + 2:
        return y
    y_c = y[mask].copy();  X_c = X[mask].copy()
    # Add intercept
    X_c = np.column_stack([np.ones(len(X_c)), X_c])
    try:
        beta = np.linalg.lstsq(X_c, y_c, rcond=None)[0]
        res_tmp = np.full(len(y), np.nan)
        res_tmp[mask] = y_c - X_c @ beta
        return res_tmp
    except np.linalg.LinAlgError:
        return y

panel = panel.reset_index(drop=True)
vals  = panel["raw_bp"].values.astype(float)
amt   = panel["log_amount_20d"].values.astype(float)

factor_vals = pd.Series(np.nan, index=panel.index)

for date, group_idx in panel.groupby("date").groups.items():
    idx  = np.array(list(group_idx))
    y    = vals[idx].copy()
    X_pr = amt[idx].copy()

    valid = np.isfinite(y) & np.isfinite(X_pr)
    if valid.sum() < 30:
        continue

    # Winsorize on raw
    y_series = pd.Series(y[valid])
    y_w      = mad_winsorize(y_series).values
    y_full   = y.copy()
    y_full[valid] = y_w

    # Neutralize: regress out log_amount_20d
    X_val = X_pr.reshape(-1, 1)
    res   = ols_neutralize(y_full, X_val)

    # z-score
    valid2 = np.isfinite(res)
    if valid2.sum() > 0:
        mu  = res[valid2].mean()
        std = res[valid2].std()
        if std > 1e-8:
            factor_vals.loc[idx] = (res - mu) / std
        else:
            factor_vals.loc[idx] = 0.0

panel["factor_value"] = factor_vals
panel = panel.dropna(subset=["factor_value"])

print(f"  有效截面数: {panel['date'].nunique()}")
print(f"  因子值均值: {panel['factor_value'].mean():.4f}")
print(f"  因子值标准差: {panel['factor_value'].std():.4f}")

# ─────────────────────────────────────────
# 6. 输出
# ─────────────────────────────────────────
out = panel[["date", "stock_code", "factor_value"]].copy()
out["date"] = out["date"].dt.strftime("%Y-%m-%d")
out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
out.to_csv(OUTPUT_DIR / "factor_book_value_ratio_v1.csv", index=False)
print(f"\n✅ 因子值已保存: {OUTPUT_DIR / 'factor_book_value_ratio_v1.csv'}")
print(f"  形状: {out.shape}")
print(f"  样本:\n{out.head(6)}")
print(f"\n  截面样例 (cross-section sample at first 3 dates):")
for d in out["date"].unique()[:3]:
    subset = out[out["date"] == d]["factor_value"]
    print(f"  {d}: n={len(subset)}, mean={subset.mean():.4f}, std={subset.std():.4f}")
