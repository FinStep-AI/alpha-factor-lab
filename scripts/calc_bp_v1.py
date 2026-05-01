#!/usr/bin/env python3
"""
因子计算：账面价值比 (Book-to-Price = BPS/close)
基本面Value类因子，Barra Value风格
 raw_value = BPS/close (等价于 PB倒数) → 成交额OLS中性化 → MAD winsorize → z-score
高BP = 低估值 = 账面价值相对价格更高 = 价值股
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numpy.linalg import lstsq
import warnings
warnings.filterwarnings("ignore")

DATA_DIR   = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab/data")
OUTPUT_DIR = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab")
WINDOW     = 20

print("=== BP Book-to-Price因子 计算 ===")

# ── Step 1: 加载数据 ──
print("[Step 1] 加载数据...")
kline  = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv").assign(date=lambda d: pd.to_datetime(d["date"]))
fund   = pd.read_csv(DATA_DIR / "csi1000_fundamental_cache.csv").assign(report_date=lambda d: pd.to_datetime(d["report_date"]))

# ── Step 2: 报告期 → 覆盖交易日期的映射 ──
#  每个季报的BPS值映射到[report_date, next_report_date) 覆盖
fund_sorted  = fund.sort_values("report_date")
report_dates = sorted(fund_sorted["report_date"].unique())

spans = []
for i, rd in enumerate(report_dates):
    end = report_dates[i+1] if i+1 < len(report_dates) else pd.Timestamp("2026-12-31")
    sub = fund_sorted[fund_sorted["report_date"] == rd][["stock_code", "bps"]].copy()
    sub["start"] = rd
    sub["end"]   = end
    spans.append(sub)

bp_mapper = pd.concat(spans, ignore_index=True)

# 合并：K线 × BPS[领先9个月]
print("[Step 2] 合并K线与BPS...")
merged = (kline.merge(bp_mapper, on=["stock_code"], how="inner")
          .query("start <= date < end")   # 领先9个月，映射到下一期的开始
          .copy())

merged["bp_raw"] = merged["bps"] / merged["close"]
# 过滤异常：BP ∈ [0.01, 50]（极端偏高偏低去除）
merged = merged.dropna(subset=["bp_raw", "amount"])
merged = merged[merged["bp_raw"].between(0.01, 50)].copy()

n_after_merge = len(merged)
print(f"  合并后截面样本: {n_after_merge} 行 | BP范围 [{merged.bp_raw.min():.3f}, {merged.bp_raw.max():.3f}]")

# ── Step 3: 成交额中性化 ──
print("[Step 3] 成交额OLS截面中性化...")

# 20日平均成交额（对数）
kline_sorted = kline.sort_values(["stock_code", "date"]).copy()
kline_sorted["log_amount_20d"] = (
    kline_sorted.groupby("stock_code")["amount"]
    .transform(lambda x: np.log1p(x.rolling(WINDOW, min_periods=int(WINDOW*0.7)).mean()))
)

# 回 Merge，仅对齐 date + stock_code
merged = (merged
          .merge(kline_sorted[["date", "stock_code", "log_amount_20d"]], on=["date", "stock_code"], how="left"))
merged = merged.dropna(subset=["log_amount_20d"])

def cs_ols_neutralize(df, y_col, x_col):
    rows = []
    for dt, grp in df.groupby("date", sort=True):
        y = grp[y_col].values.astype(float)
        x = grp[x_col].values.astype(float)
        ok = np.isfinite(y) & np.isfinite(x)
        if ok.sum() < 30:
            continue
        X = np.column_stack([np.ones(ok.sum()), x[ok]])
        beta, _, _, _ = lstsq(X, y[ok], rcond=None)
        resid = np.full(len(grp), np.nan)
        resid[ok] = y[ok] - X @ beta
        for idx, sc, rv in zip(grp.index, grp["stock_code"].values, resid):
            rows.append({"date": dt, "stock_code": sc, "neutralized": float(rv) if np.isfinite(rv) else np.nan})
    return pd.DataFrame(rows)

neutral_df = cs_ols_neutralize(merged, "bp_raw", "log_amount_20d")
neutral_df = neutral_df.dropna(subset=["neutralized"]).reset_index(drop=True)
print(f"  中性化后截面: {len(neutral_df)} 行, {neutral_df.date.nunique()} 个截面")

# ── Step 4: MAD winsorize + z-score ──
print("[Step 4] MAD winsorize + z-score...")

def mad_winsorize(series, n_sigma=5.0):
    med = series.median()
    mad = (series - med).abs().median() * 1.4826
    if mad < 1e-10:
        return series
    return series.clip(med - n_sigma*mad, med + n_sigma*mad)

output_rows = []
for dt, grp in neutral_df.groupby("date", sort=True):
    n = len(grp)
    if n < 30:
        continue
    win = mad_winsorize(grp["neutralized"], 5.0)
    mu, sd = win.mean(), win.std()
    if sd < 1e-10:
        continue
    z = (win - mu) / sd
    for sc, fv in zip(grp["stock_code"].values, z.values):
        output_rows.append({
            "date":         dt.date().isoformat(),
            "stock_code":   int(sc),
            "factor_value": round(float(fv), 6),
        })

final_df = pd.DataFrame(output_rows).sort_values(["date", "stock_code"]).reset_index(drop=True)
out_file = OUTPUT_DIR / "data/factor_bp_v1.csv"
final_df.to_csv(out_file, index=False)

print(f"\n✅ 因子输出完成: {out_file}")
print(f"  行数={len(final_df)}  | 股票={final_df.stock_code.nunique()}  | 天={final_df.date.nunique()}")
print(f"  日期范围: {final_df.date.min()} ~ {final_df.date.max()}")
print(f"  factor_value 描述统计:\n{final_df['factor_value'].describe().round(4)}")
