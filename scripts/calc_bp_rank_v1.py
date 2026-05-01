#!/usr/bin/env python3
"""
calc_bp_rank_v1.py — BP因子 + 截面Rank变换
步骤:
  1) BPS/close (已有原始值)
  2) 截面 Rank(0-1): 消除分布偏态
  3) 成交额OLS中性化
  4) MAD winsorize (5σ)
  5) z-score
"""

import pandas as pd, numpy as np
from numpy.linalg import lstsq
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

DATA_DIR   = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab/data")
OUTPUT_DIR = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab")
WINDOW     = 20

print("=== BP Rank因子 (截面Rank变换) ===")

kline  = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv").assign(date=lambda d: pd.to_datetime(d["date"]))
fund   = pd.read_csv(DATA_DIR / "csi1000_fundamental_cache.csv").assign(report_date=lambda d: pd.to_datetime(d["report_date"]))
fund_sorted = fund.sort_values("report_date")
report_dates = sorted(fund_sorted["report_date"].unique())

# BPS映射到截面
spans = []
for i, rd in enumerate(report_dates):
    end = report_dates[i+1] if i+1 < len(report_dates) else pd.Timestamp("2026-12-31")
    sub = fund_sorted[fund_sorted["report_date"] == rd][["stock_code","bps"]].copy()
    sub["start"] = rd; sub["end"] = end
    spans.append(sub)
bp_mapper = pd.concat(spans).reset_index(drop=True)

merged = kline.merge(bp_mapper, on=["stock_code"], how="inner").query("start <= date < end").copy()
merged["bp_raw"] = merged["bps"] / merged["close"]
merged = merged.dropna(subset=["bp_raw","amount"]).copy()
merged = merged[merged["bp_raw"].between(0.01, 50)]

n = len(merged)
print(f"  合并截面: {n} 行, BP∈ [{merged.bp_raw.min():.3f}, {merged.bp_raw.max():.3f}]")

# Step 1: 截面 Rank 变换
print("[1] 截面Rank(0-1)变换...")
merged["bp_rank"] = merged.groupby("date")["bp_raw"].rank(method="average", pct=True).astype(float)
print(f"  Rank范围 [{merged.bp_rank.min():.3f}, {merged.bp_rank.max():.3f}], 均值={merged.bp_rank.mean():.3f}")

# Step 2: 成交额OLS中性化
kline_sorted = kline.sort_values(["stock_code","date"]).copy()
kline_sorted["log_amount_20d"] = kline_sorted.groupby("stock_code")["amount"].transform(
    lambda x: np.log1p(x.rolling(WINDOW, min_periods=int(WINDOW*0.7)).mean()))

merged2 = (merged
    .merge(kline_sorted[["date","stock_code","log_amount_20d"]], on=["date","stock_code"], how="left")
    .dropna(subset=["log_amount_20d"]).copy())

def cs_neutralize(df, y_col, x_col):
    """逐截面OLS中性化"""
    rows = []
    for dt, g in df.groupby("date", sort=True):
        y, x = g[y_col].values.astype(float), g[x_col].values.astype(float)
        ok = np.isfinite(y) & np.isfinite(x)
        if ok.sum() < 30: continue
        X = np.column_stack([np.ones(ok.sum()), x[ok]])
        bet, _, _, _ = lstsq(X, y[ok], rcond=None)
        resid = np.full(len(g), np.nan)
        resid[ok] = y[ok] - X @ bet
        for idx, sc, rv in zip(g.index, g["stock_code"].values, resid):
            rows.append({"date":dt,"stock_code":sc,"neutralized":float(rv) if np.isfinite(rv) else np.nan})
    return pd.DataFrame(rows)

print("[2] 截面OLS neutralization...")
neut = cs_neutralize(merged2, "bp_rank", "log_amount_20d").dropna(subset=["neutralized"])
print(f"  中性化: {len(neut)} 行")

# Step 3: MAD + z-score
def mad_winsorize(s, n_s=5.0):
    med=s.median(); mad=(s-med).abs().median()*1.4826
    if mad<1e-10: return s
    return s.clip(med-n_s*mad, med+n_s*mad)

print("[3] MAD z-score...")
rows = []
for dt, g in neut.groupby("date", sort=True):
    n = len(g)
    if n < 30: continue
    win = mad_winsorize(g["neutralized"], 5.0)
    mu, sd = win.mean(), win.std()
    if sd < 1e-10: continue
    z = (win - mu) / sd
    for sc, fv in zip(g["stock_code"].values, z.values):
        rows.append({"date":dt.date().isoformat(),"stock_code":int(sc),"factor_value":round(float(fv),6)})

final = pd.DataFrame(rows).sort_values(["date","stock_code"]).reset_index(drop=True)
out = OUTPUT_DIR / "data/factor_bp_rank_v1.csv"
final.to_csv(out, index=False)
print(f"\n✅ 因子写入: {out}")
print(f"  行={len(final)}, 股票={final.stock_code.nunique()}, 天={final.date.nunique()}")
print(f"  均值={final.factor_value.mean():.4f}, std={final.factor_value.std():.4f}")
