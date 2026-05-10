#!/usr/bin/env python3
"""
calc_bps_roe_composite_v2.py — BPS×ROE复合因子
原理: BPS/price(Value) + ROE(Quality) → 截面分别中立 → 等权叠加 → 截面临中立
双风格兼顾: 低估值同时高盈利质量
barra: Value + Quality 复合
"""
import pandas as pd, numpy as np
from numpy.linalg import lstsq
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

DATA_DIR   = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab/data")
OUTPUT_DIR = DATA_DIR

print("=== BPS×ROE复合因子 v2 ===")

# 读数据
kline  = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv")
fund   = pd.read_csv(DATA_DIR / "csi1000_fundamental_cache.csv")

kline["date"]   = pd.to_datetime(kline["date"])
fund["report_date"] = pd.to_datetime(fund["report_date"])

# 基本面排序 & 迭代報告期
fund_sorted = fund.sort_values("report_date").drop_duplicates(subset=["stock_code","report_date"], keep="last")
report_dates    = sorted(fund_sorted["report_date"].unique())
report_dates_next = report_dates[1:] + [pd.Timestamp("2026-12-31")]

rows = []
for i, (rd, end_rd) in enumerate(zip(report_dates, report_dates_next)):
    sub = fund_sorted[fund_sorted["report_date"] == rd][["stock_code","bps","roe"]].copy()
    if len(sub) == 0:
        continue
    sub["report_start"] = rd; sub["report_end"] = end_rd
    rows.append(sub)

bp_mapper = pd.concat(rows, ignore_index=True)
print(f"  BPS/ROE映射: {bp_mapper['report_start'].min().date()} ~ {bp_mapper['report_end'].max().date()}")

# 合并
merged = kline.merge(bp_mapper, on=["stock_code"], how="inner")
merged = merged[(merged["date"] >= merged["report_start"]) & (merged["date"] < merged["report_end"])].copy()

# BPS/Price 原始值和ROE原始值
merged["bps_raw"] = merged["bps"] / merged["close"]
merged = merged.dropna(subset=["bps_raw","roe","amount","close"]).copy()
merged = merged[(merged["bps_raw"] > 0.001) & (merged["bps_raw"] < 100)].copy()
merged = merged[(merged["roe"].between(-200, 200))].copy()

print(f"  填充后截面: {merged['date'].nunique()} 天, 日均持仓: {merged.groupby('date')['stock_code'].count().mean():.1f}")

# Step 1: BPS/cPrice 截面中立（成交额OLS）
def neutralize_one(group, col):
    x = np.log(group["amount"].clip(lower=1).values)
    y = group[col].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        return pd.Series(np.nan, index=group.index)
    A = np.column_stack([np.ones(mask.sum()), x[mask]])
    coef, _, _, _ = lstsq(A, y[mask], rcond=None)
    resid = np.full(len(y), np.nan)
    resid[mask] = y[mask] - A @ coef
    return pd.Series(resid, index=group.index)

print("[1] 截面中性化 BPS/price & ROE...")
merged["bps_resid"]  = merged.groupby("date").apply(neutralize_one, col="bps_raw").reset_index(level=0, drop=True)
merged["roe_resid"]  = merged.groupby("date").apply(neutralize_one, col="roe").reset_index(level=0, drop=True)

# Step 2: MAD winsorize
def mad_clip(s, n=5):
    med = s.median()
    mad = np.median(np.abs(s - med))
    if mad < 1e-8:
        return s.clip(0, 1)
    return s.clip(med - n*1.4826*mad, med + n*1.4826*mad)

print("[2] MAD winsorize...")
merged["bps_resid"] = merged.groupby("date")["bps_resid"].transform(lambda x: mad_clip(x))
merged["roe_resid"] = merged.groupby("date")["roe_resid"].transform(lambda x: mad_clip(x))

# Step 3: Z-score 截面标准化
print("[3] Z-score标准化...")
def zscore(s):
    m, std = s.mean(), s.std()
    return (s - m) / (std + 1e-8)

merged["bps_z"] = merged.groupby("date")["bps_resid"].transform(zscore)
merged["roe_z"] = merged.groupby("date")["roe_resid"].transform(zscore)

# Step 4: 等权叠加
print("[4] 等权叠加 BPS + ROE (都正向: BPS高=价值好, ROE高=质量好)...")
merged["composite_raw"] = 0.4 * merged["bps_z"] + 0.6 * merged["roe_z"]

# Step 5: 再中立一次

# 先看看单因子方向:
print(f"\n单因子截面相关性检查:")
corr_stock = merged[["bps_z","roe_z"]].corr()
print(f"  BPS_z vs ROE_z corr = {corr_stock.loc['bps_z','roe_z']:.3f}")

# Step 5: 成交额 OLS 最终中立
def neutralize_final(group):
    x = np.log(group["amount"].clip(lower=1).values)
    y = group["composite_raw"].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        return pd.Series(0.0, index=group.index)
    A = np.column_stack([np.ones(mask.sum()), x[mask]])
    coef, _, _, _ = lstsq(A, y[mask], rcond=None)
    resid = np.full(len(y), 0.0)
    redund = y[mask] - A @ coef
    # 重新 z-score 一下
    std = redund.std()
    if std < 1e-8:
        redund = np.zeros(len(mask))
    else:
        redund = (redund - redund.mean()) / std
    resid[mask] = redund
    return pd.Series(resid, index=group.index)

print("[5] 最终成交额OLS中立 + z-score...")
merged["factor_value"] = merged.groupby("date").apply(neutralize_final).reset_index(level=0, drop=True)

print(f"\n最终因子: mean={merged.factor_value.mean():.4f} std={merged.factor_value.std():.4f}")
print(merged["factor_value"].describe())

# 保存
out = merged[["date","stock_code","factor_value"]].copy()
out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
out_path = OUTPUT_DIR / "factor_bps_roe_comp_v2.csv"
out.to_csv(out_path, index=False)
print(f"\n✅ 保存: {out_path} ({len(out)} 行, {out['date'].min().date()} ~ {out['date'].max().date()})")
