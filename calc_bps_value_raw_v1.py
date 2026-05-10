#!/usr/bin/env python3
"""
calc_bps_value_raw_v1.py — BPS/收盘价 原始差值
性质: Barra Value风格 | 学术因子: Book-to-Market (Fama-French)
逻辑: 高BPS/Price = 账面/市值高 = Value（市价低于账面）→ 做多
"""
import pandas as pd, numpy as np
from numpy.linalg import lstsq
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

DATA_DIR   = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab/data")
OUTPUT_DIR = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab/data")

print("=== BPS原始差值因子 v1 ===")

kline  = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv")
fund   = pd.read_csv(DATA_DIR / "csi1000_fundamental_cache.csv")

kline["date"]   = pd.to_datetime(kline["date"])
fund["report_date"] = pd.to_datetime(fund["report_date"])

# 基本面数据排序 + 只保留最新报告
fund_sorted = fund.sort_values("report_date").drop_duplicates(subset=["stock_code","report_date"], keep="last")

# 报告期 → 截面日期的映射（持仓期逻辑）
report_dates    = sorted(fund_sorted["report_date"].unique())
report_dates_next = report_dates[1:] + [pd.Timestamp("2026-12-31")]

print(f"  基本面区间: {report_dates[1].date()} ~ {report_dates_next[-1].date()}")
print(f"  报告期数量: {len(report_dates)}")

rows = []
for i, (rd, end_rd) in enumerate(zip(report_dates, report_dates_next)):
    sub = fund_sorted[fund_sorted["report_date"] == rd][["stock_code","bps"]].copy()
    if len(sub) == 0:
        continue
    sub["report_start"] = rd; sub["report_end"] = end_rd
    rows.append(sub)

bp_mapper = pd.concat(rows, ignore_index=True)
print(f"  BPS映射区间: {bp_mapper['report_start'].min().date()} ~ {bp_mapper['report_end'].max().date()}")

# 合并 K 线数据（组内匹配）
merged = kline.merge(bp_mapper, on=["stock_code"], how="inner")
print(f"  内连接合并后原始行数: {len(merged)}")
merged = merged[(merged["date"] >= merged["report_start"]) & (merged["date"] < merged["report_end"])].copy()
print(f"  持仓期间过滤后行数: {len(merged)}")

merged["bp_raw"] = merged["bps"] / merged["close"]
merged = merged.dropna(subset=["bp_raw","amount","close"]).copy()
merged = merged[merged["bp_raw"].between(0.001, 100)].copy()
print(f"  过滤后截面数: {merged['date'].nunique()} 天")
print(f"  每日平均持仓: {merged.groupby('date')['stock_code'].count().mean():.1f} 只")
print(f"  BP原始值: mean={merged.bp_raw.mean():.4f} std={merged.bp_raw.std():.4f}")

# 截面中性化:  (log_bp - log(close))  →  
# 直接做 BPS/close，不做 Rank，保留原始尺度差异
factor_raw = merged["bp_raw"].copy()

# 成交额OLS中性化
def neutralize_ols(group):
    x = np.log(group["amount"].clip(lower=1))
    y = group["bp_raw"].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        group["factor_raw"] = np.nan
        return group
    A = np.column_stack([np.ones(mask.sum()), x[mask].values])
    coef, _, _, _ = lstsq(A, y[mask], rcond=None)
    resid = np.full(len(y), np.nan)
    resid[mask] = y[mask] - A @ coef
    group["factor_raw"] = resid
    return group

print("\n[2] 成交额OLS中性化...")
merged = merged.groupby("date").apply(neutralize_ols).reset_index(drop=True)
print(f"  残差mean={merged.factor_raw.mean():.4f} std={merged.factor_raw.std():.4f}")

# MAD winsorize (5σ)
print("[3] MAD winsorize 5σ...")
def mad_winsorize(group, n=5):
    med = group.median()
    mad = np.median(np.abs(group - med))
    if mad < 1e-8:
        return group.clip(med - 0.01, med + 0.01)
    lower = med - n * 1.4826 * mad
    upper = med + n * 1.4826 * mad
    return group.clip(lower, upper)

merged["factor_clean"] = merged.groupby("date")["factor_raw"].transform(
    lambda x: mad_winsorize(x, n=5))

# Z-score 标准化
print("[4] Z-score标准化...")
merged["factor_value"] = merged.groupby("date")["factor_clean"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-8) if x.std() > 1e-8 else 0.0)

# 取负方向: BPS/close 高 = 高账面价值 = 做多 → 直接正向
merged["factor_value_directed"] = merged["factor_value"]

print(f"\n最终因子统计:")
print(merged["factor_value_directed"].describe())

# 保存
out = merged[["date","stock_code","factor_value_directed"]].copy()
out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
out = out.rename(columns={"factor_value_directed": "factor_value"})
out_path = OUTPUT_DIR / "factor_bps_value_raw_v1.csv"
out.to_csv(out_path, index=False)
print(f"\n✅ 保存到: {out_path} ({len(out)} 行, {out['date'].min().date()} ~ {out['date'].max().date()})")
