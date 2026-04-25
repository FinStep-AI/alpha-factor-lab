#!/usr/bin/env python3
"""
因子计算：实体强度 (Body Strength)
20日滚动均值 of (close-open)/(high-low)，成交额OLS中性化 + MAD winsorize + z-score

逻辑：实体比例高 = 日内价格力向明确 = 信息/趋势延续信号
反向使用（低实体=高因子值=高预期收益），即 deduct 好实体到头部的)。
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# --- 参数 ---
DATA_DIR = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab/data")
OUTPUT_DIR = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab")
WINDOW = 20

print("=== Step 1: 加载数据 ===")
kline = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv", encoding="utf-8")
kline["date"] = pd.to_datetime(kline["date"])
kline = kline.sort_values(["stock_code", "date"]).reset_index(drop=True)

print(f"K线数据: {len(kline)} 行, 股票数: {kline['stock_code'].nunique()}, 日期范围: {kline['date'].min()} ~ {kline['date'].max()}")

# --- Step 2: 计算实体比例 ---
# 防止除以0
kline["hl_range"] = kline["high"] - kline["low"]
kline.loc[kline["hl_range"] <= 0.001, "hl_range"] = 0.001  # 极小值保护

kline["body_ratio"] = (kline["close"] - kline["open"]) / kline["hl_range"]

print(f"实体比例统计:\n{kline['body_ratio'].describe().round(4)}")

# --- Step 3: 计算20日滚动均值 ---
print("=== Step 2: 20日滚动均值 ===")
kline = kline.sort_values(["stock_code", "date"])
kline["body_strength_raw"] = kline.groupby("stock_code")["body_ratio"].transform(
    lambda x: x.rolling(WINDOW, min_periods=int(WINDOW * 0.8)).mean()
)

print(f"实体强度raw统计:\n{kline['body_strength_raw'].describe().round(4)}")

# --- Step 4: 
import statsmodels.api as sm

print("=== Step 3: 成交额OLS中性化 ===")

# 构建矩阵对齐：cross-section中性化 (截面中性化, 非滞后)
# 使用 cross-sectional OLS neutralize

# Step 4a: 计算20日平均成交额的对数
kline["log_amount_20d"] = kline.groupby("stock_code")["amount"].transform(
    lambda x: np.log1p(x.rolling(WINDOW, min_periods=int(WINDOW * 0.8)).mean())
)

# 过滤有效值
valid_mask = kline["body_strength_raw"].notna() & kline["log_amount_20d"].notna() & np.isfinite(kline["body_strength_raw"])
work_df = kline[valid_mask].copy()
print(f"中性化样本量: {len(work_df)}")

# 截面OLS中性化
from numpy.linalg import lstsq

def cs_neutralize(df, y_col="body_strength_raw", x_col="log_amount_20d"):
    """逐截面OLS中性化，返回残差"""
    results = []
    for dt, grp in df.groupby("date"):
        y = grp[y_col].values.astype(float)
        x = grp[x_col].values.astype(float)
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            results.append(pd.Series(np.nan, index=grp.index))
            continue
        ones = np.ones(mask.sum())
        X = np.column_stack([ones, x[mask]])
        beta, _, _, _ = lstsq(X, y[mask], rcond=None)
        resid = np.full(len(grp), np.nan)
        resid_valid = y[mask] - X @ beta
        resid[mask] = resid_valid
        results.append(pd.Series(resid, index=grp.index))
    return pd.concat(results)

work_df["neutralized"] = cs_neutralize(work_df, "body_strength_raw", "log_amount_20d")
work_df = work_df.dropna(subset=["neutralized"])

print(f"中性化后样本: {len(work_df)}")

# --- Step 5: MAD winsorize + z-score ---
print("=== Step 4: MAD winsorize + z-score ===")

def mad_winsorize(s, n_sigma=5.0):
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if mad < 1e-10:
        return s
    lo, hi = med - n_sigma * mad, med + n_sigma * mad
    return s.clip(lo, hi)

output_rows = []
for dt, grp in work_df.sort_index().groupby("date", sort=True):
    vals = mad_winsorize(grp["neutralized"], n_sigma=5.0)
    mean_v = vals.mean()
    std_v = vals.std()
    if std_v < 1e-10 or len(vals) < 30:
        continue
    z = (vals - mean_v) / std_v
    codes = grp["stock_code"].values
    z_vals = z.values
    for code, fv in zip(codes, z_vals):
        output_rows.append({"date": dt.date().isoformat(), "stock_code": code, "factor_value": round(float(fv), 6)})

final_df = pd.DataFrame(output_rows)
print(f"最终因子矩阵: {len(final_df)} 行, {final_df['stock_code'].nunique()} 股票, {final_df['date'].nunique()} 天")
print(f"日期范围: {final_df['date'].min()} ~ {final_df['date'].max()}")

# --- 输出 CSV ---
OUT_FILE = OUTPUT_DIR / "data/factor_body_strength_v1.csv"
final_df.to_csv(OUT_FILE, index=False)
print(f"\n✅ 因子已保存: {OUT_FILE}")
print(f"   统计:\n{final_df['factor_value'].describe().round(4)}")
