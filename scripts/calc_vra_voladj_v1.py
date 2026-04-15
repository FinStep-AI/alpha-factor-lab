#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子: vol_ret_align_v1 — 波动率调整的量价一致性因子 (v1)
构造:
  1. Sign Alignment: sign(ret_t) * sign(vol_chg_t), 20日均值
  2. Volatility 标准化: 除以MA20(|ret|), 消除波动率水平影响
  3. 市值中性化(OLS) + MAD winsorize + z-score
  4. 方向: 正向(高值=高预期收益, 专业资金持续推动逻辑)

与 vol_ret_consistency_v1_20d 的区别:
  - 原版: raw sign alignment mean, 信号强度被波动率稀释 → IC极弱(≈0)但mono=0.9
  - 本版: divided by avg(|ret|) 标准化, 控制波动率水平后更纯净

理论依据:
- Brennan, Chordia & Subrahmanyam (1998) "Alternative factor specifications..."
  → Order imbalance 对 cross-section 有预测力
- Chordia & Subrahmanyam (2004) "Order Imbalance and Individual Stock Returns"
  → 量价同向 → 知情交易/流动性需求信号
- A股中证1000: 波动率标准的量价一致性 = 排除纯波动噪音后的真实信息流
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab/data")
OUTPUT_DIR = Path("/home/sunrui/.openclaw/workspace-buffett/alpha-factor-lab/output/vol_ret_align_v1_5d")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINSORIZE_PCT = 0.05
LOG_AMT_WINDOW = 20  # 市值代理窗口

print("=" * 60)
print("  因子:vol_ret_align_v1 — 波动率调整量价一致性")
print("=" * 60)

# [1] Load data
print("\n[1] 加载行情数据...")
df = pd.read_csv(DATA_DIR / "csi1000_kline_raw.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

volume_piv  = df.pivot_table(index="date", columns="stock_code", values="volume")
close_piv   = df.pivot_table(index="date", columns="stock_code", values="close")
amount_piv  = df.pivot_table(index="date", columns="stock_code", values="amount")

ret_piv     = close_piv.pct_change()

# [2] Sign Alignment  (equally-weighted)
print("\n[2] 构造量价一致性信号...")
sign_ret = np.sign(ret_piv)
vol_chg  = volume_piv.diff()
sign_vol = np.sign(vol_chg)
align_sign = sign_ret * sign_vol  # ∈ {-1, 0, 1}

# 20日均值: 信号强度
align_20d = align_sign.rolling(20, min_periods=15).mean()
align_10d = align_sign.rolling(10, min_periods=8).mean()

# 40日均值 (更稳定)
align_40d = align_sign.rolling(40, min_periods=30).mean()

# [3] 波动率标准化 — 核心创新!
print("\n[3] 波动率标准化 (除以MA20(|ret|))...")
abs_ret = ret_piv.abs()
avg_abs_ret_20d = abs_ret.rolling(20, min_periods=15).mean()

# Avoid division by very small numbers
avg_abs_ret_20d = avg_abs_ret_20d.clip(lower=1e-6)

# ===== v1a: 原始sign alignment =====
factor_raw = align_20d.copy()

# ===== v1b: 波动率标准化版 =====
factor_voladj = (align_20d / avg_abs_ret_20d).replace([np.inf, -np.inf], np.nan)

# Further clip extreme values (some stocks with very low avg_abs_ret)
factor_voladj = factor_voladj.clip(-5, 5)

print(f"    raw_align range: [{factor_raw.min().min():.3f}, {factor_raw.max().max():.3f}]")
print(f"    vol_adj range: [{factor_voladj.min().min():.3f}, {factor_voladj.max().max():.3f}]")

# [4] 市值中性化 (O^2祥)
print("\n[4] 市值中性化 (OLS on log_amount20d)...")
log_amount_20d = np.log(amount_piv.rolling(LOG_AMT_WINDOW).mean().clip(lower=1))

def neutralize(factor_matrix: pd.DataFrame, mktcap_matrix: pd.DataFrame) -> pd.DataFrame:
    """OLS 市值中性化: factor = alpha + beta*log_mktcap + residual"""
    result = factor_matrix.copy()
    all_dates = sorted(factor_matrix.dropna(how='all').index)
    for date in all_dates:
        f = factor_matrix.loc[date].dropna()
        m = mktcap_matrix.loc[date].reindex(f.index).dropna()
        common = f.index.intersection(m.index)
        if len(common) < 30:
            continue
        fv = f[common].values.astype(float)
        mv = m[common].values.astype(float)
        X = np.column_stack([np.ones(len(mv)), mv])
        try:
            beta = np.linalg.lstsq(X, fv, rcond=None)[0]
            result.loc[date, common] = fv - X @ beta
        except Exception:
            pass
    return result

factor_neutral_raw      = neutralize(factor_raw, log_amount_20d)
factor_neutral_voladj   = neutralize(factor_voladj, log_amount_20d)

# [5] MAD winsorize + z-score, 并保存 CSV
def standardize_and_save(factor_neutral: pd.DataFrame, output_csv: Path, label: str):
    """MAD winsorize + z-score by date"""
    f = factor_neutral.copy()
    for date in sorted(f.dropna(how='all').index):
        vals = f.loc[date].dropna()
        if len(vals) < 20:
            continue
        median = vals.median()
        mad = np.abs(vals - median).median()
        if mad < 1e-10:
            continue
        k = 3.0
        lo = median - k * 1.4826 * mad
        hi = median + k * 1.4826 * mad
        f.loc[date] = f.loc[date].clip(lo, hi)
        
        vals2 = f.loc[date].dropna()
        mean = vals2.mean()
        std = vals2.std()
        if std > 1e-10:
            f.loc[date, vals2.index] = (vals2 - mean) / std
    
    # Export
    long_df = f.stack().reset_index()
    long_df.columns = ["date", "stock_code", "factor_value"]
    long_df = long_df.sort_values(["date", "stock_code"]).dropna(subset=["factor_value"])
    long_df.to_csv(output_csv, index=False)
    print(f"\n    [{label}] → {output_csv}")
    print(f"      rows={len(long_df)}, nonna_ratio={f.notna().mean().mean():.2%}")
    return f

print(f"\n[5] 保存因子CSV...")
f_raw      = standardize_and_save(factor_neutral_raw,
    DATA_DIR / "factor_vra_voladj_v1.csv", "vol-adjusted")
f_voladj   = standardize_and_save(factor_neutral_voladj,
    DATA_DIR / "factor_vra_voladj_v1.csv", "vol-adjusted")

print("\n" + "=" * 60)
print("  因子构造完成！两个variant:")
print("  1. factor_vra_voladj_v1.csv — 波动率标准化版(← 重点测试)")
print("  2. factor_vra_raw_v1.csv — 原始sign均值版")
print("  Barra风格: MICRO (微观结构)")
print("=" * 60)
