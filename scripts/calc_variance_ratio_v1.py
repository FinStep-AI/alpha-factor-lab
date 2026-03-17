#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variance Ratio 因子 v1
========================
基于 Lo & MacKinlay (1988) 的方差比检验思想。

VR(q) = Var(q日收益) / (q × Var(1日收益))

- VR > 1 → 正自相关（趋势型）
- VR < 1 → 负自相关（反转型）

构造：
  - 使用 q=5（一周），滚动窗口 W=20 天
  - VR = var(5日累积收益, 20d) / (5 * var(1日收益, 20d))
  - 取 log(VR) 使分布更对称
  - 成交额中性化 + MAD winsorize + z-score

方向假设：
  - 基于 CSI1000 的经验（动量确认 > 反转），预期高 VR（趋势型）可能更好
  - 但也准备测试反向

数据：data/csi1000_kline_raw.csv (OHLCV)
输出：data/factor_variance_ratio_v1.csv
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# 参数
# ============================================================
Q = 5            # 方差比的时间尺度 (5日 = 一周)
WINDOW = 20      # 滚动窗口
MIN_PERIODS = 15 # 最少有效天数

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
KLINE_FILE = DATA_DIR / "csi1000_kline_raw.csv"
OUTPUT_FILE = DATA_DIR / "factor_variance_ratio_v1.csv"


def mad_winsorize(s, n_mad=5):
    """MAD winsorize"""
    median = s.median()
    mad = (s - median).abs().median() * 1.4826
    lower = median - n_mad * mad
    upper = median + n_mad * mad
    return s.clip(lower, upper)


def neutralize_ols(df, factor_col, neutral_col):
    """OLS 中性化"""
    from numpy.linalg import lstsq
    result = df[[factor_col]].copy()
    result["residual"] = np.nan

    for date, group in df.groupby("date"):
        y = group[factor_col].values
        x = group[neutral_col].values
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 30:
            continue
        X = np.column_stack([np.ones(mask.sum()), x[mask]])
        beta, _, _, _ = lstsq(X, y[mask], rcond=None)
        resid = y.copy()
        resid[mask] = y[mask] - X @ beta
        resid[~mask] = np.nan
        result.loc[group.index, "residual"] = resid

    return result["residual"]


def main():
    print("=" * 60)
    print("Variance Ratio Factor v1")
    print("=" * 60)

    # 1. 读取数据
    print("\n[1] 读取K线数据...")
    df = pd.read_csv(KLINE_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    print(f"  数据量: {len(df):,} 行, {df['stock_code'].nunique()} 只股票")
    print(f"  日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")

    # 2. 计算日收益率
    print("\n[2] 计算日收益率...")
    df["ret"] = df.groupby("stock_code")["close"].pct_change()

    # 3. 计算 q 日累积收益
    print(f"\n[3] 计算 {Q} 日累积收益...")
    df["ret_q"] = df.groupby("stock_code")["ret"].transform(
        lambda x: x.rolling(Q, min_periods=Q).sum()
    )

    # 4. 计算方差比
    print(f"\n[4] 计算 VR({Q})，窗口={WINDOW}...")
    # var(1日收益, WINDOW)
    df["var_1d"] = df.groupby("stock_code")["ret"].transform(
        lambda x: x.rolling(WINDOW, min_periods=MIN_PERIODS).var()
    )
    # var(q日收益, WINDOW)
    df["var_qd"] = df.groupby("stock_code")["ret_q"].transform(
        lambda x: x.rolling(WINDOW, min_periods=MIN_PERIODS).var()
    )

    # VR = var(q日) / (q * var(1日))
    df["vr_raw"] = df["var_qd"] / (Q * df["var_1d"])

    # log 变换使分布更对称
    df["vr_log"] = np.log(df["vr_raw"].clip(lower=1e-6))

    # 5. 成交额中性化
    print("\n[5] 中性化处理...")
    # 20日平均成交额
    df["log_amount_20d"] = np.log(
        df.groupby("stock_code")["amount"].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        ).clip(lower=1)
    )

    # 截面 MAD winsorize
    df["vr_win"] = df.groupby("date")["vr_log"].transform(mad_winsorize)

    # OLS 中性化
    df["factor_neutral"] = neutralize_ols(df, "vr_win", "log_amount_20d")

    # 截面 z-score
    df["factor_value"] = df.groupby("date")["factor_neutral"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

    # 6. 统计
    valid = df.dropna(subset=["factor_value"])
    print(f"\n[6] 统计:")
    print(f"  有效样本: {len(valid):,} 行")
    print(f"  日均覆盖: {valid.groupby('date')['factor_value'].count().mean():.0f} 只")
    print(f"  VR_raw 均值: {df['vr_raw'].mean():.4f}")
    print(f"  VR_raw 中位数: {df['vr_raw'].median():.4f}")
    print(f"  VR_raw std: {df['vr_raw'].std():.4f}")
    print(f"  VR > 1 占比: {(df['vr_raw'] > 1).mean():.1%}")

    # 7. 输出
    print(f"\n[7] 输出 → {OUTPUT_FILE}")
    out = valid[["stock_code", "date", "factor_value"]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"  写入 {len(out):,} 行")

    print("\n✅ 完成！")


if __name__ == "__main__":
    main()
