#!/usr/bin/env python3
"""
pos_day_freq_v2 — 正向极端收益日频率因子 (v2 upgrade)

对比 neg_day_freq_v1（极端负收益日频率）的 complementary 因子：

v1: count(equity_ret <= -3%) / 10d
v2: count(equity_ret >= +3%) / 10d

再加一个 v3 不对称因子: (pos_freq - neg_freq) / 10d = net_extreme_freq

三版同批跑，比较 alpha 强度。
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────────────────────────────
WINDOW = 10          # 回看窗口（天）
THRESHOLD = 3.0      # 极端收益阈值（%）
FREQ_THRESHOLD = 0.3 # 中性化前至少有多少比例有数据
MIN_STOCKS = 50      # 截面至少多少只股票


def compute_factors(
    df: pd.DataFrame,
    window: int = WINDOW,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    """
    计算三个因子：
    - pos_day_freq_v2:  过去10日中收益率>=+3%的天数占比
    - neg_day_freq_v2:  过去10日中收益率<=-3%的天数占比(v1的更新版，校准窗口)
    - net_extreme_freq_v1: (pos - neg) / 10d

    返回: DataFrame[date, stock_code, factor_raw, factor_neutralized, factor_zscore]
    """
    df = df.sort_values(["stock_code", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ret"] = df.groupby("stock_code")["pct_change"].pct_change() * 100  # pct_change是变化pp，转percentage change

    # 极端标志
    thresh = threshold
    df["is_pos_extreme"] = (df["ret"] >= thresh).astype(float)
    df["is_neg_extreme"] = (df["ret"] <= -thresh).astype(float)

    # 滚动计数与比例
    for col in ["is_pos_extreme", "is_neg_extreme"]:
        df[f"roll_{col}"] = (
            df.groupby("stock_code")[col]
            .rolling(window, min_periods=window)
            .sum()
            .reset_index(level=0, drop=True)
        )

    df["pos_day_freq_v2"] = df["roll_is_pos_extreme"] / window
    df["neg_day_freq_v2"] = df["roll_is_neg_extreme"] / window
    df["net_extreme_freq_v1"] = (df["roll_is_pos_extreme"] - df["roll_is_neg_extreme"]) / window

    # 计算20日平均成交额
    df["amount_20d"] = (
        df.groupby("stock_code")["amount"]
        .rolling(20, min_periods=15)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # 对数转换
    df["log_amount_20d"] = np.log(df["amount_20d"].clip(lower=1))

    # ── 截面中性化 + z-score ──────────────────────────────────────────────
    factor_cols = ["pos_day_freq_v2", "neg_day_freq_v2", "net_extreme_freq_v1"]
    results = []

    dates = sorted(df["date"].unique())
    for dt in dates:
        mask = (df["date"] == dt) & (df["amount_20d"].notna())
        sub = df.loc[mask].copy()
        if len(sub) < MIN_STOCKS:
            continue

        for fcol in factor_cols:
            # 过滤NaN
            valid = sub[sub[fcol].notna()].copy()
            if len(valid) < MIN_STOCKS:
                continue

            # OLS 中性化 ~ log_amount_20d
            X = valid["log_amount_20d"].values.reshape(-1, 1)
            y = valid[fcol].values
            lr = LinearRegression().fit(X, y)
            residual = y - lr.predict(X)

            # MAD+winsorize + z-score
            med = np.median(residual)
            mad = np.median(np.abs(residual - med)) * 1.4826
            if mad < 1e-6:
                continue
            lower = med - 5.2 * mad
            upper = med + 5.2 * mad
            clipped = np.clip(residual, lower, upper)
            std = np.std(clipped)
            if std < 1e-6:
                continue
            z = (clipped - np.mean(clipped)) / std

            out = valid[["date", "stock_code"]].copy()
            out["factor_raw"] = valid[fcol].values
            out["factor_neutralized"] = z
            out["factor_id"] = fcol  # mark which factor
            results.append(out)

    if not results:
        raise RuntimeError("No factor rows generated")

    return pd.concat(results, ignore_index=True)


def save_factor(df: pd.DataFrame, factor_id: str, output_dir: str) -> None:
    """保存因子CSV和元数据"""
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)

    csv_path = od / f"factor_{factor_id}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"  → 因子值CSV: {csv_path} ({len(df)} rows)")

    # 因子元数据
    val_col = "factor_neutralized" if "factor_neutralized" in df.columns else factor_id
    scale_info = df[val_col].describe()
    meta = {
        "factor_id": factor_id,
        "n_stocks_mean": float(df.groupby("date")["stock_code"].nunique().mean()),
        "n_dates": int(df["date"].nunique()),
        "date_range": [str(df["date"].min()), str(df["date"].max())],
        "value_mean": float(scale_info["mean"]),
        "value_std": float(scale_info["std"]),
        "value_min": float(scale_info["min"]),
        "value_max": float(scale_info["max"]),
    }
    import json
    with open(od / f"meta_{factor_id}.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="极端收益日频率因子 (v2/v3)")
    parser.add_argument("--input", default="data/csi1000_kline_raw.csv", help="输入K线CSV")
    parser.add_argument("--output-dir", default="data", help="输出目录")
    parser.add_argument("--window", type=int, default=WINDOW, help="回看窗口")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="极端阈值(%)")
    args = parser.parse_args()

    t0 = time.time()
    logger.info(f"读取数据: {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"  原始: {len(df)} rows, {df['stock_code'].nunique()} stocks, 最新: {df['date'].max()}")

    logger.info("计算因子...")
    result = compute_factors(df, window=args.window, threshold=args.threshold)

    # 保存三个因子（从总表中按 factor_id 过滤）
    factor_ids = ["pos_day_freq_v2", "neg_day_freq_v2", "net_extreme_freq_v1"]
    for fid in factor_ids:
        sub = result[result["factor_id"] == fid][
            ["date", "stock_code", "factor_raw", "factor_neutralized"]
        ].copy()
        if len(sub) == 0:
            logger.warning(f"  {fid}: 无有效数据, 跳过")
            continue
        sub.columns = ["date", "stock_code", "factor_raw", fid]
        save_factor(sub, fid, args.output_dir)

    logger.info(f"\n✅ 完成 ({(time.time()-t0):.1f}s)")
    logger.info("生成3个因子: pos_day_freq_v2 / neg_day_freq_v2 / net_extreme_freq_v1")


if __name__ == "__main__":
    main()
