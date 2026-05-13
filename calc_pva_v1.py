#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
振荡/价格误差率因子 (Price Velocity Anomaly, pva_v1)
====================================================

定义: 过去20日中，价格动量与振幅动量的"吻合度"偏差。
      近5日价格变化率 vs. 近5日振幅变化率的偏差。

公式:
  velocity_price  = log(close_t / close_{t-5})    5日价格变化率
  velocity_amp    = log(1 + amplitude_t) - log(1 + amplitude_{t-5})
  anomaly         = 近5日均值(velocity_price) - 近5日均值(velocity_amp)

信号:
  价格变化 > 振幅变化(正 anomaly) → 价格在相对较低的振幅下上升
    → 可能存在不确定性 → 反转/修正风险
  
  振幅变化 > 价格变化(负 anomaly) → 波动上升但价格不动
    → 信息积累但尚未充分定价 → 后续价格突破机会

策略: 正anomaly → 负向 (反转), 负anomaly → 正向

最终因子:
  因子原始值 = MA20(anomaly)
  中性化    : OLS残差 ~ log_amount_20d  
  标准化    : MAD winsorize → z-score

Args:
  --input      数据目录 (默认: .)
  --output     输出因子CSV (默认: data/factor_pva_v1.csv)
  --window     回看窗口 (默认: 20)
  --price-w    价格动量窗口 (默认: 5)
  --amp-w      振幅动量窗口 (默认: 5)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from numpy.linalg import lstsq


def load_kline_data(data_dir: Path) -> pd.DataFrame:
    kline_path = data_dir / 'csi1000_kline_raw.csv'
    if not kline_path.exists():
        print(f"错误: 找不到 {kline_path}")
        sys.exit(1)
    df = pd.read_csv(kline_path, parse_dates=['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    return df


def compute_amount_20d(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df['amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(window, min_periods=10).mean()
    )
    return df


def compute_pva_raw(df: pd.DataFrame, price_w: int = 5, amp_w: int = 5, window: int = 20) -> pd.DataFrame:
    """
    计算价格速度 vs 振幅速度偏差异常
    """
    # -(变化率)
    df['price_change_ratio'] = df.groupby('stock_code')['close'].transform(
        lambda x: np.log(x / x.shift(price_w)).replace([np.inf, -np.inf], np.nan)
    )
    df['amp_change_ratio'] = df.groupby('stock_code')['amplitude'].transform(
        lambda x: (np.log(1 + x.shift(1).clip(0.01, None) + 1e-8) - 
                   np.log(1 + x.shift(amp_w + 1).clip(0.01, None) + 1e-8))
        # 5日振幅对数变(经去噪处理)
    ) if False else None  # 逐次变式: 行为太繁, 改用滚动窗口效率比

    # 改用手动改进版: 简单5日"价格vs振幅"相对变化速度
    df['price_vel_5'] = df.groupby('stock_code')['close'].transform(
        lambda x: (x - x.shift(5)) / (x.shift(5) + 1e-8)
    )
    df['amp_vel_5'] = df.groupby('stock_code')['amplitude'].transform(
        lambda x: x.rolling(6).apply(
            lambda arr: (arr[-1] - arr[0]) / (arr[0] + 1e-8), raw=False
        )
    )
    # 异常 = 价格5日归一化收益率
    df['pva_raw_daily'] = df['price_vel_5'] - df['amp_vel_5']

    # 20日滚动均值 → 最终原始值=MA20(pva_raw_daily)
    df['pva_raw'] = df.groupby('stock_code')['pva_raw_daily'].transform(
        lambda x: x.rolling(window, min_periods=10).mean()
    )
    return df


def neutralize_factor(df: pd.DataFrame, factor_col: str, neutralize_col: str) -> pd.DataFrame:
    result = []
    for dt, group in df.groupby('date'):
        g = group[[factor_col, neutralize_col, 'stock_code']].dropna()
        if len(g) < 30:
            continue
        y = g[factor_col].values.astype(float)
        x = g[neutralize_col].values.astype(float)
        X = np.column_stack([np.ones(len(x)), x])
        try:
            coeffs, _, _, _ = lstsq(X, y, rcond=None)
            residuals = y - X @ coeffs
        except:
            continue
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med)) * 1.4826
        if mad < 1e-10:
            continue
        z = (residuals - med) / mad
        z = np.clip(z, -5, 5)
        mu, sigma = np.mean(z), np.std(z)
        if sigma < 1e-10:
            continue
        z = (z - mu) / sigma
        for idx, val in zip(g.index, z):
            result.append((dt, g.loc[idx, 'stock_code'], val))
    res_df = pd.DataFrame(result, columns=['date', 'stock_code', 'factor_value'])
    print(f"中性化后有效截面数: {res_df['date'].nunique()}")
    return res_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='.', help='数据目录')
    parser.add_argument('--output', default='data/factor_pva_v1.csv', help='输出CSV')
    parser.add_argument('--window', type=int, default=20, help='回看窗口')
    parser.add_argument('--price-w', type=int, default=5, help='价格动量窗口')
    parser.add_argument('--amp-w', type=int, default=5, help='振幅动量窗口')
    args = parser.parse_args()

    data_dir = Path(args.input)
    output = Path(args.output)
    print(f"=== 价格速度异常因子 (pva_v1) ===")
    print(f"  价格窗口: {args.price_w}d  振幅窗口: {args.amp_w}d  均化: {args.window}d")

    df = load_kline_data(data_dir)
    print(f"加载: {len(df)}行, {df['stock_code'].nunique()}只股票")

    df = compute_amount_20d(df)
    df = compute_pva_raw(df, price_w=args.price_w, amp_w=args.amp_w, window=args.window)
    df['log_amount_20d'] = np.log1p(df['amount_20d'].clip(lower=1))
    df = df.dropna(subset=['pva_raw', 'log_amount_20d'])

    print(f"有效数据: {len(df)}行, {df['date'].nunique()}个截面")

    print("中性化 + MAD winsorize → z-score...")
    result = neutralize_factor(df, 'pva_raw', 'log_amount_20d')
    if result.empty:
        print("错误: 中性化后无有效数据!")
        sys.exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"保存完成: {output}")
    print(f"因子统计:\n{result['factor_value'].describe()}")


if __name__ == '__main__':
    main()
