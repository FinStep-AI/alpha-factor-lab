#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高量日收盘强度因子 (High-Volume Close Strength, hvcs_v1)
================================================================

定义: 过去20日中，成交量>MA20×1.1的交易日，收盘价在日内位置CLV的均值。
      CLV = (close-low)/(high-low+1e-8) ∈ [0,1]，越接近1表示收在日内高位。

信号逻辑:
  放量日收在高位 = 主动买入(taker buy dominant)
  放量日收在低位 = 主动卖出(taker sell dominant)
  只用放量日(过滤低量噪音)→信号更纯粹

最终因子:
  因子原始值 = MA20(hvcs_raw)
  中性化    : OLS残差 ~ log_amount_20d
  标准化    : MAD winsorize → z-score

Args:
  --input      数据目录 (默认: .)
  --output     输出因子CSV (默认: data/factor_hvcs_v1.csv)
  --vol-thresh 高量阈值倍数 (默认: 1.1, MA20×1.1)
  --window     回看窗口 (默认: 20)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from numpy.linalg import lstsq


def load_kline_data(data_dir: Path) -> pd.DataFrame:
    """加载K线数据"""
    kline_path = data_dir / 'csi1000_kline_raw.csv'
    if not kline_path.exists():
        print(f"错误: 找不到 {kline_path}")
        sys.exit(1)
    df = pd.read_csv(kline_path, parse_dates=['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    return df


def compute_amount_20d(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """计算20日平均成交额"""
    df['amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: x.rolling(window, min_periods=10).mean()
    )
    return df


def compute_volume_ma20(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """计算20日平均成交量"""
    df['vol_ma20'] = df.groupby('stock_code')['volume'].transform(
        lambda x: x.rolling(window, min_periods=10).mean()
    )
    return df


def compute_clv(df: pd.DataFrame) -> pd.DataFrame:
    """计算收盘位置CLV = (close-low)/(high-low)"""
    df['clv'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    return df


def compute_hvcs_raw(df: pd.DataFrame, vol_thresh: float = 1.1, window: int = 20) -> pd.DataFrame:
    """
    高量日CLV均值: 只在成交量>MA20×vol_thresh时取CLV，滚动窗口均值。
    """
    # 高量日标记
    df['is_high_vol'] = (df['volume'] > df['vol_ma20'] * vol_thresh).astype(float)
    # 只在放量日取CLV，其余为NaN
    df['hvcs_contrib'] = np.where(df['is_high_vol'] == 1.0, df['clv'], np.nan)
    # 滚动窗口均值
    df['hvcs_raw'] = df.groupby('stock_code')['hvcs_contrib'].transform(
        lambda x: x.rolling(window, min_periods=5).mean()
    )
    return df


def neutralize_factor(df: pd.DataFrame, factor_col: str, neutralize_col: str) -> pd.DataFrame:
    """
    成交额OLS中性化 + MAD去极值 + z-score
    """
    result = []
    for dt, group in df.groupby('date'):
        g = group[[factor_col, neutralize_col, 'stock_code']].dropna()
        if len(g) < 30:
            continue
        y = g[factor_col].values.astype(float)
        x = g[neutralize_col].values.astype(float)
        # OLS: y = alpha + beta * x + residual
        X = np.column_stack([np.ones(len(x)), x])
        try:
            coeffs, _, _, _ = lstsq(X, y, rcond=None)
            residuals = y - X @ coeffs
        except:
            continue
        # MAD winsorize → sigma estimator
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med)) * 1.4826
        if mad < 1e-10:
            continue
        z = (residuals - med) / mad
        z = np.clip(z, -5, 5)
        # z-score
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
    parser.add_argument('--output', default='data/factor_hvcs_v1.csv', help='输出CSV')
    parser.add_argument('--vol-thresh', type=float, default=1.1, help='高量阈值倍数')
    parser.add_argument('--window', type=int, default=20, help='回看窗口')
    args = parser.parse_args()

    data_dir = Path(args.input)
    output = Path(args.output)
    print(f"=== 高量日收盘强度因子 (hvcs_v1) ===")
    print(f"  高量阈值: MA20×{args.vol_thresh}")
    print(f"  回望窗口: {args.window}日")

    # 1. 加载数据
    df = load_kline_data(data_dir)
    print(f"加载数据: {len(df)}行, {df['stock_code'].nunique()}只股票")

    # 2. 特征计算
    df = compute_amount_20d(df)
    df = compute_volume_ma20(df)
    df = compute_clv(df)
    df = compute_hvcs_raw(df, vol_thresh=args.vol_thresh, window=args.window)
    df['log_amount_20d'] = np.log1p(df['amount_20d'].clip(lower=1))

    # 有效过滤
    df = df.dropna(subset=['hvcs_raw', 'log_amount_20d'])
    print(f"有效数据: {len(df)}行, {df['date'].nunique()}个可截面")

    # 3. 中性化 + MAD + z-score
    print("OLS中性化 + MAD winsorize → z-score...")
    result = neutralize_factor(df, 'hvcs_raw', 'log_amount_20d')

    if result.empty:
        print("错误: 中性化后无有效数据!")
        sys.exit(1)

    # 4. 保存
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"因子值已保存: {output}")
    print(f"因子统计:\n{result['factor_value'].describe()}")

    return result


if __name__ == '__main__':
    main()
