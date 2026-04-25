#!/usr/bin/env python3
"""
因子计算脚本: CLV-Return Direction Synchrony (clv_ret_sync_v1)

构造逻辑:
  CLV = (2*close - high - low) / (high - low + eps)   [-1, 1]
  sign(CLV) * sign(ret)                                  [-1, 1]
  factor_raw = MA20(sign(CLV)*sign(ret))                 [-1, 1]

高值含义: 收盘在高位时继续上涨 + 低位继续下跌 = 信息驱动的价格趋势
          (低位跳空, 收盘低于两三年一线)

中性化: 成交额OLS中性化 + MAD winsorize + z-score
"""

import numpy as np
import pandas as pd
import sys
import os

def calculate_factor(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)

    eps = 1e-8

    # 日收益率
    df['ret'] = df.groupby('stock_code')['close'].pct_change()

    # CLV (Close Location Value): -1 (收在最低) ~ +1 (收在最高)
    hl_range = df['high'] - df['low']
    df['clv'] = (2 * df['close'] - df['high'] - df['low']) / (hl_range + eps)
    df['clv'] = df['clv'].clip(-1, 1)  # 极端值截断

    # 方向信号: sign(CLV) * sign(ret)
    # +1: 半收又涨 (或 低結又跌) → 同向
    # -1: 半收但反向 → 反向
    df['direction_signal'] = np.sign(df['clv']) * np.sign(df['ret'])

    # 20日滚动均值 → [-1, 1], 值越大方向同步性越强
    df['factor_raw'] = df.groupby('stock_code')['direction_signal'].transform(
        lambda x: x.rolling(window=20, min_periods=10).mean()
    )

    # log_amount_20d: 20日平均成交额(对数)
    df['log_amount_20d'] = df.groupby('stock_code')['amount'].transform(
        lambda x: np.log(x.rolling(20, min_periods=10).mean() + 1)
    )

    # ---------- 中性化 ----------
    from numpy.linalg import lstsq

    def neutralize_group(g):
        y = g['factor_raw'].values.astype(float)
        x_col = 'log_amount_20d'
        x = g[x_col].values.astype(float)
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 5:
            g['factor_neutral'] = np.nan
            return g
        # OLS: add intercept
        A = np.column_stack([np.ones(mask.sum()), x[mask]])
        b, _, _, _ = lstsq(A, y[mask], rcond=None)
        resid = np.full(len(y), np.nan)
        resid[mask] = y[mask] - A @ b
        g['factor_neutral'] = resid
        return g

    df = df.groupby('date', group_keys=False).apply(neutralize_group)

    # MAD winsorize
    def mad_winsorize(series, n_mad=3.0):
        med = series.median()
        mad = (series - med).abs().median()
        if mad == 0 or np.isnan(mad):
            return series
        scaled = 1.4826 * mad
        return series.clip(med - n_mad * scaled, med + n_mad * scaled)

    df['factor_neutral'] = df.groupby('date')['factor_neutral'].transform(
        lambda x: mad_winsorize(x, 3.0)
    )

    # z-score
    def zscore_group(g):
        v = g['factor_neutral']
        valid = v.notna()
        if valid.sum() < 3:
            g['factor'] = np.nan
            return g
        m, s = v[valid].mean(), v[valid].std()
        if s == 0:
            g['factor'] = 0.0
            return g
        g['factor'] = (v - m) / s
        return g

    df = df.groupby('date', group_keys=False).apply(zscore_group)

    # 输出
    out = df[['date', 'stock_code', 'factor']].dropna(subset=['factor'])
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    out.to_csv(output_path, index=False)

    print(f"✅ 因子 saved to {output_path}")
    print(f"   rows: {len(out)}, stocks: {out['stock_code'].nunique()}, dates: {out['date'].nunique()}")
    print(f"   factor stats: mean={out['factor'].mean():.4f} std={out['factor'].std():.4f}")
    print(f"   factor range: [{out['factor'].min():.4f}, {out['factor'].max():.4f}]")

if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(base)  # alpha-factor-lab
    input_path = os.path.join(proj, 'data', 'csi1000_kline_raw.csv')
    output_path = os.path.join(proj, 'data', 'factor_clv_ret_sync_v1.csv')
    calculate_factor(input_path, output_path)
