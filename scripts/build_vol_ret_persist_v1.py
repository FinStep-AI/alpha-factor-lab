#!/usr/bin/env python3
"""
因子: 量价共振持续性 (Volume-Return Persistence) v1
factor_id: vol_ret_persist_v1

逻辑:
  过去5日的正收益率绝对值作为"趋势动量强度"，乘以过去20天中
  sign(ret)×sign(vol_chg) 同号的净天数比例(=mean(sign_align, 20d))。

  乘法合成后：同时满足"近期趋势明确(5d ret幅度)"和"量价共振一致(20d)"
  预期强信号出现在：
    - 5日已有趋势(ret放大) + 过去20日趋势方向与量变化方向高度一致

  方向: 正向（高共振持续+趋势明确 → 后续动量延续 → 高预期收益）


数据说明:
  5日收益幅度的计算绝对不能用未来数据，它已经是实时的历史5日绝对值。
"""

import numpy as np
import pandas as pd
from pathlib import Path

def main():
    data_dir = Path("data")
    print("Loading kline data...")
    df = pd.read_csv(data_dir / "csi1000_kline_raw.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)

    print(f"Stocks: {df['stock_code'].nunique()}, "
          f"Dates: {df['date'].min().date()} ~ {df['date'].max().date()}")

    # ---- 基础指标 ----
    # 收益率
    df['ret'] = df.groupby('stock_code')['close'].pct_change()

    # 成交量变化率：当日volume / 过去20日均量 - 1
    df['vol_ma20'] = df.groupby('stock_code')['volume'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    df['vol_chg'] = df['volume'] / df['vol_ma20'].replace(0, np.nan) - 1.0

    # 过去5日收益率绝对值之和（趋势幅度，已实现）
    df['abs_ret_5d'] = df.groupby('stock_code')['ret'].transform(
        lambda x: x.abs().rolling(5, min_periods=3).sum()
    )

    # ---- 20日量价净一致比例 ----
    df['sign_ret'] = np.sign(df['ret'])
    df['sign_vol'] = np.sign(df['vol_chg'])
    df['sign_align'] = df['sign_ret'] * df['sign_vol']   # +1=同向, -1=反向

    df['net_align_20d'] = df.groupby('stock_code')['sign_align'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )

    # ---- 合成因子：净一致比例 × 5日趋势幅度 ----
    raw = df['net_align_20d'] * df['abs_ret_5d']

    # 市值代理：成交额的对数
    df['log_amount'] = np.log(df['amount'].replace(0, np.nan))

    result = df[['date', 'stock_code', 'log_amount']].copy()
    result['raw_factor'] = raw.values
    result = result.dropna(subset=['raw_factor', 'log_amount']).copy()

    # ---- MAD 5% winsorize (截面) ----
    print("5% winsorization...")
    result2 = []
    for dt, g in result.groupby('date', sort=False):
        v = g['raw_factor'].values
        lo, hi = np.nanquantile(v, 0.025), np.nanquantile(v, 0.975)
        v2 = np.clip(v, lo, hi)
        g2 = g.copy(); g2['raw_factor'] = v2
        result2.append(g2)
    result = pd.concat(result2, ignore_index=True)

    # ---- 截面 z-score ----
    print("Cross-sectional z-score...")
    result['factor_zscore'] = result.groupby('date')['raw_factor'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

    # ---- 成交额OLS中性化 ----
    print("OLS neutralization on log_amount...")
    def neutralize(group):
        g = group.dropna(subset=['factor_zscore', 'log_amount'])
        if len(g) < 20:
            g = g.copy()
            g['factor_neutral'] = np.nan
            return g[['factor_neutral']]
        x = g['log_amount'].values
        y = g['factor_zscore'].values
        xm = np.nanmean(x); ym = np.nanmean(y)
        b = np.nansum((x-xm)*(y-ym)) / (np.nansum((x-xm)**2) + 1e-10)
        a = ym - b*xm
        g = g.copy()
        g['factor_neutral'] = y - (a + b*x)
        return g[['factor_neutral']]

    neu = result.groupby('date', group_keys=False).apply(neutralize)
    result['factor_neutral'] = neu['factor_neutral'].values

    # ---- 最终截面 z-score + 方向 ----
    result['factor_value'] = result.groupby('date')['factor_neutral'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    result['factor_value'] = result['factor_value'].clip(-3, 3)

    out = result[['date', 'stock_code', 'factor_value']].dropna().copy()
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')

    out_path = data_dir / "factor_vol_ret_persist_v1.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Shape: {out.shape}")
    print(f"Dates: {out['date'].min()} ~ {out['date'].max()}")
    print(out['factor_value'].describe())

if __name__ == "__main__":
    main()
